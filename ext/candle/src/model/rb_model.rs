#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use crate::model::{
    errors::{wrap_candle_err, wrap_hf_err, wrap_std_err},
    rb_tensor::RbTensor,
};
use crate::model::rb_device::RbDevice;
use candle_core::{DType, Device, Module, Tensor, quantized::ggml_file::Content};
use safetensors::tensor::SafeTensors;
use candle_nn::VarBuilder;
use candle_transformers::models::{
    bert::{BertModel as StdBertModel, Config as BertConfig},
    jina_bert::{BertModel as JinaBertModel, Config as JinaConfig},
    quantized_llama::ModelWeights
};
use magnus::Error;
use crate::model::RbResult;
use std::sync::Arc;
use std::fs::File;
use std::path::Path;
use tokenizers::Tokenizer;
use serde_json;
use magnus::{value::ReprValue, TryConvert};

#[magnus::wrap(class = "Candle::Model", free_immediately, size)]
pub struct RbModel(pub RbModelInner);

/// Supported model types for embedding generation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    JinaBert,
    StandardBert,
    MiniLM,
    Llama
}

impl ModelType {
    pub fn from_string(model_type: &str) -> Option<Self> {
        match model_type.to_lowercase().as_str() {
            "jina_bert" | "jinabert" | "jina" => Some(ModelType::JinaBert),
            "bert" | "standard_bert" | "standardbert" => Some(ModelType::StandardBert),
            "minilm" => Some(ModelType::MiniLM),
            "llama" => Some(ModelType::Llama),
            _ => None
        }
    }
}

/// Model variants that can produce embeddings
pub enum ModelVariant {
    JinaBert(JinaBertModel),
    StandardBert(StdBertModel),
    MiniLM(StdBertModel),
    Llama(Arc<ModelWeights>),
}

impl ModelVariant {
    pub fn model_type(&self) -> ModelType {
        match self {
            ModelVariant::JinaBert(_) => ModelType::JinaBert,
            ModelVariant::StandardBert(_) => ModelType::StandardBert,
            ModelVariant::MiniLM(_) => ModelType::MiniLM,
            ModelVariant::Llama(_) => ModelType::Llama,
        }
    }
}

pub struct RbModelInner {
    device: Device,
    tokenizer_path: Option<String>,
    model_path: Option<String>,
    model_type: Option<ModelType>,
    model: Option<ModelVariant>,
    tokenizer: Option<Tokenizer>,
    embedding_size: Option<usize>,
}

impl RbModel {
    pub fn new(model_path: Option<String>, tokenizer_path: Option<String>, device: Option<RbDevice>, model_type: Option<String>, embedding_size: Option<usize>) -> RbResult<Self> {
        let device = device.unwrap_or(RbDevice::Cpu).as_device()?;
        let model_type = model_type
            .and_then(|mt| ModelType::from_string(&mt))
            .unwrap_or(ModelType::JinaBert);
        Ok(RbModel(RbModelInner {
            device: device.clone(),
            model_path: model_path.clone(),
            tokenizer_path: tokenizer_path.clone(),
            model_type: Some(model_type),
            model: match model_path {
                Some(mp) => Some(Self::build_model(Path::new(&mp), device, model_type, embedding_size)?),
                None => None
            },
            tokenizer: match tokenizer_path {
                Some(tp) => Some(Self::build_tokenizer(tp)?),
                None => None
            },
            embedding_size,
        }))
    }

    /// Generates an embedding vector for the input text
    /// &RETURNS&: Tensor
    pub fn embedding(&self, input: String) -> RbResult<RbTensor> {
        match &self.0.model {
            Some(model) => {
                match &self.0.tokenizer {
                    Some(tokenizer) => Ok(RbTensor(self.compute_embedding(input, model, tokenizer)?)),
                    None => Err(magnus::Error::new(magnus::exception::runtime_error(), "Tokenizer not found"))
                }
            }
            None => Err(magnus::Error::new(magnus::exception::runtime_error(), "Model not found"))
        }
    }

    /// Infers and validates the embedding size from a safetensors file
    fn resolve_embedding_size(model_path: &Path, embedding_size: Option<usize>) -> Result<usize, magnus::Error> {
        let inferred_emb_dim = match SafeTensors::deserialize(&std::fs::read(model_path).map_err(|e| wrap_std_err(Box::new(e)))?) {
            Ok(st) => {
                if let Some(tensor) = st.tensor("embeddings.word_embeddings.weight").ok() {
                    let shape = tensor.shape();
                    if shape.len() == 2 { Some(shape[1] as usize) } else { None }
                } else { None }
            },
            Err(_) => None
        };
        match embedding_size {
            Some(user_dim) => {
                if let Some(inferred) = inferred_emb_dim {
                    if user_dim != inferred {
                        return Err(magnus::Error::new(magnus::exception::runtime_error(), format!("User-specified embedding_size {} does not match model file's embedding size {}", user_dim, inferred)));
                    }
                }
                Ok(user_dim)
            },
            None => inferred_emb_dim.ok_or_else(|| magnus::Error::new(magnus::exception::runtime_error(), "Could not infer embedding size from model file. Please specify embedding_size explicitly."))
        }
    }

    fn build_model(model_path: &Path, device: Device, model_type: ModelType, embedding_size: Option<usize>) -> RbResult<ModelVariant> {
        use hf_hub::{api::sync::Api, Repo, RepoType};
        let api = Api::new().map_err(wrap_hf_err)?;
        let repo = Repo::new(model_path.to_str().unwrap().to_string(), RepoType::Model);
        match model_type {
            ModelType::JinaBert => {
                let model_path = api.repo(repo).get("model.safetensors").map_err(wrap_hf_err)?;
                let final_emb_dim = Self::resolve_embedding_size(Path::new(&model_path), embedding_size)?;
                let mut config = JinaConfig::v2_base();
                config.hidden_size = final_emb_dim;
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)
                        .map_err(wrap_candle_err)?
                };
                let model = JinaBertModel::new(vb, &config).map_err(wrap_candle_err)?;
                Ok(ModelVariant::JinaBert(model))
            },
            ModelType::StandardBert => {
                let model_path = api.repo(repo).get("model.safetensors").map_err(wrap_hf_err)?;
                let final_emb_dim = Self::resolve_embedding_size(Path::new(&model_path), embedding_size)?;
                let mut config = BertConfig::default();
                config.hidden_size = final_emb_dim;
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)
                        .map_err(wrap_candle_err)?
                };
                let model = StdBertModel::load(vb, &config).map_err(wrap_candle_err)?;
                Ok(ModelVariant::StandardBert(model))
            },
            ModelType::MiniLM => {
                let model_path = api.repo(repo.clone()).get("model.safetensors").map_err(wrap_hf_err)?;
                let config_path = api.repo(repo).get("config.json").map_err(wrap_hf_err)?;
                let config_file = std::fs::File::open(&config_path).map_err(|e| wrap_std_err(Box::new(e)))?;
                let mut config: BertConfig = serde_json::from_reader(config_file).map_err(|e| wrap_std_err(Box::new(e)))?;
                if let Some(embedding_size) = embedding_size {
                    config.hidden_size = embedding_size;
                }
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)
                        .map_err(wrap_candle_err)?
                };
                let model = StdBertModel::load(vb, &config).map_err(wrap_candle_err)?;
                Ok(ModelVariant::MiniLM(model))
            },
            ModelType::Llama => {
                let model_path = api.repo(repo).get("model.ggml").map_err(wrap_hf_err)?;
                let mut file = File::open(&model_path).map_err(|e| wrap_std_err(Box::new(e)))?;
                let ct = Content::read(&mut file, &device).map_err(wrap_candle_err)?;
                let model = ModelWeights::from_ggml(ct, 1).map_err(wrap_candle_err)?;
                Ok(ModelVariant::Llama(Arc::new(model)))
            }
        }
    }

    fn build_tokenizer(tokenizer_path: String) -> RbResult<Tokenizer> {
        use hf_hub::{api::sync::Api, Repo, RepoType};
        let tokenizer_path = Api::new()
                .map_err(wrap_hf_err)?
                .repo(Repo::new(
                    tokenizer_path,
                    RepoType::Model,
                ))
                .get("tokenizer.json")
                .map_err(wrap_hf_err)?;
        let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(wrap_std_err)?;
        let pp = tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));

        Ok(tokenizer)
    }

    fn pooled_normalized_embedding(result: &Tensor) -> Result<Tensor, Error> {
        let (_n_sentence, n_tokens, _hidden_size) = result.dims3().map_err(wrap_candle_err)?;
        let sum = result.sum(1).map_err(wrap_candle_err)?;
        let mean = (sum / (n_tokens as f64)).map_err(wrap_candle_err)?;
        let norm = Self::normalize_l2(&mean).map_err(wrap_candle_err)?;
        Ok(norm)
    }

    fn compute_embedding(
        &self,
        prompt: String,
        model: &ModelVariant,
        tokenizer: &Tokenizer,
    ) -> Result<Tensor, Error> {
        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(wrap_std_err)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], &self.0.device)
            .map_err(wrap_candle_err)?
            .unsqueeze(0)
            .map_err(wrap_candle_err)?;
        let batch_size = token_ids.dims()[0];
        let seq_len = token_ids.dims()[1];
        let token_type_ids = Tensor::zeros(&[batch_size, seq_len], DType::U32, &self.0.device)
            .map_err(wrap_candle_err)?;
        let attention_mask = Tensor::ones(&[batch_size, seq_len], DType::U32, &self.0.device)
            .map_err(wrap_candle_err)?;
        match model {
            ModelVariant::JinaBert(model) => {
                let result = model.forward(&token_ids).map_err(wrap_candle_err)?;
                Self::pooled_normalized_embedding(&result)
            },
            ModelVariant::StandardBert(model) => {
                let result = model.forward(&token_ids, &token_type_ids, Some(&attention_mask)).map_err(wrap_candle_err)?;
                Self::pooled_normalized_embedding(&result)
            },
            ModelVariant::MiniLM(model) => {
                let result = model.forward(&token_ids, &token_type_ids, Some(&attention_mask)).map_err(wrap_candle_err)?;
                Self::pooled_normalized_embedding(&result)
            },
            ModelVariant::Llama(_) => {
                Err(Error::new(magnus::exception::runtime_error(), "Llama embedding not implemented for quantized model"))
            }
        }
    }

    #[allow(dead_code)]
    fn normalize_l2(v: &Tensor) -> Result<Tensor, candle_core::Error> {
        v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
    }

    pub fn model_type(&self) -> String {
        match self.0.model_type {
            Some(model_type) => format!("{:?}", model_type),
            None => "nil".to_string(),
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "#<Candle::Model model_type: {}, model_path: {}, tokenizer_path: {}, embedding_size: {}>", 
            self.model_type(), 
            self.0.model_path.as_deref().unwrap_or("nil"), 
            self.0.tokenizer_path.as_deref().unwrap_or("nil"),
            self.0.embedding_size.map(|x| x.to_string()).unwrap_or("nil".to_string())
        )
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}
