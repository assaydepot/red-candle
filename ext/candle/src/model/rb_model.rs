#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use crate::model::{
    errors::{wrap_candle_err, wrap_hf_err, wrap_std_err},
    rb_tensor::RbTensor,
};
use candle_core::{DType, Device, Module, Tensor, quantized::ggml_file::Content};
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
use tokenizers::Tokenizer;

#[magnus::wrap(class = "Candle::Model", free_immediately, size)]
pub struct RbModel(pub RbModelInner);

/// Supported model types for embedding generation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    JinaBert,
    StandardBert,
    Llama
}

impl ModelType {
    pub fn from_string(model_type: &str) -> Option<Self> {
        match model_type.to_lowercase().as_str() {
            "jina_bert" | "jinabert" | "jina" => Some(ModelType::JinaBert),
            "bert" | "standard_bert" | "standardbert" => Some(ModelType::StandardBert),
            "llama" => Some(ModelType::Llama),
            _ => None
        }
    }
}

/// Model variants that can produce embeddings
pub enum ModelVariant {
    JinaBert(JinaBertModel),
    StandardBert(StdBertModel),
    Llama(Arc<ModelWeights>),
}

impl ModelVariant {
    pub fn model_type(&self) -> ModelType {
        match self {
            ModelVariant::JinaBert(_) => ModelType::JinaBert,
            ModelVariant::StandardBert(_) => ModelType::StandardBert,
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
}

impl RbModel {
    pub fn new() -> RbResult<Self> {
        Self::new2(Some("jinaai/jina-embeddings-v2-base-en".to_string()), Some("sentence-transformers/all-MiniLM-L6-v2".to_string()), Some(Device::Cpu), Some("jina_bert".to_string()))
    }

    pub fn new2(model_path: Option<String>, tokenizer_path: Option<String>, device: Option<Device>, model_type: Option<String>) -> RbResult<Self> {
        let device = device.unwrap_or(Device::Cpu);
        let model_type = model_type
            .and_then(|mt| ModelType::from_string(&mt))
            .unwrap_or(ModelType::JinaBert);
        
        Ok(RbModel(RbModelInner {
            device: device.clone(),
            model_path: model_path.clone(),
            tokenizer_path: tokenizer_path.clone(),
            model_type: Some(model_type),
            model: match model_path {
                Some(mp) => Some(Self::build_model(mp, device, model_type)?),
                None => None
            },
            tokenizer: match tokenizer_path {
                Some(tp) => Some(Self::build_tokenizer(tp)?),
                None => None
            }
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

    fn build_model(model_path: String, device: Device, model_type: ModelType) -> RbResult<ModelVariant> {
        use hf_hub::{api::sync::Api, Repo, RepoType};
        let api = Api::new().map_err(wrap_hf_err)?;
        let repo = Repo::new(model_path, RepoType::Model);
        match model_type {
            ModelType::JinaBert => {
                let model_path = api.repo(repo).get("model.safetensors").map_err(wrap_hf_err)?;
                let config = JinaConfig::v2_base();
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)
                        .map_err(wrap_candle_err)?
                };
                let model = JinaBertModel::new(vb, &config).map_err(wrap_candle_err)?;
                Ok(ModelVariant::JinaBert(model))
            },
            ModelType::StandardBert => {
                let model_path = api.repo(repo).get("model.safetensors").map_err(wrap_hf_err)?;
                let config = BertConfig::default();
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)
                        .map_err(wrap_candle_err)?
                };
                let model = StdBertModel::load(vb, &config).map_err(wrap_candle_err)?;
                Ok(ModelVariant::StandardBert(model))
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
                Ok(result)
            },
            ModelVariant::StandardBert(model) => {
                let result = model.forward(&token_ids, &token_type_ids, Some(&attention_mask)).map_err(wrap_candle_err)?;
                Ok(result)
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
            "#<Candle::Model model_type: {}, model_path: {}, tokenizer_path: {}>", 
            self.model_type(), 
            self.0.model_path.as_deref().unwrap_or("nil"), 
            self.0.tokenizer_path.as_deref().unwrap_or("nil")
        )
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}
