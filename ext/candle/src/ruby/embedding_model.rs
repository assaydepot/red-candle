// MKL and Accelerate are handled by candle-core when their features are enabled

use crate::ruby::{
    errors::{wrap_candle_err, wrap_hf_err, wrap_std_err},
};
use crate::ruby::{Tensor, Device, Result};
use crate::tokenizer::{TokenizerWrapper, loader::TokenizerLoader};
use candle_core::{DType as CoreDType, Device as CoreDevice, Module, Tensor as CoreTensor};
use safetensors::tensor::SafeTensors;
use candle_nn::VarBuilder;
use candle_transformers::models::{
    bert::{BertModel as StdBertModel, Config as BertConfig},
    jina_bert::{BertModel as JinaBertModel, Config as JinaConfig},
    distilbert::{DistilBertModel, Config as DistilBertConfig}
};
use magnus::{class, function, method, prelude::*, Error, RModule, RHash};
use std::path::Path;
use serde_json;


#[magnus::wrap(class = "Candle::EmbeddingModel", free_immediately, size)]
pub struct EmbeddingModel(pub EmbeddingModelInner);

/// Supported model types for embedding generation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmbeddingModelType {
    JinaBert,
    StandardBert,
    DistilBert,
    MiniLM,
}

impl EmbeddingModelType {
    pub fn from_string(model_type: &str) -> Option<Self> {
        match model_type.to_lowercase().as_str() {
            "jina_bert" | "jinabert" | "jina" => Some(EmbeddingModelType::JinaBert),
            "bert" | "standard_bert" | "standardbert" => Some(EmbeddingModelType::StandardBert),
            "minilm" => Some(EmbeddingModelType::MiniLM),
    
            "distilbert" => Some(EmbeddingModelType::DistilBert),
            _ => None
        }
    }
}

/// Model variants that can produce embeddings
pub enum EmbeddingModelVariant {
    JinaBert(JinaBertModel),
    StandardBert(StdBertModel),
    DistilBert(DistilBertModel),
    MiniLM(StdBertModel),

}

impl EmbeddingModelVariant {
    pub fn model_type(&self) -> EmbeddingModelType {
        match self {
            EmbeddingModelVariant::JinaBert(_) => EmbeddingModelType::JinaBert,
            EmbeddingModelVariant::StandardBert(_) => EmbeddingModelType::StandardBert,
            EmbeddingModelVariant::DistilBert(_) => EmbeddingModelType::DistilBert,
            EmbeddingModelVariant::MiniLM(_) => EmbeddingModelType::MiniLM,
    
        }
    }
}

pub struct EmbeddingModelInner {
    device: CoreDevice,
    tokenizer_id: Option<String>,
    model_id: Option<String>,
    model_type: Option<EmbeddingModelType>,
    model: Option<EmbeddingModelVariant>,
    tokenizer: Option<TokenizerWrapper>,
    embedding_size: Option<usize>,
}

impl EmbeddingModel {
    pub fn new(model_id: Option<String>, tokenizer: Option<String>, device: Option<Device>, model_type: Option<String>, embedding_size: Option<usize>) -> Result<Self> {
        let device = device.unwrap_or(Device::best()).as_device()?;
        let model_type = model_type
            .and_then(|mt| EmbeddingModelType::from_string(&mt))
            .unwrap_or(EmbeddingModelType::JinaBert);
        Ok(EmbeddingModel(EmbeddingModelInner {
            device: device.clone(),
            model_id: model_id.clone(),
            tokenizer_id: tokenizer.clone(),
            model_type: Some(model_type),
            model: match model_id.as_ref() {
                Some(id) => Some(Self::build_embedding_model(id, device, model_type, embedding_size)?),
                None => None
            },
            tokenizer: match tokenizer {
                Some(tid) => Some(Self::build_tokenizer(tid)?),
                None => None
            },
            embedding_size,
        }))
    }

    /// Generates an embedding vector for the input text
    /// &RETURNS&: Tensor
    /// Generates an embedding vector for the input text using the specified pooling method.
    /// &RETURNS&: Tensor
    /// pooling_method: "pooled", "pooled_normalized", or "cls" (default: "pooled")
    pub fn embedding(&self, input: String, pooling_method: String) -> Result<Tensor> {
        match &self.0.model {
            Some(model) => {
                match &self.0.tokenizer {
                    Some(tokenizer) => Ok(Tensor(self.compute_embedding(input, model, tokenizer, &pooling_method)?)),
                    None => Err(magnus::Error::new(magnus::exception::runtime_error(), "Tokenizer not found"))
                }
            }
            None => Err(magnus::Error::new(magnus::exception::runtime_error(), "Model not found"))
        }
    }

    /// Returns the unpooled embedding tensor ([1, SEQLENGTH, DIM]) for the input text
    /// &RETURNS&: Tensor
    pub fn embeddings(&self, input: String) -> Result<Tensor> {
        match &self.0.model {
            Some(model) => {
                match &self.0.tokenizer {
                    Some(tokenizer) => Ok(Tensor(self.compute_embeddings(input, model, tokenizer)?)),
                    None => Err(magnus::Error::new(magnus::exception::runtime_error(), "Tokenizer not found"))
                }
            }
            None => Err(magnus::Error::new(magnus::exception::runtime_error(), "Model not found"))
        }
    }

    /// Pools and normalizes a sequence embedding tensor ([1, SEQLENGTH, DIM]) to [1, DIM]
    /// &RETURNS&: Tensor
    pub fn pool_embedding(&self, tensor: &Tensor) -> Result<Tensor> {
        let pooled = Self::pooled_embedding(&tensor.0)?;
        Ok(Tensor(pooled))
    }

    /// Pools and normalizes a sequence embedding tensor ([1, SEQLENGTH, DIM]) to [1, DIM]
    /// &RETURNS&: Tensor
    pub fn pool_and_normalize_embedding(&self, tensor: &Tensor) -> Result<Tensor> {
        let pooled = Self::pooled_normalized_embedding(&tensor.0)?;
        Ok(Tensor(pooled))
    }

    /// Pools the embedding tensor by extracting the CLS token ([1, SEQLENGTH, DIM] -> [1, DIM])
    /// &RETURNS&: Tensor
    pub fn pool_cls_embedding(&self, tensor: &Tensor) -> Result<Tensor> {
        let pooled = Self::pooled_cls_embedding(&tensor.0)?;
        Ok(Tensor(pooled))
    }

    /// Infers and validates the embedding size from a safetensors file
    fn resolve_embedding_size(model_path: &Path, embedding_size: Option<usize>) -> std::result::Result<usize, magnus::Error> {
        match embedding_size {
            Some(user_dim) => {
                Ok(user_dim)
            },
            None => {
                let inferred_emb_dim = match SafeTensors::deserialize(&std::fs::read(model_path).map_err(|e| wrap_std_err(Box::new(e)))?) {
                    Ok(st) => {
                        if let Some(tensor) = st.tensor("embeddings.word_embeddings.weight").ok() {
                            let shape = tensor.shape();
                            if shape.len() == 2 { Some(shape[1] as usize) } else { None }
                        } else { None }
                    },
                    Err(_) => None
                };
                inferred_emb_dim.ok_or_else(|| magnus::Error::new(magnus::exception::runtime_error(), "Could not infer embedding size from model file. Please specify embedding_size explicitly."))
            }
        }
    }

    fn build_embedding_model(model_id: &str, device: CoreDevice, model_type: EmbeddingModelType, embedding_size: Option<usize>) -> Result<EmbeddingModelVariant> {
        use hf_hub::{api::sync::Api, Repo, RepoType};
        let api = Api::new().map_err(wrap_hf_err)?;
        let repo = Repo::new(model_id.to_string(), RepoType::Model);
        match model_type {
            EmbeddingModelType::JinaBert => {
                let model_path = api.repo(repo).get("model.safetensors").map_err(wrap_hf_err)?;
                if !std::path::Path::new(&model_path).exists() {
                    return Err(magnus::Error::new(
                        magnus::exception::runtime_error(),
                        "model.safetensors not found after download. Only safetensors models are supported. Please ensure your model repo contains model.safetensors."
                    ));
                }
                let final_emb_dim = Self::resolve_embedding_size(Path::new(&model_path), embedding_size)?;
                let mut config = JinaConfig::v2_base();
                config.hidden_size = final_emb_dim;
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(&[model_path], CoreDType::F32, &device)
                        .map_err(wrap_candle_err)?
                };
                let model = JinaBertModel::new(vb, &config).map_err(wrap_candle_err)?;
                Ok(EmbeddingModelVariant::JinaBert(model))
            },
            EmbeddingModelType::StandardBert => {
                let model_path = api.repo(repo).get("model.safetensors").map_err(wrap_hf_err)?;
                if !std::path::Path::new(&model_path).exists() {
                    return Err(magnus::Error::new(
                        magnus::exception::runtime_error(),
                        "model.safetensors not found after download. Only safetensors models are supported. Please ensure your model repo contains model.safetensors."
                    ));
                }
                let final_emb_dim = Self::resolve_embedding_size(Path::new(&model_path), embedding_size)?;
                let mut config = BertConfig::default();
                config.hidden_size = final_emb_dim;
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(&[model_path], CoreDType::F32, &device)
                        .map_err(wrap_candle_err)?
                };
                let model = StdBertModel::load(vb, &config).map_err(wrap_candle_err)?;
                Ok(EmbeddingModelVariant::StandardBert(model))
            },
            EmbeddingModelType::DistilBert => {
                let model_path = api.repo(repo.clone()).get("model.safetensors").map_err(wrap_hf_err)?;
                if !std::path::Path::new(&model_path).exists() {
                    return Err(magnus::Error::new(
                        magnus::exception::runtime_error(),
                        "model.safetensors not found after download. Only safetensors models are supported. Please ensure your model repo contains model.safetensors."
                    ));
                }
                let config_path = api.repo(repo.clone()).get("config.json").map_err(wrap_hf_err)?;
                let config_file = std::fs::File::open(&config_path).map_err(|e| wrap_std_err(Box::new(e)))?;
                let mut config: DistilBertConfig = serde_json::from_reader(config_file).map_err(|e| wrap_std_err(Box::new(e)))?;
                if let Some(embedding_size) = embedding_size {
                    config.dim = embedding_size;
                }
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(&[model_path], CoreDType::F32, &device)
                        .map_err(wrap_candle_err)?
                };
                let model = DistilBertModel::load(vb, &config).map_err(wrap_candle_err)?;
                Ok(EmbeddingModelVariant::DistilBert(model))
            },
            EmbeddingModelType::MiniLM => {
                let model_path = api.repo(repo.clone()).get("model.safetensors").map_err(wrap_hf_err)?;
                if !std::path::Path::new(&model_path).exists() {
                    return Err(magnus::Error::new(
                        magnus::exception::runtime_error(),
                        "model.safetensors not found after download. Only safetensors models are supported. Please ensure your model repo contains model.safetensors."
                    ));
                }
                let config_path = api.repo(repo.clone()).get("config.json").map_err(wrap_hf_err)?;
                let config_file = std::fs::File::open(&config_path).map_err(|e| wrap_std_err(Box::new(e)))?;
                let mut config: BertConfig = serde_json::from_reader(config_file).map_err(|e| wrap_std_err(Box::new(e)))?;
                if let Some(embedding_size) = embedding_size {
                    config.hidden_size = embedding_size;
                }
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(&[model_path], CoreDType::F32, &device)
                        .map_err(wrap_candle_err)?
                };
                let model = StdBertModel::load(vb, &config).map_err(wrap_candle_err)?;
                Ok(EmbeddingModelVariant::MiniLM(model))
            },

        }
    }

    fn build_tokenizer(tokenizer_id: String) -> Result<TokenizerWrapper> {
        use hf_hub::{api::sync::Api, Repo, RepoType};
        let tokenizer_path = Api::new()
                .map_err(wrap_hf_err)?
                .repo(Repo::new(
                    tokenizer_id,
                    RepoType::Model,
                ))
                .get("tokenizer.json")
                .map_err(wrap_hf_err)?;
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(wrap_std_err)?;
        
        let tokenizer = TokenizerLoader::with_padding(tokenizer, None);
        Ok(TokenizerWrapper::new(tokenizer))
    }

    /// Pools the embedding tensor by extracting the CLS token ([1, SEQLENGTH, DIM] -> [1, DIM])
    /// &RETURNS&: Tensor
    fn pooled_cls_embedding(result: &CoreTensor) -> std::result::Result<CoreTensor, Error> {
        // 1) sanity-check that we have a 3D tensor
        let (_batch, _seq_len, _hidden_size) = result.dims3().map_err(wrap_candle_err)?;
    
        // 2) slice out just the first token (CLS) along the sequence axis:
        //    [B, seq_len, H] → [B, 1, H]
        let first = result
            .narrow(1, 0, 1)
            .map_err(wrap_candle_err)?;
    
        // 3) remove that length-1 axis → [B, H]
        let cls = first
            .squeeze(1)
            .map_err(wrap_candle_err)?;
    
        Ok(cls)
    }

    fn pooled_embedding(result: &CoreTensor) -> std::result::Result<CoreTensor, Error> {
        let (_n_sentence, n_tokens, _hidden_size) = result.dims3().map_err(wrap_candle_err)?;
        let sum = result.sum(1).map_err(wrap_candle_err)?;
        let mean = (sum / (n_tokens as f64)).map_err(wrap_candle_err)?;
        Ok(mean)
    }

    fn pooled_normalized_embedding(result: &CoreTensor) -> std::result::Result<CoreTensor, Error> {
        let mean = Self::pooled_embedding(result)?;
        let norm = Self::normalize_l2(&mean).map_err(wrap_candle_err)?;
        Ok(norm)
    }

    fn compute_embeddings(
        &self,
        prompt: String,
        model: &EmbeddingModelVariant,
        tokenizer: &TokenizerWrapper,
    ) -> std::result::Result<CoreTensor, Error> {
        let tokens = tokenizer
            .encode(&prompt, true)
            .map_err(wrap_candle_err)?;
        let token_ids = CoreTensor::new(&tokens[..], &self.0.device)
            .map_err(wrap_candle_err)?
            .unsqueeze(0)
            .map_err(wrap_candle_err)?;
        let batch_size = token_ids.dims()[0];
        let seq_len = token_ids.dims()[1];
        let token_type_ids = CoreTensor::zeros(&[batch_size, seq_len], CoreDType::U32, &self.0.device)
            .map_err(wrap_candle_err)?;
        let attention_mask = CoreTensor::ones(&[batch_size, seq_len], CoreDType::U32, &self.0.device)
            .map_err(wrap_candle_err)?;
        match model {
            EmbeddingModelVariant::JinaBert(model) => {
                model.forward(&token_ids).map_err(wrap_candle_err)
            },
            EmbeddingModelVariant::StandardBert(model) => {
                model.forward(&token_ids, &token_type_ids, Some(&attention_mask)).map_err(wrap_candle_err)
            },
            EmbeddingModelVariant::DistilBert(model) => {
                model.forward(&token_ids, &attention_mask).map_err(wrap_candle_err)
            },
            EmbeddingModelVariant::MiniLM(model) => {
                model.forward(&token_ids, &token_type_ids, Some(&attention_mask)).map_err(wrap_candle_err)
            },

        }
    }

    /// Computes an embedding for the prompt using the specified pooling method.
    /// pooling_method: "pooled", "pooled_normalized", or "cls"
    fn compute_embedding(
        &self,
        prompt: String,
        model: &EmbeddingModelVariant,
        tokenizer: &TokenizerWrapper,
        pooling_method: &str,
    ) -> std::result::Result<CoreTensor, Error> {
        let result = self.compute_embeddings(prompt, model, tokenizer)?;
        match pooling_method {
            "pooled" => Self::pooled_embedding(&result),
            "pooled_normalized" => Self::pooled_normalized_embedding(&result),
            "cls" => Self::pooled_cls_embedding(&result),
            _ => Err(magnus::Error::new(magnus::exception::runtime_error(), "Unknown pooling method")),
        }
    }

    fn normalize_l2(v: &CoreTensor) -> candle_core::Result<CoreTensor> {
        v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
    }

    pub fn model_type(&self) -> String {
        match self.0.model_type {
            Some(mt) => format!("{:?}", mt),
            None => "nil".to_string(),
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "#<Candle::EmbeddingModel model_type: {}, model_id: {}, tokenizer: {}, embedding_size: {}>",
            self.model_type(), 
            self.0.model_id.as_deref().unwrap_or("nil"), 
            self.0.tokenizer_id.as_deref().unwrap_or("nil"),
            self.0.embedding_size.map(|x| x.to_string()).unwrap_or("nil".to_string())
        )
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Get the tokenizer used by this model
    pub fn tokenizer(&self) -> Result<crate::ruby::tokenizer::Tokenizer> {
        match &self.0.tokenizer {
            Some(tokenizer) => Ok(crate::ruby::tokenizer::Tokenizer(tokenizer.clone())),
            None => Err(magnus::Error::new(magnus::exception::runtime_error(), "No tokenizer loaded for this model"))
        }
    }
    
    /// Get the model_id
    pub fn model_id(&self) -> Result<String> {
        match &self.0.model_id {
            Some(id) => Ok(id.clone()),
            None => Ok("unknown".to_string())
        }
    }
    
    /// Get the device
    pub fn device(&self) -> Device {
        Device::from_device(&self.0.device)
    }
    
    /// Get all options as a hash
    pub fn options(&self) -> Result<RHash> {
        let hash = RHash::new();
        
        // Add model_id
        if let Some(model_id) = &self.0.model_id {
            hash.aset("model_id", model_id.clone())?;
        }
        
        // Add tokenizer
        if let Some(tokenizer_id) = &self.0.tokenizer_id {
            hash.aset("tokenizer", tokenizer_id.clone())?;
        }
        
        // Add device
        hash.aset("device", self.device().__str__())?;
        
        // Add model_type
        if let Some(model_type) = &self.0.model_type {
            hash.aset("model_type", format!("{:?}", model_type))?;
        }
        
        // Add embedding_size
        if let Some(size) = self.0.embedding_size {
            hash.aset("embedding_size", size)?;
        }
        
        Ok(hash)
    }
}

pub fn init(rb_candle: RModule) -> Result<()> {
    let rb_embedding_model = rb_candle.define_class("EmbeddingModel", class::object())?;
    rb_embedding_model.define_singleton_method("_create", function!(EmbeddingModel::new, 5))?;
    // Expose embedding with an optional pooling_method argument (default: "pooled")
    rb_embedding_model.define_method("_embedding", method!(EmbeddingModel::embedding, 2))?;
    rb_embedding_model.define_method("embeddings", method!(EmbeddingModel::embeddings, 1))?;
    rb_embedding_model.define_method("pool_embedding", method!(EmbeddingModel::pool_embedding, 1))?;
    rb_embedding_model.define_method("pool_and_normalize_embedding", method!(EmbeddingModel::pool_and_normalize_embedding, 1))?;
    rb_embedding_model.define_method("pool_cls_embedding", method!(EmbeddingModel::pool_cls_embedding, 1))?;
    rb_embedding_model.define_method("model_type", method!(EmbeddingModel::model_type, 0))?;
    rb_embedding_model.define_method("to_s", method!(EmbeddingModel::__str__, 0))?;
    rb_embedding_model.define_method("inspect", method!(EmbeddingModel::__repr__, 0))?;
    rb_embedding_model.define_method("tokenizer", method!(EmbeddingModel::tokenizer, 0))?;
    rb_embedding_model.define_method("model_id", method!(EmbeddingModel::model_id, 0))?;
    rb_embedding_model.define_method("device", method!(EmbeddingModel::device, 0))?;
    rb_embedding_model.define_method("options", method!(EmbeddingModel::options, 0))?;
    Ok(())
}