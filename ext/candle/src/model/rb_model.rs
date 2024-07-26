#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use crate::model::{
    errors::{wrap_candle_err, wrap_hf_err, wrap_std_err},
    rb_tensor::RbTensor,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use magnus::Error;
use crate::model::RbResult;
use serde_json;
use std::fs;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[magnus::wrap(class = "Candle::Model", free_immediately, size)]
pub struct RbModel(pub RbModelInner);

pub struct RbModelInner {
    device: Device,
    tokenizer_path: Option<String>,
    model_path: Option<String>,
    model: Option<BertModel>,
    tokenizer: Option<Tokenizer>,
}

impl RbModel {
    pub fn new() -> RbResult<Self> {
        Self::new3(Some("jinaai/jina-embeddings-v2-base-en".to_string()), Some("sentence-transformers/all-MiniLM-L6-v2".to_string()), Some(Device::Cpu))
    }

    pub fn new1(model_path: Option<String>) -> RbResult<Self> {
        Self::new3(model_path, Some("sentence-transformers/all-MiniLM-L6-v2".to_string()), Some(Device::Cpu))
    }

    pub fn new2(model_path: Option<String>, tokenizer_path: Option<String>) -> RbResult<Self> {
        Self::new3(model_path, tokenizer_path, Some(Device::Cpu))
    }

    pub fn new3(model_path: Option<String>, tokenizer_path: Option<String>, device: Option<Device>) -> RbResult<Self> {
        let device = device.unwrap_or(Device::Cpu);
        Ok(RbModel(RbModelInner {
            device: device.clone(),
            model_path: model_path.clone(),
            tokenizer_path: tokenizer_path.clone(),
            model: match model_path {
                Some(mp) => Some(Self::build_model(mp, device)?),
                None => None
            },
            tokenizer: match tokenizer_path {
                Some(tp) => Some(Self::build_tokenizer(tp)?),
                None => None
            }
        }))
    }

    /// Performs the `sin` operation on the tensor.
    /// &RETURNS&: Tensor
    pub fn embedding(&self, input: String) -> RbResult<RbTensor> {
        match &self.0.model {
            Some(model) => {
                match &self.0.tokenizer {
                    Some(tokenizer) => Ok(RbTensor(self.compute_embedding(input, model, tokenizer)?)),
                    None => Err(magnus::Error::new(magnus::exception::runtime_error(), "Tokenizer not found"))
                }
            }
            None => Err(magnus::Error::new(magnus::exception::runtime_error(), "Tokenizer or Model not found"))
        }
    }

    fn build_model(model_path: String, device: Device) -> RbResult<BertModel> {
        use hf_hub::{api::sync::Api, Repo, RepoType};
        let model_path = Api::new()
                .map_err(wrap_hf_err)?
                .repo(Repo::new(
                    model_path.clone(),
                    RepoType::Model,
                ))
                .get("model.safetensors")
                .map_err(wrap_hf_err)?;
        println!("Model path: {:?}", model_path);
        let config_path = model_path.parent().unwrap().join("config.json");
        println!("Config path: {:?}", config_path);

        // let config_path = Api::new()
        //         .map_err(wrap_hf_err)?
        //         .repo(Repo::new(
        //             model_path.to_str().unwrap().to_string(),
        //             RepoType::Model,
        //         ))
        //         .get("config.json")
        //         .map_err(wrap_hf_err)?;
        
        let config: Config = read_config(config_path)?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)
                .map_err(wrap_candle_err)?
        };
        let model = BertModel::load(vb, &config).map_err(wrap_candle_err)?;
        Ok(model)
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
        model: &BertModel,
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

        let token_type_ids = Tensor::zeros(&*token_ids.shape(), DType::I64, &self.0.device)
            .map_err(wrap_candle_err)?;

        let result = model.forward(&token_ids, &token_type_ids).map_err(wrap_candle_err)?;

        // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
        let (_n_sentence, n_tokens, _hidden_size) = result.dims3()
            .map_err(wrap_candle_err)?;
        let sum = result.sum(1)
            .map_err(wrap_candle_err)?;
        let embeddings = (sum / (n_tokens as f64))
            .map_err(wrap_candle_err)?;

        Ok(embeddings)
    }

    #[allow(dead_code)]
    fn normalize_l2(v: &Tensor) -> Result<Tensor, candle_core::Error> {
        v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
    }

    pub fn __repr__(&self) -> String {
        format!("#<Candle::Model model_path: {} tokenizer_path: {})", self.0.model_path.as_deref().unwrap_or("nil"), self.0.tokenizer_path.as_deref().unwrap_or("nil"))
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

fn read_config(config_path: PathBuf) -> Result<Config, magnus::Error> {
    let config_str = fs::read_to_string(config_path).map_err(|e| wrap_std_err(Box::new(e)))?;
    println!("Config string: {}", config_str);
    let config_json: Config = serde_json::from_str(&config_str).map_err(|e| wrap_std_err(Box::new(e)))?;
    Ok(config_json)
}

// #[cfg(test)]
// mod tests {
//     #[test]
//     fn my_first_test() {
//         assert_eq!(2 + 2, 4);
//     }

//     #[test]
//     fn test_build_model_and_tokenizer() {
//         let config = super::RbModel::build();
//         let (_model, tokenizer) = config.build_model_and_tokenizer().unwrap();
//         assert_eq!(tokenizer.get_vocab_size(true), 30522);
//     }

//     #[test]
//     fn test_embedding() {
//         let config = super::RbModel::build();
//         // let (_model, tokenizer) = config.build_model_and_tokenizer().unwrap();
//         // assert_eq!(config.embedding("Scientist.com is a marketplace for pharmaceutical services.")?, None);
//     }
// }
