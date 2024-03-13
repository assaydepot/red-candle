#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use crate::model::{
    errors::{wrap_candle_err, wrap_hf_err, wrap_std_err},
    rb_tensor::RbTensor,
};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::jina_bert::{BertModel, Config};
use magnus::Error;
use crate::model::RbResult;
use tokenizers::Tokenizer;

#[magnus::wrap(class = "Candle::Model", free_immediately, size)]
pub struct ModelConfig(pub ModelConfigInner);

pub struct ModelConfigInner {
    device: Device,
    tokenizer_path: Option<String>,
    model_path: Option<String>,
    model: Option<BertModel>,
    tokenizer: Option<Tokenizer>,
}

impl ModelConfig {
    pub fn new() -> RbResult<Self> {
        Self::new2(Some("jinaai/jina-embeddings-v2-base-en".to_string()), Some("sentence-transformers/all-MiniLM-L6-v2".to_string()), Some(Device::Cpu))
    }

    pub fn new2(model_path: Option<String>, tokenizer_path: Option<String>, device: Option<Device>) -> RbResult<Self> {
        let device = device.unwrap_or(Device::Cpu);
        Ok(ModelConfig(ModelConfigInner {
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
                    model_path,
                    RepoType::Model,
                ))
                .get("model.safetensors")
                .map_err(wrap_hf_err)?;
        let config = Config::v2_base();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)
                .map_err(wrap_candle_err)?
        };
        let model = BertModel::new(vb, &config).map_err(wrap_candle_err)?;
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
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            // .with_padding(None)
            // .with_truncation(None)
            .map_err(wrap_std_err)?;
        Ok(tokenizer)
    }

    fn compute_embedding(
        &self,
        prompt: String,
        model: &BertModel,
        tokenizer: &Tokenizer,
    ) -> Result<Tensor, Error> {
        let start: std::time::Instant = std::time::Instant::now();
        // let tokenizer_impl = tokenizer
        //     .map_err(wrap_std_err)?;
        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(wrap_std_err)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], &self.0.device)
            .map_err(wrap_candle_err)?
            .unsqueeze(0)
            .map_err(wrap_candle_err)?;
        println!("Loaded and encoded {:?}", start.elapsed());
        let start: std::time::Instant = std::time::Instant::now();
        let result = model.forward(&token_ids).map_err(wrap_candle_err)?;
        // println!("{result}");
        println!("Took {:?}", start.elapsed());
        Ok(result)
    }

    pub fn __repr__(&self) -> String {
        format!("Candle::Model(path={})", self.0.model_path.as_deref().unwrap_or("None"))
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn my_first_test() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn test_build_model_and_tokenizer() {
        let config = super::ModelConfig::build();
        let (_model, tokenizer) = config.build_model_and_tokenizer().unwrap();
        assert_eq!(tokenizer.get_vocab_size(true), 30522);
    }

    #[test]
    fn test_embedding() {
        let config = super::ModelConfig::build();
        // let (_model, tokenizer) = config.build_model_and_tokenizer().unwrap();
        // assert_eq!(config.embedding("Scientist.com is a marketplace for pharmaceutical services.")?, None);
    }
}
