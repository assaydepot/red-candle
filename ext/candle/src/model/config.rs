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
use core::result::Result;
use magnus::Error;
use tokenizers::Tokenizer;

#[magnus::wrap(class = "Candle::Model", free_immediately, size)]
pub struct ModelConfig {
    device: Device,

    tokenizer_path: Option<String>,

    model_path: Option<String>,
}

impl ModelConfig {
    pub fn build() -> ModelConfig {
        ModelConfig {
            device: Device::Cpu,
            model_path: None,
            tokenizer_path: None,
        }
    }

    pub fn build_model_and_tokenizer(&self) -> Result<(BertModel, tokenizers::Tokenizer), Error> {
        use hf_hub::{api::sync::Api, Repo, RepoType};
        let model_path = match &self.model_path {
            Some(model_file) => std::path::PathBuf::from(model_file),
            None => Api::new()
                .map_err(wrap_hf_err)?
                .repo(Repo::new(
                    "jinaai/jina-embeddings-v2-base-en".to_string(),
                    RepoType::Model,
                ))
                .get("model.safetensors")
                .map_err(wrap_hf_err)?,
        };
        let tokenizer_path = match &self.tokenizer_path {
            Some(file) => std::path::PathBuf::from(file),
            None => Api::new()
                .map_err(wrap_hf_err)?
                .repo(Repo::new(
                    "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                    RepoType::Model,
                ))
                .get("tokenizer.json")
                .map_err(wrap_hf_err)?,
        };
        // let device = candle_examples::device(self.cpu)?;
        let config = Config::v2_base();
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(wrap_std_err)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &self.device)
                .map_err(wrap_candle_err)?
        };
        let model = BertModel::new(vb, &config).map_err(wrap_candle_err)?;
        Ok((model, tokenizer))
    }

    pub fn embedding(&self, input: String) -> Result<RbTensor, Error> {
        let config = ModelConfig::build();
        let (model, tokenizer) = config.build_model_and_tokenizer()?;
        return Ok(RbTensor(self.compute_embedding(input, model, tokenizer)?));
    }

    fn compute_embedding(
        &self,
        prompt: String,
        model: BertModel,
        mut tokenizer: Tokenizer,
    ) -> Result<Tensor, Error> {
        let start: std::time::Instant = std::time::Instant::now();
        // let prompt = args.prompt.as_deref().unwrap_or("Hello, world!");
        let tokenizer = tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(wrap_std_err)?;
        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(wrap_std_err)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], &self.device)
            .map_err(wrap_candle_err)?
            .unsqueeze(0)
            .map_err(wrap_candle_err)?;
        println!("Loaded and encoded {:?}", start.elapsed());
        let start: std::time::Instant = std::time::Instant::now();
        let result = model.forward(&token_ids).map_err(wrap_candle_err)?;
        println!("{result}");
        println!("Took {:?}", start.elapsed());
        return Ok(result);
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
