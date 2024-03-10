#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_transformers::models::jina_bert::{BertModel, Config};

use anyhow::Error as E;
use anyhow::Result;
use candle_core::{Device, DType, Module, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

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
            tokenizer_path: None
        }
    }

    pub fn build_model_and_tokenizer(&self) -> anyhow::Result<(BertModel, tokenizers::Tokenizer)> {
        use hf_hub::{api::sync::Api, Repo, RepoType};
        let model_path = match &self.model_path {
            Some(model_file) => std::path::PathBuf::from(model_file),
            None => Api::new()?
                .repo(Repo::new(
                    "jinaai/jina-embeddings-v2-base-en".to_string(),
                    RepoType::Model,
                ))
                .get("model.safetensors")?,
        };
        let tokenizer_path = match &self.tokenizer_path {
            Some(file) => std::path::PathBuf::from(file),
            None => Api::new()?
                .repo(Repo::new(
                    "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                    RepoType::Model,
                ))
                .get("tokenizer.json")?,
        };
        // let device = candle_examples::device(self.cpu)?;
        let config = Config::v2_base();
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &self.device)? };
        let model = BertModel::new(vb, &config)?;
        Ok((model, tokenizer))
    }

    pub fn embedding(&self, input: String) -> anyhow::Result<Tensor> {
        let config = ModelConfig::build();
        let (model, tokenizer) = config.build_model_and_tokenizer()?;
        return self.compute_embedding(input, model, tokenizer);
    }

    fn compute_embedding(&self, prompt: String, model: BertModel, mut tokenizer: Tokenizer) -> Result<Tensor, E> {
        let start: std::time::Instant = std::time::Instant::now();
        // let prompt = args.prompt.as_deref().unwrap_or("Hello, world!");
        let tokenizer = tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)?;
        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        println!("Loaded and encoded {:?}", start.elapsed());
        let start: std::time::Instant = std::time::Instant::now();
        let result = model.forward(&token_ids)?;
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