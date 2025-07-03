use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::mistral::{Config, Model as MistralModel};
use hf_hub::{api::tokio::Api, Repo};
use tokenizers::Tokenizer;

use super::{GenerationConfig, TextGeneration, TextGenerator, TokenizerWrapper};

#[derive(Debug)]
pub struct Mistral {
    model: MistralModel,
    tokenizer: TokenizerWrapper,
    device: Device,
    model_id: String,
    eos_token_id: u32,
}

impl Mistral {
    /// Load a Mistral model from HuggingFace Hub
    pub async fn from_pretrained(model_id: &str, device: Device) -> CandleResult<Self> {
        let api = Api::new()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create HF API: {}", e)))?;
        
        let repo = api.repo(Repo::model(model_id.to_string()));
        
        // Download model files
        let config_filename = repo
            .get("config.json")
            .await
            .map_err(|e| candle_core::Error::Msg(format!("Failed to download config: {}", e)))?;
        
        let tokenizer_filename = repo
            .get("tokenizer.json")
            .await
            .map_err(|e| candle_core::Error::Msg(format!("Failed to download tokenizer: {}", e)))?;
        
        let weights_filename = repo
            .get("model.safetensors")
            .await
            .map_err(|e| candle_core::Error::Msg(format!("Failed to download weights: {}", e)))?;
        
        // Load config
        let config: Config = serde_json::from_reader(std::fs::File::open(config_filename)?)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to parse config: {}", e)))?;
        
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?;
        
        let eos_token_id = tokenizer
            .get_vocab(true)
            .get("</s>")
            .copied()
            .unwrap_or(2);
        
        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)?
        };
        
        let model = MistralModel::new(&config, vb)?;
        
        Ok(Self {
            model,
            tokenizer: TokenizerWrapper::new(tokenizer),
            device,
            model_id: model_id.to_string(),
            eos_token_id,
        })
    }

    /// Create from existing components (useful for testing)
    pub fn new(
        model: MistralModel,
        tokenizer: Tokenizer,
        device: Device,
        model_id: String,
    ) -> Self {
        let eos_token_id = tokenizer
            .get_vocab(true)
            .get("</s>")
            .copied()
            .unwrap_or(2);
        
        Self {
            model,
            tokenizer: TokenizerWrapper::new(tokenizer),
            device,
            model_id,
            eos_token_id,
        }
    }

    fn generate_tokens(
        &mut self,
        prompt_tokens: Vec<u32>,
        config: &GenerationConfig,
        mut callback: Option<impl FnMut(&str)>,
    ) -> CandleResult<Vec<u32>> {
        let mut text_gen = TextGeneration::from_config(config);
        text_gen.set_eos_token_id(self.eos_token_id);
        text_gen.set_tokens(prompt_tokens.clone());
        
        let mut all_tokens = prompt_tokens.clone();
        let start_gen = all_tokens.len();
        
        for index in 0..config.max_length {
            let context_tokens = if index == 0 {
                all_tokens.as_slice()
            } else {
                &all_tokens[all_tokens.len().saturating_sub(1)..]
            };
            
            let input = Tensor::new(context_tokens, &self.device)?;
            let logits = self.model.forward(&input, all_tokens.len())?;
            let logits = logits.squeeze(0)?;
            
            let next_token = text_gen.sample_next_token(
                &logits,
                Some((config.repetition_penalty, config.repetition_penalty_last_n)),
            )?;
            
            all_tokens.push(next_token);
            
            // Stream callback
            if let Some(ref mut cb) = callback {
                let token_text = self.tokenizer.token_to_piece(next_token)?;
                cb(&token_text);
            }
            
            // Check stop conditions
            if text_gen.should_stop(next_token, config.max_length) {
                break;
            }
            
            // Check stop sequences
            let generated_text = self.tokenizer.decode(&all_tokens[start_gen..], true)?;
            if text_gen.check_stop_sequences(&generated_text, &config.stop_sequences) {
                break;
            }
        }
        
        Ok(if config.include_prompt {
            all_tokens
        } else {
            all_tokens[start_gen..].to_vec()
        })
    }
}

impl TextGenerator for Mistral {
    fn generate(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> CandleResult<String> {
        let prompt_tokens = self.tokenizer.encode(prompt, true)?;
        let output_tokens = self.generate_tokens(prompt_tokens, config, None::<fn(&str)>)?;
        self.tokenizer.decode(&output_tokens, true)
    }

    fn generate_stream(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
        mut callback: impl FnMut(&str),
    ) -> CandleResult<String> {
        let prompt_tokens = self.tokenizer.encode(prompt, true)?;
        let output_tokens = self.generate_tokens(prompt_tokens, config, Some(&mut callback))?;
        self.tokenizer.decode(&output_tokens, true)
    }

    fn model_name(&self) -> &str {
        &self.model_id
    }

    fn device(&self) -> &Device {
        &self.device
    }
}