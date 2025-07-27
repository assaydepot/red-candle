use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_transformers::models::qwen2::{Config, Model as QwenModel};
use hf_hub::api::tokio::Api;
use tokenizers::Tokenizer;

use crate::llm::{GenerationConfig, TextGeneration, TextGenerator, TokenizerWrapper};

/// Qwen model wrapper for text generation
#[derive(Debug)]
pub struct Qwen {
    model: QwenModel,
    tokenizer: TokenizerWrapper,
    device: Device,
    model_id: String,
    eos_token_id: u32,
}

impl Qwen {
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get the tokenizer
    pub fn tokenizer(&self) -> &TokenizerWrapper {
        &self.tokenizer
    }
    
    /// Clear the KV cache between generations
    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
    
    /// Load a Qwen model from HuggingFace
    pub async fn from_pretrained(model_id: &str, device: Device) -> CandleResult<Self> {
        let api = Api::new()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create HF API: {}", e)))?;
        
        let repo = api.model(model_id.to_string());
        
        // Download configuration
        let config_filename = repo.get("config.json").await
            .map_err(|e| candle_core::Error::Msg(format!("Failed to download config: {}", e)))?;
        let config_str = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to parse config: {}", e)))?;
        
        // Download tokenizer
        let tokenizer_filename = repo.get("tokenizer.json").await
            .map_err(|e| candle_core::Error::Msg(format!("Failed to download tokenizer: {}", e)))?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?;
        
        // Determine EOS token
        let vocab = tokenizer.get_vocab(true);
        let eos_token_id = vocab.get("<|endoftext|>")
            .or_else(|| vocab.get("<|im_end|>"))
            .or_else(|| vocab.get("</s>"))
            .copied()
            .unwrap_or(151643); // Default Qwen3 EOS token
        
        // Download model weights
        // NOTE: Qwen uses hardcoded shard counts based on model size rather than
        // reading model.safetensors.index.json. This works for official Qwen models
        // but may fail for custom configurations with different shard counts.
        let mut filenames = vec![];
        let num_shards = if model_id.contains("72b") || model_id.contains("72B") { 8 } 
                        else if model_id.contains("14b") || model_id.contains("14B") { 3 }
                        else { 1 };
        
        if num_shards == 1 {
            // Single file model
            let filename = repo.get("model.safetensors").await
                .map_err(|e| candle_core::Error::Msg(format!("Failed to download model weights: {}", e)))?;
            filenames.push(filename);
        } else {
            // Sharded model
            for shard_idx in 1..=num_shards {
                let filename = repo.get(&format!("model-{:05}-of-{:05}.safetensors", shard_idx, num_shards)).await
                    .map_err(|e| candle_core::Error::Msg(format!("Failed to download shard {}: {}", shard_idx, e)))?;
                filenames.push(filename);
            }
        }
        
        // Load the model
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&filenames, DType::F32, &device)?
        };
        
        let model = QwenModel::new(&config, vb)?;
        
        Ok(Self {
            model,
            tokenizer: TokenizerWrapper::new(tokenizer),
            device,
            model_id: model_id.to_string(),
            eos_token_id,
        })
    }
    
    /// Apply Qwen chat template to messages
    pub fn apply_chat_template(&self, messages: &[serde_json::Value]) -> CandleResult<String> {
        let mut prompt = String::new();
        
        for message in messages {
            let role = message["role"].as_str().unwrap_or("");
            let content = message["content"].as_str().unwrap_or("");
            
            match role {
                "system" => {
                    prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", content));
                }
                "user" => {
                    prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", content));
                }
                "assistant" => {
                    prompt.push_str(&format!("<|im_start|>assistant\n{}<|im_end|>\n", content));
                }
                _ => {}
            }
        }
        
        // Add generation prompt
        prompt.push_str("<|im_start|>assistant\n");
        
        Ok(prompt)
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
            let context_size = if index > 0 { 1 } else { all_tokens.len() };
            let start_pos = all_tokens.len().saturating_sub(context_size);
            let ctxt = &all_tokens[start_pos..];
            
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos, None)?;
            let logits = logits.squeeze(0)?;
            
            // Handle different output shapes
            let logits = if logits.dims().len() == 2 {
                let seq_len = logits.dim(0)?;
                logits.narrow(0, seq_len - 1, 1)?.squeeze(0)?
            } else {
                logits
            };
            
            let logits = logits.to_dtype(DType::F32)?;
            
            let next_token = text_gen.sample_next_token(
                &logits,
                Some((config.repetition_penalty, config.repetition_penalty_last_n)),
            )?;
            
            all_tokens.push(next_token);
            
            // Stream callback
            if let Some(ref mut cb) = callback {
                if config.debug_tokens {
                    let token_piece = self.tokenizer.token_to_piece(next_token)?;
                    cb(&format!("[{}:{}]", next_token, token_piece));
                } else {
                    let decoded_text = self.tokenizer.decode_incremental(&all_tokens, all_tokens.len() - 1)?;
                    cb(&decoded_text);
                }
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

impl TextGenerator for Qwen {
    fn generate(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> CandleResult<String> {
        let prompt_tokens = self.tokenizer.encode(prompt, true)?;
        let output_tokens = self.generate_tokens(prompt_tokens, config, None::<fn(&str)>)?;
        
        if config.debug_tokens {
            self.tokenizer.format_tokens_with_debug(&output_tokens)
        } else {
            self.tokenizer.decode(&output_tokens, true)
        }
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
    
    fn clear_cache(&mut self) {
        self.clear_kv_cache();
    }
}