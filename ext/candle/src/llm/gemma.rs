use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma::{Config, Model as GemmaModel};
use hf_hub::{api::tokio::Api, Repo};
use tokenizers::Tokenizer;

use super::{GenerationConfig, TextGeneration, TextGenerator, TokenizerWrapper};

#[derive(Debug)]
pub struct Gemma {
    model: GemmaModel,
    tokenizer: TokenizerWrapper,
    device: Device,
    model_id: String,
    eos_token_id: u32,
}

impl Gemma {
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Clear the KV cache between generations
    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
    
    /// Get the tokenizer
    pub fn tokenizer(&self) -> &TokenizerWrapper {
        &self.tokenizer
    }
    
    /// Load a Gemma model from HuggingFace Hub
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
        
        // Try different file patterns for model weights
        let weights_filenames = if let Ok(single_file) = repo.get("model.safetensors").await {
            vec![single_file]
        } else {
            // Try to find sharded model files
            // NOTE: This uses a brute-force approach, trying common shard counts.
            // A better approach would be to read model.safetensors.index.json which
            // contains the exact file list, but this works for most models (≤12 shards).
            let mut sharded_files = Vec::new();
            let mut index = 1;
            loop {
                // Try common shard counts for Gemma models
                let mut found = false;
                for total in [2, 3, 4, 5, 6, 7, 8, 10, 12] {
                    let filename = format!("model-{:05}-of-{:05}.safetensors", index, total);
                    if let Ok(file) = repo.get(&filename).await {
                        sharded_files.push(file);
                        found = true;
                        break;
                    }
                }
                if !found {
                    break;
                }
                index += 1;
            }
            
            if sharded_files.is_empty() {
                return Err(candle_core::Error::Msg(
                    "Could not find model weights. Tried: model.safetensors, model-*-of-*.safetensors".to_string()
                ));
            }
            sharded_files
        };
        
        // Load config
        let config: Config = serde_json::from_reader(std::fs::File::open(config_filename)?)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to parse config: {}", e)))?;
        
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?;
        
        // Gemma uses specific tokens
        let eos_token_id = {
            let vocab = tokenizer.get_vocab(true);
            vocab.get("<eos>")
                .or_else(|| vocab.get("<end_of_turn>"))
                .copied()
                .unwrap_or(1) // Default Gemma EOS
        };
        
        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weights_filenames, DType::F32, &device)?
        };
        
        let model = GemmaModel::new(false, &config, vb)?; // Don't use flash attention for now
        
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
        model: GemmaModel,
        tokenizer: Tokenizer,
        device: Device,
        model_id: String,
    ) -> Self {
        let eos_token_id = {
            let vocab = tokenizer.get_vocab(true);
            vocab.get("<eos>")
                .or_else(|| vocab.get("<end_of_turn>"))
                .copied()
                .unwrap_or(1)
        };
        
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
        let mut text_gen = TextGeneration::new(config);
        text_gen.set_eos_token_id(self.eos_token_id);
        text_gen.set_tokens(prompt_tokens.clone());
        
        let mut all_tokens = prompt_tokens.clone();
        let start_gen = all_tokens.len();
        
        for index in 0..config.max_length {
            let context_size = if index > 0 { 1 } else { all_tokens.len() };
            let start_pos = all_tokens.len().saturating_sub(context_size);
            let ctxt = &all_tokens[start_pos..];
            
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let input = input.contiguous()?;
            let logits = self.model.forward(&input, start_pos)?;
            
            let logits = logits.squeeze(0)?;
            let logits = if logits.dims().len() == 2 {
                let seq_len = logits.dim(0)?;
                logits.narrow(0, seq_len - 1, 1)?.squeeze(0)?
            } else {
                logits
            };
            
            let logits = logits.to_dtype(DType::F32)?;
            
            let next_token = text_gen.sample_next_token(&logits)?;
            
            all_tokens.push(next_token);
            
            // Stream callback
            if let Some(ref mut cb) = callback {
                if config.debug_tokens {
                    // In debug mode, only show debug tokens
                    let token_piece = self.tokenizer.token_to_piece(next_token)?;
                    cb(&format!("[{}:{}]", next_token, token_piece));
                } else {
                    // Normal mode: use incremental decoding for proper text
                    let decoded_text = self.tokenizer.decode_incremental(&all_tokens, all_tokens.len() - 1)?;
                    cb(&decoded_text);
                }
            }
            
            // Check stop conditions
            if text_gen.should_stop(next_token, config.max_length) {
                break;
            }
            
            // Check if constraint is satisfied (early stopping)
            if config.stop_on_constraint_satisfaction {
                let satisfied = if config.stop_on_match {
                    text_gen.is_constraint_satisfied_stop_on_match()
                } else {
                    text_gen.is_constraint_satisfied()
                };
                if satisfied {
                    break;
                }
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
    
    /// Apply Gemma chat template
    pub fn apply_chat_template(&self, messages: &[serde_json::Value]) -> CandleResult<String> {
        let mut prompt = String::new();
        
        // Gemma uses a specific format:
        // <start_of_turn>user\n{user_message}<end_of_turn>
        // <start_of_turn>model\n{model_message}<end_of_turn>
        
        for message in messages {
            let role = message["role"].as_str().unwrap_or("");
            let content = message["content"].as_str().unwrap_or("");
            
            match role {
                "system" => {
                    // Gemma doesn't have explicit system messages, prepend to first user message
                    prompt.push_str(&format!("<start_of_turn>user\nSystem: {}\n", content));
                }
                "user" => {
                    if !prompt.contains("<start_of_turn>user") || prompt.ends_with("<end_of_turn>\n") {
                        prompt.push_str("<start_of_turn>user\n");
                    }
                    prompt.push_str(&format!("{}<end_of_turn>\n", content));
                }
                "assistant" | "model" => {
                    prompt.push_str(&format!("<start_of_turn>model\n{}<end_of_turn>\n", content));
                }
                _ => {}
            }
        }
        
        // Add the model prompt
        prompt.push_str("<start_of_turn>model\n");
        
        Ok(prompt)
    }
}

impl TextGenerator for Gemma {
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