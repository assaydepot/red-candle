use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_transformers::models::phi::{Config, Model as PhiModel};
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3Model};
use hf_hub::api::tokio::Api;
use tokenizers::Tokenizer;

use crate::llm::{GenerationConfig, TextGeneration, TextGenerator, TokenizerWrapper};

/// Phi model wrapper for text generation
pub struct Phi {
    model: PhiVariant,
    tokenizer: TokenizerWrapper,
    device: Device,
    model_id: String,
    eos_token_id: u32,
}

enum PhiVariant {
    Phi2(PhiModel),
    Phi3(Phi3Model),
}

impl Phi {
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get the tokenizer
    pub fn tokenizer(&self) -> &TokenizerWrapper {
        &self.tokenizer
    }
    
    /// Clear the KV cache between generations
    pub fn clear_kv_cache(&mut self) {
        match &mut self.model {
            PhiVariant::Phi2(model) => model.clear_kv_cache(),
            PhiVariant::Phi3(model) => model.clear_kv_cache(),
        }
    }
    
    /// Load a Phi model from HuggingFace
    pub async fn from_pretrained(model_id: &str, device: Device) -> CandleResult<Self> {
        let api = Api::new()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create HF API: {}", e)))?;
        
        let repo = api.model(model_id.to_string());
        
        // Download configuration
        let config_filename = repo.get("config.json").await
            .map_err(|e| candle_core::Error::Msg(format!("Failed to download config: {}", e)))?;
        let config_str = std::fs::read_to_string(config_filename)?;
        
        // Download tokenizer
        let tokenizer_filename = repo.get("tokenizer.json").await
            .map_err(|e| candle_core::Error::Msg(format!("Failed to download tokenizer: {}", e)))?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?;
        
        // Determine EOS token
        let vocab = tokenizer.get_vocab(true);
        let eos_token_id = vocab.get("<|endoftext|>")
            .or_else(|| vocab.get("<|end|>"))
            .or_else(|| vocab.get("</s>"))
            .copied()
            .unwrap_or(50256); // Default GPT-2 style EOS token
        
        // Determine model variant based on model_id or config
        let is_phi3 = model_id.contains("phi-3") || model_id.contains("Phi-3");
        
        // Download model weights (handle both single and sharded files)
        let weights_filenames = if let Ok(single_file) = repo.get("model.safetensors").await {
            vec![single_file]
        } else {
            // Try to find sharded model files
            // NOTE: This uses a brute-force approach, trying common shard counts.
            // A better approach would be to read model.safetensors.index.json which
            // contains the exact file list, but this works for most models (≤30 shards).
            let mut sharded_files = Vec::new();
            let mut index = 1;
            loop {
                // Try common shard counts
                let mut found = false;
                for total in [2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30] {
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
        
        let model = if is_phi3 {
            // Load Phi3 model
            let config: Phi3Config = serde_json::from_str(&config_str)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to parse Phi3 config: {}", e)))?;
            
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&weights_filenames, DType::F32, &device)?
            };
            
            let model = Phi3Model::new(&config, vb)?;
            PhiVariant::Phi3(model)
        } else {
            // Load Phi2 model
            let config: Config = serde_json::from_str(&config_str)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to parse Phi config: {}", e)))?;
            
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&weights_filenames, DType::F32, &device)?
            };
            
            let model = PhiModel::new(&config, vb)?;
            PhiVariant::Phi2(model)
        };
        
        Ok(Self {
            model,
            tokenizer: TokenizerWrapper::new(tokenizer),
            device,
            model_id: model_id.to_string(),
            eos_token_id,
        })
    }
    
    /// Apply Phi chat template to messages
    pub fn apply_chat_template(&self, messages: &[serde_json::Value]) -> CandleResult<String> {
        let mut prompt = String::new();
        
        // Phi-3 uses a specific format
        if matches!(self.model, PhiVariant::Phi3(_)) {
            for message in messages {
                let role = message["role"].as_str().unwrap_or("");
                let content = message["content"].as_str().unwrap_or("");
                
                match role {
                    "system" => {
                        prompt.push_str(&format!("<|system|>\n{}<|end|>\n", content));
                    }
                    "user" => {
                        prompt.push_str(&format!("<|user|>\n{}<|end|>\n", content));
                    }
                    "assistant" => {
                        prompt.push_str(&format!("<|assistant|>\n{}<|end|>\n", content));
                    }
                    _ => {}
                }
            }
            prompt.push_str("<|assistant|>\n");
        } else {
            // Phi-2 uses a simpler format
            for message in messages {
                let role = message["role"].as_str().unwrap_or("");
                let content = message["content"].as_str().unwrap_or("");
                
                match role {
                    "system" => prompt.push_str(&format!("System: {}\n", content)),
                    "user" => prompt.push_str(&format!("User: {}\n", content)),
                    "assistant" => prompt.push_str(&format!("Assistant: {}\n", content)),
                    _ => {}
                }
            }
            prompt.push_str("Assistant: ");
        }
        
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
            let logits = match &mut self.model {
                PhiVariant::Phi2(model) => model.forward(&input)?,
                PhiVariant::Phi3(model) => model.forward(&input, start_pos)?,
            };
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
            
            // Check if constraint is satisfied (early stopping)
            if config.stop_on_constraint_satisfaction {
                let satisfied = if config.constraint_greedy {
                    text_gen.is_constraint_satisfied_greedy()
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
}

impl TextGenerator for Phi {
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