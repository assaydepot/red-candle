use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Config, LlamaConfig, Llama as LlamaModel, Cache};
use hf_hub::{api::tokio::Api, Repo};
use tokenizers::Tokenizer;

use super::{GenerationConfig, TextGeneration, TextGenerator, TokenizerWrapper};

#[derive(Debug)]
pub struct Llama {
    model: LlamaModel,
    tokenizer: TokenizerWrapper,
    device: Device,
    model_id: String,
    eos_token_id: u32,
    cache: Cache,
    config: Config,
}

impl Llama {
    /// Clear the KV cache between generations
    pub fn clear_kv_cache(&mut self) {
        // Since Cache doesn't expose a reset method and kvs is private,
        // we'll recreate the cache to clear it
        // This is a workaround until candle provides a proper reset method
        if let Ok(new_cache) = Cache::new(self.cache.use_kv_cache, DType::F32, &self.config, &self.device) {
            self.cache = new_cache;
        }
    }
    
    /// Load a Llama model from HuggingFace Hub
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
        } else if let Ok(consolidated_file) = repo.get("consolidated.safetensors").await {
            vec![consolidated_file]
        } else {
            // Try to find sharded model files
            let mut sharded_files = Vec::new();
            let mut index = 1;
            loop {
                // Try common shard counts for Llama models
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
                    "Could not find model weights. Tried: model.safetensors, consolidated.safetensors, model-*-of-*.safetensors".to_string()
                ));
            }
            sharded_files
        };
        
        // Load config
        let llama_config: LlamaConfig = serde_json::from_reader(std::fs::File::open(config_filename)?)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to parse config: {}", e)))?;
        let config = llama_config.into_config(false); // Don't use flash attention for now
        
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?;
        
        // Determine EOS token ID based on model type
        let eos_token_id = if model_id.contains("Llama-3") || model_id.contains("llama-3") {
            // Llama 3 uses different special tokens
            {
                let vocab = tokenizer.get_vocab(true);
                vocab.get("<|eot_id|>")
                    .or_else(|| vocab.get("<|end_of_text|>"))
                    .copied()
                    .unwrap_or(128009) // Default Llama 3 EOS
            }
        } else {
            // Llama 2 and earlier
            tokenizer
                .get_vocab(true)
                .get("</s>")
                .copied()
                .unwrap_or(2)
        };
        
        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weights_filenames, DType::F32, &device)?
        };
        
        let model = LlamaModel::load(vb, &config)?;
        let cache = Cache::new(true, DType::F32, &config, &device)?;
        
        Ok(Self {
            model,
            tokenizer: TokenizerWrapper::new(tokenizer),
            device,
            model_id: model_id.to_string(),
            eos_token_id,
            cache,
            config,
        })
    }

    /// Create from existing components (useful for testing)
    pub fn new(
        model: LlamaModel,
        tokenizer: Tokenizer,
        device: Device,
        model_id: String,
        config: &Config,
    ) -> CandleResult<Self> {
        let eos_token_id = if model_id.contains("Llama-3") || model_id.contains("llama-3") {
            {
                let vocab = tokenizer.get_vocab(true);
                vocab.get("<|eot_id|>")
                    .or_else(|| vocab.get("<|end_of_text|>"))
                    .copied()
                    .unwrap_or(128009)
            }
        } else {
            tokenizer
                .get_vocab(true)
                .get("</s>")
                .copied()
                .unwrap_or(2)
        };
        
        let cache = Cache::new(true, DType::F32, config, &device)?;
        
        Ok(Self {
            model,
            tokenizer: TokenizerWrapper::new(tokenizer),
            device,
            model_id,
            eos_token_id,
            cache,
            config: config.clone(),
        })
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
            let input = input.contiguous()?;
            let logits = self.model.forward(&input, start_pos, &mut self.cache)?;
            
            let logits = logits.squeeze(0)?;
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
    
    fn generate_tokens_decoded(
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
        let mut previously_decoded = String::new();
        
        for index in 0..config.max_length {
            let context_size = if index > 0 { 1 } else { all_tokens.len() };
            let start_pos = all_tokens.len().saturating_sub(context_size);
            let ctxt = &all_tokens[start_pos..];
            
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let input = input.contiguous()?;
            let logits = self.model.forward(&input, start_pos, &mut self.cache)?;
            
            let logits = logits.squeeze(0)?;
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
            
            // Stream callback with incremental decoding
            if let Some(ref mut cb) = callback {
                let current_decoded = self.tokenizer.decode(&all_tokens[start_gen..], true)?;
                
                if current_decoded.len() > previously_decoded.len() {
                    let new_text = &current_decoded[previously_decoded.len()..];
                    cb(new_text);
                    previously_decoded = current_decoded;
                }
            }
            
            // Check stop conditions
            if text_gen.should_stop(next_token, config.max_length) {
                break;
            }
            
            // Check stop sequences
            let generated_text = if callback.is_some() {
                previously_decoded.clone()
            } else {
                self.tokenizer.decode(&all_tokens[start_gen..], true)?
            };
            
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
    
    /// Apply chat template based on Llama version
    pub fn apply_chat_template(&self, messages: &[serde_json::Value]) -> CandleResult<String> {
        let is_llama3 = self.model_id.contains("Llama-3") || self.model_id.contains("llama-3");
        
        if is_llama3 {
            self.apply_llama3_template(messages)
        } else {
            self.apply_llama2_template(messages)
        }
    }
    
    fn apply_llama2_template(&self, messages: &[serde_json::Value]) -> CandleResult<String> {
        let mut prompt = String::new();
        let mut system_message = String::new();
        
        for (i, message) in messages.iter().enumerate() {
            let role = message["role"].as_str().unwrap_or("");
            let content = message["content"].as_str().unwrap_or("");
            
            match role {
                "system" => {
                    system_message = content.to_string();
                }
                "user" => {
                    if i == 1 || (i == 0 && system_message.is_empty()) {
                        // First user message
                        if !system_message.is_empty() {
                            prompt.push_str(&format!("<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]", system_message, content));
                        } else {
                            prompt.push_str(&format!("<s>[INST] {} [/INST]", content));
                        }
                    } else {
                        prompt.push_str(&format!(" [INST] {} [/INST]", content));
                    }
                }
                "assistant" => {
                    prompt.push_str(&format!(" {} </s>", content));
                }
                _ => {}
            }
        }
        
        Ok(prompt)
    }
    
    fn apply_llama3_template(&self, messages: &[serde_json::Value]) -> CandleResult<String> {
        let mut prompt = String::new();
        
        prompt.push_str("<|begin_of_text|>");
        
        for message in messages {
            let role = message["role"].as_str().unwrap_or("");
            let content = message["content"].as_str().unwrap_or("");
            
            prompt.push_str(&format!("<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>", role, content));
        }
        
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        
        Ok(prompt)
    }
}

impl TextGenerator for Llama {
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
        let output_tokens = self.generate_tokens_decoded(prompt_tokens, config, Some(&mut callback))?;
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