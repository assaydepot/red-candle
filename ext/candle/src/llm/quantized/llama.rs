use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_llama::ModelWeights as QuantizedLlamaModel;
use hf_hub::api::tokio::{Api, ApiRepo};
use tokenizers::Tokenizer;

use crate::llm::{GenerationConfig, TextGeneration, TextGenerator, TokenizerWrapper};

#[derive(Debug)]
pub struct QuantizedLlama {
    model: QuantizedLlamaModel,
    tokenizer: TokenizerWrapper,
    device: Device,
    model_id: String,
    eos_token_id: u32,
}

impl QuantizedLlama {
    /// Clear the KV cache between generations
    pub fn clear_kv_cache(&mut self) {
        // Quantized models don't expose cache clearing
        // Cache is managed internally
    }
    
    /// Load a quantized Llama model from HuggingFace Hub
    pub async fn from_pretrained(model_id: &str, device: Device) -> CandleResult<Self> {
        // Check if user specified an exact GGUF filename
        let (actual_model_id, gguf_file) = if let Some(pos) = model_id.find('@') {
            let (id, filename) = model_id.split_at(pos);
            (id, Some(&filename[1..]))
        } else {
            (model_id, None)
        };
        let api = Api::new()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create HF API: {}", e)))?;
        
        let repo = api.model(actual_model_id.to_string());
        
        // Try to download tokenizer, with fallback options
        let tokenizer_filename = Self::download_tokenizer(&api, &repo, model_id).await?;
        
        // Try to find GGUF file
        let gguf_filename = if let Some(filename) = gguf_file {
            // User specified exact filename
            repo.get(filename).await
                .map_err(|e| candle_core::Error::Msg(format!("Failed to download GGUF file '{}': {}", filename, e)))?
                .to_string_lossy().to_string()
        } else {
            // Search for GGUF file
            Self::find_gguf_file(&api, &repo, actual_model_id).await?
        };
        
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?;
        
        // Determine EOS token ID based on model type
        let eos_token_id = if model_id.contains("Llama-3") || model_id.contains("llama-3") {
            // Llama 3 uses different special tokens
            let vocab = tokenizer.get_vocab(true);
            vocab.get("<|eot_id|>")
                .or_else(|| vocab.get("<|end_of_text|>"))
                .copied()
                .unwrap_or(128009) // Default Llama 3 EOS
        } else {
            // Llama 2 and earlier
            tokenizer
                .get_vocab(true)
                .get("</s>")
                .copied()
                .unwrap_or(2)
        };
        
        // Load GGUF model
        let mut file = std::fs::File::open(&gguf_filename)?;
        let content = gguf_file::Content::read(&mut file)?;
        let model = QuantizedLlamaModel::from_gguf(content, &mut file, &device)?;
        
        Ok(Self {
            model,
            tokenizer: TokenizerWrapper::new(tokenizer),
            device,
            model_id: actual_model_id.to_string(),
            eos_token_id,
        })
    }
    
    /// Download tokenizer with fallback options
    async fn download_tokenizer(api: &Api, repo: &ApiRepo, model_id: &str) -> CandleResult<std::path::PathBuf> {
        // First try to get tokenizer.json from the GGUF repo
        if let Ok(path) = repo.get("tokenizer.json").await {
            return Ok(path);
        }
        
        // Try tokenizer.model (for models that use sentencepiece)
        if let Ok(path) = repo.get("tokenizer.model").await {
            return Ok(path);
        }
        
        // If this is a TheBloke model, try to get tokenizer from the original model
        if model_id.starts_with("TheBloke/") {
            // Extract original model name from TheBloke's naming convention
            // e.g., "TheBloke/Llama-2-7B-Chat-GGUF" -> "meta-llama/Llama-2-7b-chat-hf"
            let model_name = model_id.strip_prefix("TheBloke/").unwrap();
            if let Some(base_name) = model_name.strip_suffix("-GGUF") {
                // Try common base model patterns for Llama
                let base_models = vec![
                    format!("meta-llama/{}-hf", base_name),
                    format!("meta-llama/{}", base_name),
                    format!("NousResearch/{}-hf", base_name),
                    format!("TinyLlama/{}", base_name),
                ];
                
                for base_model in base_models {
                    let base_repo = api.model(base_model.clone());
                    if let Ok(path) = base_repo.get("tokenizer.json").await {
                        return Ok(path);
                    }
                }
            }
        }
        
        // For other GGUF models, try common fallback tokenizers
        let fallback_tokenizers = vec![
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "meta-llama/Llama-2-7b-hf",
            "NousResearch/Llama-2-7b-hf",
        ];
        
        for fallback in fallback_tokenizers {
            let fallback_repo = api.model(fallback.to_string());
            if let Ok(path) = fallback_repo.get("tokenizer.json").await {
                return Ok(path);
            }
        }
        
        Err(candle_core::Error::Msg(format!(
            "Failed to find tokenizer for {}. GGUF models often don't include separate tokenizer files.",
            model_id
        )))
    }
    
    /// Find GGUF file in the repository
    async fn find_gguf_file(_api: &Api, repo: &ApiRepo, model_id: &str) -> CandleResult<String> {
        // Default search order (Q4_K_M is recommended)
        let common_names = vec![
            "model.gguf",
            // Try Q4_K_M first (recommended)
            "llama-2-7b-chat.Q4_K_M.gguf",
            "ggml-model-q4_k_m.gguf",
            // Then Q5_K_M
            "llama-2-7b-chat.Q5_K_M.gguf",
            "ggml-model-q5_k_m.gguf",
            // Then Q8_0 (highest quality)
            "llama-2-7b-chat.Q8_0.gguf",
            // Older formats
            "ggml-model-q4_0.gguf",
        ];
        
        for name in common_names {
            if let Ok(path) = repo.get(&name).await {
                return Ok(path.to_string_lossy().to_string());
            }
        }
        
        // If no file found, provide helpful error
        Err(candle_core::Error::Msg(format!(
            "No GGUF file found in repository {}. Try specifying a quantization level like Q4_K_M, Q5_K_M, or Q8_0.",
            model_id
        )))
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
            let logits = self.model.forward(&input, start_pos)?;
            
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
            let logits = self.model.forward(&input, start_pos)?;
            
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
}

impl TextGenerator for QuantizedLlama {
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
        // Quantized models manage cache internally
    }
}