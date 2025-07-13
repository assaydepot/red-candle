use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_llama::ModelWeights as QuantizedLlamaModel;
use candle_transformers::models::quantized_gemma3::ModelWeights as QuantizedGemmaModel;
use hf_hub::api::tokio::{Api, ApiRepo};
use tokenizers::Tokenizer;
use std::io::Seek;

use crate::llm::{GenerationConfig, TextGeneration, TextGenerator, TokenizerWrapper};

/// Unified GGUF model that can load any GGUF file and detect the architecture
#[derive(Debug)]
pub struct QuantizedGGUF {
    model: ModelType,
    tokenizer: TokenizerWrapper,
    device: Device,
    model_id: String,
    eos_token_id: u32,
    architecture: String,
    _chat_template: Option<String>,
}

#[derive(Debug)]
enum ModelType {
    Llama(QuantizedLlamaModel),
    Gemma(QuantizedGemmaModel),
    // Mistral uses Llama loader due to tensor naming compatibility
}

impl QuantizedGGUF {
    /// Load a quantized model from a GGUF file
    pub async fn from_pretrained(model_id: &str, device: Device, tokenizer_source: Option<&str>) -> CandleResult<Self> {
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
        
        // Download GGUF file
        let gguf_filename = if let Some(filename) = gguf_file {
            // User specified exact filename
            repo.get(filename).await
                .map_err(|e| candle_core::Error::Msg(format!("Failed to download GGUF file '{}': {}", filename, e)))?
                .to_string_lossy().to_string()
        } else {
            // Let Ruby handle the search, for now just try a common name
            return Err(candle_core::Error::Msg(
                "Please specify a GGUF filename using gguf_file parameter".to_string()
            ));
        };
        
        // Read GGUF metadata to determine architecture
        let mut file = std::fs::File::open(&gguf_filename)?;
        let content = gguf_file::Content::read(&mut file)?;
        
        // Detect architecture from metadata
        let architecture = Self::detect_architecture(&content, actual_model_id)?;
        
        // For Gemma 3 models, we might need to adjust the architecture
        let architecture = if actual_model_id.contains("gemma-3") || actual_model_id.contains("gemma3") {
            "gemma3".to_string()
        } else {
            architecture
        };
        
        // Download tokenizer - either from specified source or with fallback
        let tokenizer_filename = if let Some(source) = tokenizer_source {
            Self::download_tokenizer_from_source(&api, source).await?
        } else {
            Self::download_tokenizer(&api, &repo, actual_model_id, &architecture).await?
        };
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?;
        
        // Determine EOS token based on architecture and model
        let eos_token_id = Self::determine_eos_token(&tokenizer, &architecture, actual_model_id);
        
        // Load the appropriate model based on architecture
        file.seek(std::io::SeekFrom::Start(0))?;
        let content = gguf_file::Content::read(&mut file)?;
        
        let model = match architecture.as_str() {
            "llama" | "mistral" => {
                // Both use the same GGUF format with llama.cpp tensor names
                let model = QuantizedLlamaModel::from_gguf(content, &mut file, &device)?;
                ModelType::Llama(model)
            }
            "gemma" | "gemma2" | "gemma3" => {
                // Try Gemma-specific loader first, fall back to Llama if it fails
                match QuantizedGemmaModel::from_gguf(content, &mut file, &device) {
                    Ok(model) => ModelType::Gemma(model),
                    Err(e) if e.to_string().contains("gemma3.attention.head_count") => {
                        // This might be an older Gemma GGUF that uses llama format
                        // Note: Some Gemma GGUF files may not be compatible
                        file.seek(std::io::SeekFrom::Start(0))?;
                        let content = gguf_file::Content::read(&mut file)?;
                        let model = QuantizedLlamaModel::from_gguf(content, &mut file, &device)?;
                        ModelType::Llama(model)
                    }
                    Err(e) => return Err(e),
                }
            }
            _ => {
                return Err(candle_core::Error::Msg(format!(
                    "Unsupported architecture: {}. Supported: llama, mistral, gemma",
                    architecture
                )));
            }
        };
        
        // Detect chat template (for now, use defaults based on architecture)
        let chat_template = Self::detect_chat_template(&tokenizer, &architecture, actual_model_id);
        
        Ok(Self {
            model,
            tokenizer: TokenizerWrapper::new(tokenizer),
            device,
            model_id: actual_model_id.to_string(),
            eos_token_id,
            architecture: architecture.clone(),
            _chat_template: chat_template,
        })
    }
    
    /// Detect architecture from GGUF metadata or model name
    fn detect_architecture(content: &gguf_file::Content, model_id: &str) -> CandleResult<String> {
        // First try to get from metadata
        if let Some(gguf_file::Value::String(arch)) = content.metadata.get("general.architecture") {
            return Ok(arch.clone());
        }
        
        // Fallback to model name detection
        let model_lower = model_id.to_lowercase();
        if model_lower.contains("llama") || model_lower.contains("tinyllama") {
            Ok("llama".to_string())
        } else if model_lower.contains("mistral") {
            Ok("mistral".to_string())
        } else if model_lower.contains("gemma") {
            Ok("gemma".to_string())
        } else {
            Err(candle_core::Error::Msg(
                "Could not determine model architecture from metadata or name".to_string()
            ))
        }
    }
    
    /// Download tokenizer from a specific source
    async fn download_tokenizer_from_source(
        api: &Api,
        source: &str
    ) -> CandleResult<std::path::PathBuf> {
        // Check if it's a local file path
        if source.ends_with(".json") && std::path::Path::new(source).exists() {
            return Ok(std::path::PathBuf::from(source));
        }
        
        // Otherwise treat it as a HuggingFace repo
        let repo = api.model(source.to_string());
        
        // Try tokenizer.json first
        if let Ok(path) = repo.get("tokenizer.json").await {
            return Ok(path);
        }
        
        // Try tokenizer.model (for models that use sentencepiece)
        if let Ok(path) = repo.get("tokenizer.model").await {
            return Ok(path);
        }
        
        Err(candle_core::Error::Msg(format!(
            "Failed to find tokenizer in specified source: {}",
            source
        )))
    }
    
    /// Download tokenizer with architecture-specific fallbacks
    async fn download_tokenizer(
        _api: &Api, 
        repo: &ApiRepo, 
        model_id: &str,
        _architecture: &str
    ) -> CandleResult<std::path::PathBuf> {
        // First try to get tokenizer.json from the GGUF repo
        if let Ok(path) = repo.get("tokenizer.json").await {
            return Ok(path);
        }
        
        // Try tokenizer.model (for models that use sentencepiece)
        if let Ok(path) = repo.get("tokenizer.model").await {
            return Ok(path);
        }
        
        // If no tokenizer found in GGUF repo, return error
        // Ruby will handle the fallback logic
        Err(candle_core::Error::Msg(format!(
            "No tokenizer found in GGUF repository {}. Please specify a tokenizer source.",
            model_id
        )))
    }
    
    /// Determine EOS token based on architecture and model
    fn determine_eos_token(tokenizer: &Tokenizer, architecture: &str, model_id: &str) -> u32 {
        let vocab = tokenizer.get_vocab(true);
        
        match architecture {
            "llama" | "mistral" => {
                // Check if it's Llama 3
                if model_id.contains("Llama-3") || model_id.contains("llama-3") {
                    vocab.get("<|eot_id|>")
                        .or_else(|| vocab.get("<|end_of_text|>"))
                        .copied()
                        .unwrap_or(128009)
                } else {
                    // Llama 2 and Mistral
                    vocab.get("</s>")
                        .copied()
                        .unwrap_or(2)
                }
            }
            "gemma" => {
                vocab.get("<eos>")
                    .or_else(|| vocab.get("<end_of_turn>"))
                    .copied()
                    .unwrap_or(1)
            }
            _ => 2, // Default
        }
    }
    
    /// Detect chat template based on model
    fn detect_chat_template(_tokenizer: &Tokenizer, _architecture: &str, _model_id: &str) -> Option<String> {
        // For now, return None and handle templates in apply_chat_template
        // In the future, this could read from tokenizer config
        None
    }
    
    /// Apply chat template based on detected architecture
    pub fn apply_chat_template(&self, messages: &[serde_json::Value]) -> CandleResult<String> {
        // Check model name since Mistral GGUF reports as llama architecture
        let model_lower = self.model_id.to_lowercase();
        
        if model_lower.contains("mistral") {
            self.apply_mistral_template(messages)
        } else if model_lower.contains("gemma") {
            // Always use Gemma template for Gemma models, regardless of loader used
            self.apply_gemma_template(messages)
        } else {
            match self.architecture.as_str() {
                "llama" => {
                    if self.model_id.contains("Llama-3") || self.model_id.contains("llama-3") {
                        self.apply_llama3_template(messages)
                    } else {
                        self.apply_llama2_template(messages)
                    }
                }
                "gemma" => {
                    self.apply_gemma_template(messages)
                }
                _ => Ok(self.apply_generic_template(messages))
            }
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
    
    fn apply_mistral_template(&self, messages: &[serde_json::Value]) -> CandleResult<String> {
        let mut prompt = String::new();
        
        for message in messages {
            let role = message["role"].as_str().unwrap_or("");
            let content = message["content"].as_str().unwrap_or("");
            
            match role {
                "user" => prompt.push_str(&format!("[INST] {} [/INST]", content)),
                "assistant" => prompt.push_str(&format!(" {}</s>", content)),
                "system" => prompt.push_str(&format!("[INST] {} [/INST]\n", content)),
                _ => {}
            }
        }
        
        Ok(prompt)
    }
    
    fn apply_gemma_template(&self, messages: &[serde_json::Value]) -> CandleResult<String> {
        let mut prompt = String::new();
        
        for message in messages {
            let role = message["role"].as_str().unwrap_or("");
            let content = message["content"].as_str().unwrap_or("");
            
            match role {
                "system" => {
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
        
        prompt.push_str("<start_of_turn>model\n");
        Ok(prompt)
    }
    
    fn apply_generic_template(&self, messages: &[serde_json::Value]) -> String {
        let mut prompt = String::new();
        
        for message in messages {
            let role = message["role"].as_str().unwrap_or("");
            let content = message["content"].as_str().unwrap_or("");
            prompt.push_str(&format!("{}: {}\n", role, content));
        }
        
        prompt.push_str("assistant: ");
        prompt
    }
    
    /// Clear the KV cache between generations
    pub fn clear_kv_cache(&mut self) {
        // Quantized models manage cache internally
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
            
            let logits = match &mut self.model {
                ModelType::Llama(model) => model.forward(&input, start_pos)?,
                ModelType::Gemma(model) => model.forward(&input, start_pos)?,
            };
            
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
                if config.debug_tokens {
                    // In debug mode, show both raw token and decoded text
                    let token_piece = self.tokenizer.token_to_piece(next_token)?;
                    let decoded_text = self.tokenizer.decode_incremental(&all_tokens, all_tokens.len() - 1)?;
                    cb(&format!("[{}:{}]", next_token, token_piece));
                    if !decoded_text.is_empty() {
                        cb(&decoded_text);
                    }
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

impl TextGenerator for QuantizedGGUF {
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
    
    fn clear_cache(&mut self) {
        // Quantized models manage cache internally
    }
}