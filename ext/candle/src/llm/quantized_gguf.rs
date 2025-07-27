use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_llama::ModelWeights as QuantizedLlamaModel;
use candle_transformers::models::quantized_gemma3::ModelWeights as QuantizedGemmaModel;
use candle_transformers::models::quantized_qwen2::ModelWeights as QuantizedQwenModel;
use candle_transformers::models::quantized_phi::ModelWeights as QuantizedPhiModel;
use candle_transformers::models::quantized_phi3::ModelWeights as QuantizedPhi3Model;
use hf_hub::api::tokio::{Api, ApiRepo};
use tokenizers::Tokenizer;
use std::io::Seek;

use crate::llm::{GenerationConfig, TextGeneration, TextGenerator, TokenizerWrapper};

/// Unified GGUF model that can load any GGUF file and detect the architecture
pub struct QuantizedGGUF {
    model: ModelType,
    tokenizer: TokenizerWrapper,
    device: Device,
    model_id: String,
    eos_token_id: u32,
    architecture: String,
    _chat_template: Option<String>,
}

enum ModelType {
    Llama(QuantizedLlamaModel),
    Gemma(QuantizedGemmaModel),
    Qwen(QuantizedQwenModel),
    Phi(QuantizedPhiModel),
    Phi3(QuantizedPhi3Model),
    // Mistral uses Llama loader due to tensor naming compatibility
}

impl QuantizedGGUF {
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get the tokenizer
    pub fn tokenizer(&self) -> &TokenizerWrapper {
        &self.tokenizer
    }
    
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
            "qwen" | "qwen2" | "qwen3" => {
                // Try different loaders based on what metadata is available
                if content.metadata.contains_key("llama.attention.head_count") {
                    let model = QuantizedLlamaModel::from_gguf(content, &mut file, &device)?;
                    ModelType::Llama(model)
                } else if content.metadata.contains_key("qwen2.attention.head_count") {
                    let model = QuantizedQwenModel::from_gguf(content, &mut file, &device)?;
                    ModelType::Qwen(model)
                } else if content.metadata.contains_key("qwen3.attention.head_count") {
                    // Qwen3 GGUF files use a different metadata format
                    // The quantized_qwen3 module is not yet in the released version of candle-transformers
                    return Err(candle_core::Error::Msg(format!(
                        "Qwen3 GGUF format detected but not yet fully supported.\n\n\
                        The file contains qwen3.* metadata keys which require candle-transformers > 0.9.1.\n\n\
                        Current alternatives:\n\
                        1. Use Qwen2.5 GGUF models which work well:\n\
                           - Qwen/Qwen2.5-7B-Instruct-GGUF (recommended)\n\
                           - Qwen/Qwen2.5-32B-Instruct-GGUF\n\
                        2. Use non-quantized Qwen models with safetensors\n\
                        3. Wait for candle-transformers update with quantized_qwen3 support\n\n\
                        Note: Qwen2.5 models have similar capabilities to Qwen3."
                    )));
                } else {
                    // Last resort: try llama loader anyway, as it's the most common
                    let model = QuantizedLlamaModel::from_gguf(content, &mut file, &device)?;
                    ModelType::Llama(model)
                }
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
            "phi" | "phi2" => {
                let model = QuantizedPhiModel::from_gguf(content, &mut file, &device)?;
                ModelType::Phi(model)
            }
            "phi3" => {
                // QuantizedPhi3Model requires an additional `approx` parameter
                // Setting to false to avoid performance issues without flash-attn
                let approx = false;
                let model = QuantizedPhi3Model::from_gguf(approx, content, &mut file, &device)?;
                ModelType::Phi3(model)
            }
            _ => {
                return Err(candle_core::Error::Msg(format!(
                    "Unsupported architecture: {}. Supported: llama, mistral, gemma, qwen, qwen2, qwen3, phi, phi2, phi3",
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
        } else if model_lower.contains("qwen") {
            Ok("qwen".to_string())
        } else if model_lower.contains("phi-3") || model_lower.contains("phi3") {
            Ok("phi3".to_string())
        } else if model_lower.contains("phi-2") || model_lower.contains("phi2") {
            Ok("phi2".to_string())
        } else if model_lower.contains("phi") {
            Ok("phi".to_string())
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
            "qwen" | "qwen2" | "qwen3" => {
                vocab.get("<|endoftext|>")
                    .or_else(|| vocab.get("<|im_end|>"))
                    .or_else(|| vocab.get("</s>"))
                    .copied()
                    .unwrap_or(151643) // Default Qwen3 EOS token
            }
            "phi" | "phi2" | "phi3" => {
                vocab.get("<|endoftext|>")
                    .or_else(|| vocab.get("<|end|>"))
                    .or_else(|| vocab.get("</s>"))
                    .copied()
                    .unwrap_or(50256) // Default GPT-2 style EOS token
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
        } else if model_lower.contains("qwen") {
            self.apply_qwen_template(messages)
        } else if model_lower.contains("phi") {
            self.apply_phi_template(messages)
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
                "qwen" | "qwen2" | "qwen3" => {
                    self.apply_qwen_template(messages)
                }
                "phi" | "phi2" | "phi3" => {
                    self.apply_phi_template(messages)
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
    
    fn apply_qwen_template(&self, messages: &[serde_json::Value]) -> CandleResult<String> {
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
    
    fn apply_phi_template(&self, messages: &[serde_json::Value]) -> CandleResult<String> {
        let mut prompt = String::new();
        
        // Check if it's Phi-3 (newer format) or Phi-2/Phi (simpler format)
        let is_phi3 = self.model_id.contains("phi-3") || self.model_id.contains("Phi-3") || self.architecture == "phi3";
        
        if is_phi3 {
            // Phi-3 format
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
            // Phi-2 format
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
        // Quantized models don't expose cache clearing methods
        // Phi3 GGUF models have a known issue where the KV cache
        // cannot be cleared, leading to errors on subsequent generations
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
                ModelType::Qwen(model) => model.forward(&input, start_pos)?,
                ModelType::Phi(model) => model.forward(&input, start_pos)?,
                ModelType::Phi3(model) => model.forward(&input, start_pos)?,
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

impl TextGenerator for QuantizedGGUF {
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
        // Quantized models manage cache internally
    }
}