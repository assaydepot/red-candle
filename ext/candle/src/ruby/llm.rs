use magnus::{function, method, prelude::*, Error, Module, RArray, RHash, RModule, Ruby, TryConvert, Value};
use std::cell::RefCell;
use std::sync::Arc;

use crate::llm::{GenerationConfig as RustGenerationConfig, TextGenerator, mistral::Mistral as RustMistral, llama::Llama as RustLlama, gemma::Gemma as RustGemma, qwen::Qwen as RustQwen, phi::Phi as RustPhi, QuantizedGGUF as RustQuantizedGGUF};
use crate::ruby::{Result, Device};
use crate::ruby::structured::StructuredConstraint;

// Use an enum to handle different model types instead of trait objects
enum ModelType {
    Mistral(RustMistral),
    Llama(RustLlama),
    Gemma(RustGemma),
    Qwen(RustQwen),
    Phi(RustPhi),
    QuantizedGGUF(RustQuantizedGGUF),
}

impl ModelType {
    fn generate(&mut self, prompt: &str, config: &RustGenerationConfig) -> candle_core::Result<String> {
        match self {
            ModelType::Mistral(m) => m.generate(prompt, config),
            ModelType::Llama(m) => m.generate(prompt, config),
            ModelType::Gemma(m) => m.generate(prompt, config),
            ModelType::Qwen(m) => m.generate(prompt, config),
            ModelType::Phi(m) => m.generate(prompt, config),
            ModelType::QuantizedGGUF(m) => m.generate(prompt, config),
        }
    }

    fn generate_stream(
        &mut self,
        prompt: &str,
        config: &RustGenerationConfig,
        callback: impl FnMut(&str),
    ) -> candle_core::Result<String> {
        match self {
            ModelType::Mistral(m) => m.generate_stream(prompt, config, callback),
            ModelType::Llama(m) => m.generate_stream(prompt, config, callback),
            ModelType::Gemma(m) => m.generate_stream(prompt, config, callback),
            ModelType::Qwen(m) => m.generate_stream(prompt, config, callback),
            ModelType::Phi(m) => m.generate_stream(prompt, config, callback),
            ModelType::QuantizedGGUF(m) => m.generate_stream(prompt, config, callback),
        }
    }
    
    fn clear_cache(&mut self) {
        match self {
            ModelType::Mistral(m) => m.clear_cache(),
            ModelType::Llama(m) => m.clear_cache(),
            ModelType::Gemma(m) => m.clear_cache(),
            ModelType::Qwen(m) => m.clear_cache(),
            ModelType::Phi(m) => m.clear_cache(),
            ModelType::QuantizedGGUF(m) => m.clear_cache(),
        }
    }
    
    fn apply_chat_template(&self, messages: &[serde_json::Value]) -> candle_core::Result<String> {
        match self {
            ModelType::Mistral(_) => {
                // For now, use a simple template for Mistral
                // In the future, we could implement proper Mistral chat templating
                let mut prompt = String::new();
                for message in messages {
                    let role = message["role"].as_str().unwrap_or("");
                    let content = message["content"].as_str().unwrap_or("");
                    match role {
                        "system" => prompt.push_str(&format!("System: {}\n\n", content)),
                        "user" => prompt.push_str(&format!("User: {}\n\n", content)),
                        "assistant" => prompt.push_str(&format!("Assistant: {}\n\n", content)),
                        _ => {}
                    }
                }
                prompt.push_str("Assistant: ");
                Ok(prompt)
            },
            ModelType::Llama(m) => m.apply_chat_template(messages),
            ModelType::Gemma(m) => m.apply_chat_template(messages),
            ModelType::Qwen(m) => m.apply_chat_template(messages),
            ModelType::Phi(m) => m.apply_chat_template(messages),
            ModelType::QuantizedGGUF(m) => m.apply_chat_template(messages),
        }
    }
}

// Macro to extract parameters from Ruby hash to reduce boilerplate
macro_rules! extract_param {
    // Basic parameter extraction
    ($kwargs:expr, $config:expr, $param:ident) => {
        if let Some(value) = $kwargs.get(magnus::Symbol::new(stringify!($param))) {
            if let Ok(v) = TryConvert::try_convert(value) {
                $config.$param = v;
            }
        }
    };
    // Optional parameter extraction (wraps in Some)
    ($kwargs:expr, $config:expr, $param:ident, optional) => {
        if let Some(value) = $kwargs.get(magnus::Symbol::new(stringify!($param))) {
            if let Ok(v) = TryConvert::try_convert(value) {
                $config.$param = Some(v);
            }
        }
    };
}

#[derive(Clone, Debug)]
#[magnus::wrap(class = "Candle::GenerationConfig", mark, free_immediately)]
pub struct GenerationConfig {
    inner: RustGenerationConfig,
}

impl GenerationConfig {
    pub fn new(kwargs: RHash) -> Result<Self> {
        let mut config = RustGenerationConfig::default();
        
        // Extract basic parameters using macro
        extract_param!(kwargs, config, max_length);
        extract_param!(kwargs, config, temperature);
        extract_param!(kwargs, config, top_p, optional);
        extract_param!(kwargs, config, top_k, optional);
        extract_param!(kwargs, config, repetition_penalty);
        extract_param!(kwargs, config, repetition_penalty_last_n);
        extract_param!(kwargs, config, seed);
        extract_param!(kwargs, config, include_prompt);
        extract_param!(kwargs, config, debug_tokens);
        extract_param!(kwargs, config, stop_on_constraint_satisfaction);
        extract_param!(kwargs, config, stop_on_match);
        
        // Handle special cases that need custom logic
        if let Some(value) = kwargs.get(magnus::Symbol::new("stop_sequences")) {
            if let Ok(arr) = <RArray as TryConvert>::try_convert(value) {
                config.stop_sequences = arr
                    .into_iter()
                    .filter_map(|v| <String as TryConvert>::try_convert(v).ok())
                    .collect();
            }
        }
        
        if let Some(value) = kwargs.get(magnus::Symbol::new("constraint")) {
            if let Ok(constraint) = <&StructuredConstraint as TryConvert>::try_convert(value) {
                config.constraint = Some(Arc::clone(&constraint.index));
            }
        }
        
        Ok(Self { inner: config })
    }

    pub fn default() -> Self {
        Self {
            inner: RustGenerationConfig::default(),
        }
    }

    // Getters
    pub fn max_length(&self) -> usize {
        self.inner.max_length
    }

    pub fn temperature(&self) -> f64 {
        self.inner.temperature
    }

    pub fn top_p(&self) -> Option<f64> {
        self.inner.top_p
    }

    pub fn top_k(&self) -> Option<usize> {
        self.inner.top_k
    }

    pub fn repetition_penalty(&self) -> f32 {
        self.inner.repetition_penalty
    }

    pub fn seed(&self) -> u64 {
        self.inner.seed
    }

    pub fn stop_sequences(&self) -> Vec<String> {
        self.inner.stop_sequences.clone()
    }

    pub fn include_prompt(&self) -> bool {
        self.inner.include_prompt
    }

    pub fn debug_tokens(&self) -> bool {
        self.inner.debug_tokens
    }
    
    pub fn stop_on_constraint_satisfaction(&self) -> bool {
        self.inner.stop_on_constraint_satisfaction
    }
    
    pub fn stop_on_match(&self) -> bool {
        self.inner.stop_on_match
    }
    
    pub fn constraint(&self) -> Option<StructuredConstraint> {
        self.inner.constraint.as_ref().map(|c| StructuredConstraint {
            index: Arc::clone(c),
        })
    }
    
    /// Get all options as a hash
    pub fn options(&self) -> Result<RHash> {
        let hash = RHash::new();
        
        hash.aset("max_length", self.inner.max_length)?;
        hash.aset("temperature", self.inner.temperature)?;
        
        if let Some(top_p) = self.inner.top_p {
            hash.aset("top_p", top_p)?;
        }
        
        if let Some(top_k) = self.inner.top_k {
            hash.aset("top_k", top_k)?;
        }
        
        hash.aset("repetition_penalty", self.inner.repetition_penalty)?;
        hash.aset("repetition_penalty_last_n", self.inner.repetition_penalty_last_n)?;
        hash.aset("seed", self.inner.seed)?;
        hash.aset("stop_sequences", self.inner.stop_sequences.clone())?;
        hash.aset("include_prompt", self.inner.include_prompt)?;
        hash.aset("debug_tokens", self.inner.debug_tokens)?;
        hash.aset("stop_on_constraint_satisfaction", self.inner.stop_on_constraint_satisfaction)?;
        hash.aset("stop_on_match", self.inner.stop_on_match)?;
        
        if self.inner.constraint.is_some() {
            hash.aset("has_constraint", true)?;
        }
        
        Ok(hash)
    }
}

#[derive(Clone)]
#[magnus::wrap(class = "Candle::LLM", mark, free_immediately)]
pub struct LLM {
    model: std::sync::Arc<std::sync::Mutex<RefCell<ModelType>>>,
    model_id: String,
    device: Device,
}

impl LLM {
    /// Create a new LLM from a pretrained model
    pub fn from_pretrained(model_id: String, device: Option<Device>) -> Result<Self> {
        let device = device.unwrap_or(Device::best());
        let candle_device = device.as_device()?;
        
        // For now, we'll use tokio runtime directly
        // In production, you might want to share a runtime
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to create runtime: {}", e)))?;
        
        // Determine model type from ID and whether it's quantized
        let model_lower = model_id.to_lowercase();
        let is_quantized = model_lower.contains("gguf") || model_lower.contains("-q4") || model_lower.contains("-q5") || model_lower.contains("-q8");
        
        let model = if is_quantized {
            // Extract tokenizer source if provided in model_id
            let (model_id_clean, tokenizer_source) = if let Some(pos) = model_id.find("@@") {
                let (id, _tok) = model_id.split_at(pos);
                (id.to_string(), Some(&model_id[pos+2..]))
            } else {
                (model_id.clone(), None)
            };
            
            // Use unified GGUF loader for all quantized models
            let gguf_model = rt.block_on(async {
                RustQuantizedGGUF::from_pretrained(&model_id_clean, candle_device, tokenizer_source).await
            })
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to load GGUF model: {}", e)))?;
            ModelType::QuantizedGGUF(gguf_model)
        } else {
            // Load non-quantized models
            if model_lower.contains("mistral") {
                let mistral = rt.block_on(async {
                    RustMistral::from_pretrained(&model_id, candle_device).await
                })
                .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to load model: {}", e)))?;
                ModelType::Mistral(mistral)
            } else if model_lower.contains("llama") || model_lower.contains("meta-llama") || model_lower.contains("tinyllama") {
                let llama = rt.block_on(async {
                    RustLlama::from_pretrained(&model_id, candle_device).await
                })
                .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to load model: {}", e)))?;
                ModelType::Llama(llama)
            } else if model_lower.contains("gemma") || model_lower.contains("google/gemma") {
                let gemma = rt.block_on(async {
                    RustGemma::from_pretrained(&model_id, candle_device).await
                })
                .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to load model: {}", e)))?;
                ModelType::Gemma(gemma)
            } else if model_lower.contains("qwen") {
                let qwen = rt.block_on(async {
                    RustQwen::from_pretrained(&model_id, candle_device).await
                })
                .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to load model: {}", e)))?;
                ModelType::Qwen(qwen)
            } else if model_lower.contains("phi") {
                let phi = rt.block_on(async {
                    RustPhi::from_pretrained(&model_id, candle_device).await
                })
                .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to load model: {}", e)))?;
                ModelType::Phi(phi)
            } else {
                return Err(Error::new(
                    magnus::exception::runtime_error(),
                    format!("Unsupported model type: {}. Currently Mistral, Llama, Gemma, Qwen, and Phi models are supported.", model_id),
                ));
            }
        };
        
        Ok(Self {
            model: std::sync::Arc::new(std::sync::Mutex::new(RefCell::new(model))),
            model_id,
            device,
        })
    }

    /// Generate text from a prompt
    pub fn generate(&self, prompt: String, config: Option<&GenerationConfig>) -> Result<String> {
        let config = config
            .map(|c| c.inner.clone())
            .unwrap_or_default();
        
        let model = match self.model.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut model_ref = model.borrow_mut();
        
        model_ref.generate(&prompt, &config)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Generation failed: {}", e)))
    }

    /// Generate text with streaming output
    pub fn generate_stream(&self, prompt: String, config: Option<&GenerationConfig>) -> Result<String> {
        let config = config
            .map(|c| c.inner.clone())
            .unwrap_or_default();
        
        let ruby = Ruby::get().unwrap();
        let block = ruby.block_proc();
        if let Err(_) = block {
            return Err(Error::new(magnus::exception::runtime_error(), "No block given"));
        }
        let block = block.unwrap();
        
        let model = match self.model.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut model_ref = model.borrow_mut();
        
        let result = model_ref.generate_stream(&prompt, &config, |token| {
            // Call the Ruby block with each token
            let _ = block.call::<(String,), Value>((token.to_string(),));
        });
        
        result.map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Generation failed: {}", e)))
    }

    /// Get the model name
    pub fn model_name(&self) -> String {
        self.model_id.clone()
    }

    /// Get the device the model is running on
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get the tokenizer used by this model
    pub fn tokenizer(&self) -> Result<crate::ruby::tokenizer::Tokenizer> {
        let model = match self.model.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let model_ref = model.borrow();
        
        // Clone the tokenizer from the model
        match &*model_ref {
            ModelType::Mistral(m) => Ok(crate::ruby::tokenizer::Tokenizer(m.tokenizer().clone())),
            ModelType::Llama(m) => Ok(crate::ruby::tokenizer::Tokenizer(m.tokenizer().clone())),
            ModelType::Gemma(m) => Ok(crate::ruby::tokenizer::Tokenizer(m.tokenizer().clone())),
            ModelType::Qwen(m) => Ok(crate::ruby::tokenizer::Tokenizer(m.tokenizer().clone())),
            ModelType::Phi(m) => Ok(crate::ruby::tokenizer::Tokenizer(m.tokenizer().clone())),
            ModelType::QuantizedGGUF(m) => Ok(crate::ruby::tokenizer::Tokenizer(m.tokenizer().clone())),
        }
    }

    /// Get the EOS token string for this model
    pub fn eos_token(&self) -> Result<String> {
        let (eos_token_id, tokenizer_clone) = {
            let model = match self.model.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            let model_ref = model.borrow();
            
            // Get both EOS token ID and tokenizer clone in one lock scope
            let eos_id = match &*model_ref {
                ModelType::Mistral(m) => m.eos_token_id(),
                ModelType::Llama(m) => m.eos_token_id(),
                ModelType::Gemma(m) => m.eos_token_id(),
                ModelType::Qwen(m) => m.eos_token_id(),
                ModelType::Phi(m) => m.eos_token_id(),
                ModelType::QuantizedGGUF(m) => m.eos_token_id(),
            };
            
            let tokenizer = match &*model_ref {
                ModelType::Mistral(m) => m.tokenizer().clone(),
                ModelType::Llama(m) => m.tokenizer().clone(),
                ModelType::Gemma(m) => m.tokenizer().clone(),
                ModelType::Qwen(m) => m.tokenizer().clone(),
                ModelType::Phi(m) => m.tokenizer().clone(),
                ModelType::QuantizedGGUF(m) => m.tokenizer().clone(),
            };
            
            (eos_id, tokenizer)
        }; // Lock is released here
        
        // Convert ID to string using the tokenizer
        let tokenizer_wrapper = crate::ruby::tokenizer::Tokenizer(tokenizer_clone);
        tokenizer_wrapper.id_to_token(eos_token_id as i64)
    }
    
    /// Clear the model's cache (e.g., KV cache for transformers)
    pub fn clear_cache(&self) -> Result<()> {
        let model = match self.model.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                // If the mutex is poisoned, we can still recover the data
                // This happens when another thread panicked while holding the lock
                poisoned.into_inner()
            }
        };
        let mut model_ref = model.borrow_mut();
        model_ref.clear_cache();
        Ok(())
    }
    
    /// Apply chat template to messages
    pub fn apply_chat_template(&self, messages: RArray) -> Result<String> {
        // Convert Ruby array to JSON values
        let json_messages: Vec<serde_json::Value> = messages
            .into_iter()
            .filter_map(|msg| {
                if let Ok(hash) = <RHash as TryConvert>::try_convert(msg) {
                    let mut json_msg = serde_json::Map::new();
                    
                    if let Some(role) = hash.get(magnus::Symbol::new("role")) {
                        if let Ok(role_str) = <String as TryConvert>::try_convert(role) {
                            json_msg.insert("role".to_string(), serde_json::Value::String(role_str));
                        }
                    }
                    
                    if let Some(content) = hash.get(magnus::Symbol::new("content")) {
                        if let Ok(content_str) = <String as TryConvert>::try_convert(content) {
                            json_msg.insert("content".to_string(), serde_json::Value::String(content_str));
                        }
                    }
                    
                    Some(serde_json::Value::Object(json_msg))
                } else {
                    None
                }
            })
            .collect();
        
        let model = match self.model.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let model_ref = model.borrow();
        
        model_ref.apply_chat_template(&json_messages)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to apply chat template: {}", e)))
    }
    
    /// Get the model ID
    pub fn model_id(&self) -> String {
        self.model_id.clone()
    }
    
    /// Get model options
    pub fn options(&self) -> Result<RHash> {
        let hash = RHash::new();
        
        // Basic metadata
        hash.aset("model_id", self.model_id.clone())?;
        let device_str = match self.device {
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
            Device::Metal => "metal",
        };
        hash.aset("device", device_str)?;
        
        // Parse model_id to extract GGUF file if present
        if let Some(at_pos) = self.model_id.find('@') {
            let (base_model, gguf_part) = self.model_id.split_at(at_pos);
            let gguf_part = &gguf_part[1..]; // Skip the @ character
            
            // Check for tokenizer (@@)
            if let Some(tokenizer_pos) = gguf_part.find("@@") {
                let (gguf_file, tokenizer) = gguf_part.split_at(tokenizer_pos);
                hash.aset("base_model", base_model)?;
                hash.aset("gguf_file", gguf_file)?;
                hash.aset("tokenizer_source", &tokenizer[2..])?;
            } else {
                hash.aset("base_model", base_model)?;
                hash.aset("gguf_file", gguf_part)?;
            }
        }
        
        // Add model type
        let model = match self.model.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let model_ref = model.borrow();
        
        let model_type = match &*model_ref {
            ModelType::Mistral(_) => "Mistral",
            ModelType::Llama(_) => "Llama",
            ModelType::Gemma(_) => "Gemma",
            ModelType::Qwen(_) => "Qwen",
            ModelType::Phi(_) => "Phi",
            ModelType::QuantizedGGUF(_) => "QuantizedGGUF",
        };
        hash.aset("model_type", model_type)?;
        
        // For GGUF models, add architecture info
        if let ModelType::QuantizedGGUF(gguf) = &*model_ref {
            hash.aset("architecture", gguf.architecture.clone())?;
            hash.aset("eos_token_id", gguf.eos_token_id())?;
        }
        
        Ok(hash)
    }
}

// Define a standalone function for from_pretrained that handles variable arguments
fn from_pretrained_wrapper(args: &[Value]) -> Result<LLM> {
    match args.len() {
        1 => {
            let model_id: String = TryConvert::try_convert(args[0])?;
            LLM::from_pretrained(model_id, None)
        },
        2 => {
            let model_id: String = TryConvert::try_convert(args[0])?;
            let device: Device = TryConvert::try_convert(args[1])?;
            LLM::from_pretrained(model_id, Some(device))
        },
        _ => Err(Error::new(
            magnus::exception::arg_error(),
            "wrong number of arguments (expected 1..2)"
        ))
    }
}

pub fn init_llm(rb_candle: RModule) -> Result<()> {
    let rb_generation_config = rb_candle.define_class("GenerationConfig", magnus::class::object())?;
    rb_generation_config.define_singleton_method("new", function!(GenerationConfig::new, 1))?;
    rb_generation_config.define_singleton_method("default", function!(GenerationConfig::default, 0))?;
    
    rb_generation_config.define_method("max_length", method!(GenerationConfig::max_length, 0))?;
    rb_generation_config.define_method("temperature", method!(GenerationConfig::temperature, 0))?;
    rb_generation_config.define_method("top_p", method!(GenerationConfig::top_p, 0))?;
    rb_generation_config.define_method("top_k", method!(GenerationConfig::top_k, 0))?;
    rb_generation_config.define_method("repetition_penalty", method!(GenerationConfig::repetition_penalty, 0))?;
    rb_generation_config.define_method("seed", method!(GenerationConfig::seed, 0))?;
    rb_generation_config.define_method("stop_sequences", method!(GenerationConfig::stop_sequences, 0))?;
    rb_generation_config.define_method("include_prompt", method!(GenerationConfig::include_prompt, 0))?;
    rb_generation_config.define_method("debug_tokens", method!(GenerationConfig::debug_tokens, 0))?;
    rb_generation_config.define_method("stop_on_constraint_satisfaction", method!(GenerationConfig::stop_on_constraint_satisfaction, 0))?;
    rb_generation_config.define_method("stop_on_match", method!(GenerationConfig::stop_on_match, 0))?;
    rb_generation_config.define_method("constraint", method!(GenerationConfig::constraint, 0))?;
    rb_generation_config.define_method("options", method!(GenerationConfig::options, 0))?;
    
    let rb_llm = rb_candle.define_class("LLM", magnus::class::object())?;
    rb_llm.define_singleton_method("_from_pretrained", function!(from_pretrained_wrapper, -1))?;
    rb_llm.define_method("_generate", method!(LLM::generate, 2))?;
    rb_llm.define_method("_generate_stream", method!(LLM::generate_stream, 2))?;
    rb_llm.define_method("model_name", method!(LLM::model_name, 0))?;
    rb_llm.define_method("device", method!(LLM::device, 0))?;
    rb_llm.define_method("tokenizer", method!(LLM::tokenizer, 0))?;
    rb_llm.define_method("eos_token", method!(LLM::eos_token, 0))?;
    rb_llm.define_method("clear_cache", method!(LLM::clear_cache, 0))?;
    rb_llm.define_method("apply_chat_template", method!(LLM::apply_chat_template, 1))?;
    rb_llm.define_method("model_id", method!(LLM::model_id, 0))?;
    rb_llm.define_method("options", method!(LLM::options, 0))?;
    
    Ok(())
}