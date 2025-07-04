use magnus::{function, method, prelude::*, Error, Module, RArray, RHash, RModule, Ruby, TryConvert, Value};
use std::cell::RefCell;

use crate::llm::{GenerationConfig as RustGenerationConfig, TextGenerator, mistral::Mistral as RustMistral};
use crate::ruby::{Result as RbResult, Device as RbDevice};

// Use an enum to handle different model types instead of trait objects
#[derive(Debug)]
enum ModelType {
    Mistral(RustMistral),
}

impl ModelType {
    fn generate(&mut self, prompt: &str, config: &RustGenerationConfig) -> candle_core::Result<String> {
        match self {
            ModelType::Mistral(m) => m.generate(prompt, config),
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
        }
    }

    #[allow(dead_code)]
    fn model_name(&self) -> &str {
        match self {
            ModelType::Mistral(m) => m.model_name(),
        }
    }
    
    fn clear_cache(&mut self) {
        match self {
            ModelType::Mistral(m) => m.clear_cache(),
        }
    }
}

#[derive(Clone, Debug)]
#[magnus::wrap(class = "Candle::GenerationConfig", mark, free_immediately)]
pub struct GenerationConfig {
    inner: RustGenerationConfig,
}

impl GenerationConfig {
    pub fn new(kwargs: RHash) -> RbResult<Self> {
        let mut config = RustGenerationConfig::default();
        
        // Extract values from kwargs manually
        if let Some(value) = kwargs.get(magnus::Symbol::new("max_length")) {
            if let Ok(v) = TryConvert::try_convert(value) {
                config.max_length = v;
            }
        }
        
        if let Some(value) = kwargs.get(magnus::Symbol::new("temperature")) {
            if let Ok(v) = TryConvert::try_convert(value) {
                config.temperature = v;
            }
        }
        
        if let Some(value) = kwargs.get(magnus::Symbol::new("top_p")) {
            if let Ok(v) = TryConvert::try_convert(value) {
                config.top_p = Some(v);
            }
        }
        
        if let Some(value) = kwargs.get(magnus::Symbol::new("top_k")) {
            if let Ok(v) = TryConvert::try_convert(value) {
                config.top_k = Some(v);
            }
        }
        
        if let Some(value) = kwargs.get(magnus::Symbol::new("repetition_penalty")) {
            if let Ok(v) = TryConvert::try_convert(value) {
                config.repetition_penalty = v;
            }
        }
        
        if let Some(value) = kwargs.get(magnus::Symbol::new("repetition_penalty_last_n")) {
            if let Ok(v) = TryConvert::try_convert(value) {
                config.repetition_penalty_last_n = v;
            }
        }
        
        if let Some(value) = kwargs.get(magnus::Symbol::new("seed")) {
            if let Ok(v) = TryConvert::try_convert(value) {
                config.seed = v;
            }
        }
        
        if let Some(value) = kwargs.get(magnus::Symbol::new("include_prompt")) {
            if let Ok(v) = TryConvert::try_convert(value) {
                config.include_prompt = v;
            }
        }
        
        if let Some(value) = kwargs.get(magnus::Symbol::new("stop_sequences")) {
            if let Ok(arr) = <RArray as TryConvert>::try_convert(value) {
                config.stop_sequences = arr
                    .into_iter()
                    .filter_map(|v| <String as TryConvert>::try_convert(v).ok())
                    .collect();
            }
        }
        
        Ok(Self { inner: config })
    }

    pub fn default() -> Self {
        Self {
            inner: RustGenerationConfig::default(),
        }
    }

    pub fn deterministic() -> Self {
        Self {
            inner: RustGenerationConfig::deterministic(),
        }
    }

    pub fn creative() -> Self {
        Self {
            inner: RustGenerationConfig::creative(),
        }
    }

    pub fn balanced() -> Self {
        Self {
            inner: RustGenerationConfig::balanced(),
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
}

#[derive(Clone, Debug)]
#[magnus::wrap(class = "Candle::LLM", mark, free_immediately)]
pub struct LLM {
    model: std::sync::Arc<std::sync::Mutex<RefCell<ModelType>>>,
    model_id: String,
    device: RbDevice,
}

impl LLM {
    /// Create a new LLM from a pretrained model
    pub fn from_pretrained(model_id: String, device: Option<RbDevice>) -> RbResult<Self> {
        let device = device.unwrap_or(RbDevice::Cpu);
        let candle_device = device.as_device()?;
        
        // For now, we'll use tokio runtime directly
        // In production, you might want to share a runtime
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to create runtime: {}", e)))?;
        
        // Determine model type from ID and load appropriately
        let model_lower = model_id.to_lowercase();
        let model = if model_lower.contains("mistral") {
            let mistral = rt.block_on(async {
                RustMistral::from_pretrained(&model_id, candle_device).await
            })
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to load model: {}", e)))?;
            ModelType::Mistral(mistral)
        } else {
            return Err(Error::new(
                magnus::exception::runtime_error(),
                format!("Unsupported model type: {}. Currently only Mistral models are supported.", model_id),
            ));
        };
        
        Ok(Self {
            model: std::sync::Arc::new(std::sync::Mutex::new(RefCell::new(model))),
            model_id,
            device,
        })
    }

    /// Generate text from a prompt
    pub fn generate(&self, prompt: String, config: Option<&GenerationConfig>) -> RbResult<String> {
        let config = config
            .map(|c| c.inner.clone())
            .unwrap_or_default();
        
        let model = self.model.lock().unwrap();
        let mut model_ref = model.borrow_mut();
        
        model_ref.generate(&prompt, &config)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Generation failed: {}", e)))
    }

    /// Generate text with streaming output
    pub fn generate_stream(&self, prompt: String, config: Option<&GenerationConfig>) -> RbResult<String> {
        let config = config
            .map(|c| c.inner.clone())
            .unwrap_or_default();
        
        let ruby = Ruby::get().unwrap();
        let block = ruby.block_proc();
        if let Err(_) = block {
            return Err(Error::new(magnus::exception::runtime_error(), "No block given"));
        }
        let block = block.unwrap();
        
        let model = self.model.lock().unwrap();
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
    pub fn device(&self) -> RbDevice {
        self.device
    }
    
    /// Clear the model's cache (e.g., KV cache for transformers)
    pub fn clear_cache(&self) -> RbResult<()> {
        let model = self.model.lock().unwrap();
        let mut model_ref = model.borrow_mut();
        model_ref.clear_cache();
        Ok(())
    }
}

// Define a standalone function for from_pretrained that handles variable arguments
fn from_pretrained_wrapper(args: &[Value]) -> RbResult<LLM> {
    match args.len() {
        1 => {
            let model_id: String = TryConvert::try_convert(args[0])?;
            LLM::from_pretrained(model_id, None)
        },
        2 => {
            let model_id: String = TryConvert::try_convert(args[0])?;
            let device: RbDevice = TryConvert::try_convert(args[1])?;
            LLM::from_pretrained(model_id, Some(device))
        },
        _ => Err(Error::new(
            magnus::exception::arg_error(),
            "wrong number of arguments (expected 1..2)"
        ))
    }
}

pub fn init_llm(rb_candle: RModule) -> RbResult<()> {
    let rb_generation_config = rb_candle.define_class("GenerationConfig", magnus::class::object())?;
    rb_generation_config.define_singleton_method("new", function!(GenerationConfig::new, 1))?;
    rb_generation_config.define_singleton_method("default", function!(GenerationConfig::default, 0))?;
    rb_generation_config.define_singleton_method("deterministic", function!(GenerationConfig::deterministic, 0))?;
    rb_generation_config.define_singleton_method("creative", function!(GenerationConfig::creative, 0))?;
    rb_generation_config.define_singleton_method("balanced", function!(GenerationConfig::balanced, 0))?;
    
    rb_generation_config.define_method("max_length", method!(GenerationConfig::max_length, 0))?;
    rb_generation_config.define_method("temperature", method!(GenerationConfig::temperature, 0))?;
    rb_generation_config.define_method("top_p", method!(GenerationConfig::top_p, 0))?;
    rb_generation_config.define_method("top_k", method!(GenerationConfig::top_k, 0))?;
    rb_generation_config.define_method("repetition_penalty", method!(GenerationConfig::repetition_penalty, 0))?;
    rb_generation_config.define_method("seed", method!(GenerationConfig::seed, 0))?;
    rb_generation_config.define_method("stop_sequences", method!(GenerationConfig::stop_sequences, 0))?;
    rb_generation_config.define_method("include_prompt", method!(GenerationConfig::include_prompt, 0))?;
    
    let rb_llm = rb_candle.define_class("LLM", magnus::class::object())?;
    rb_llm.define_singleton_method("from_pretrained", function!(from_pretrained_wrapper, -1))?;
    rb_llm.define_method("generate", method!(LLM::generate, 2))?;
    rb_llm.define_method("generate_stream", method!(LLM::generate_stream, 2))?;
    rb_llm.define_method("model_name", method!(LLM::model_name, 0))?;
    rb_llm.define_method("device", method!(LLM::device, 0))?;
    rb_llm.define_method("clear_cache", method!(LLM::clear_cache, 0))?;
    
    Ok(())
}