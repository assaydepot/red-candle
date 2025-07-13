use magnus::{function, method, prelude::*, Error, Module, RHash, RModule, Ruby, TryConvert, Value};
use std::cell::RefCell;

use crate::image_gen::{
    ImageGenerationConfig as RustImageGenerationConfig,
    ImageGenerator as ImageGeneratorTrait, 
    StableDiffusion3 as RustStableDiffusion3,
    SchedulerType,
};
use crate::ruby::{Result as RbResult, Device as RbDevice};

#[derive(Debug)]
enum ModelType {
    StableDiffusion3(RustStableDiffusion3),
    // Future: Flux, SDXL, etc.
}

impl ModelType {
    fn generate(&mut self, prompt: &str, config: &RustImageGenerationConfig) -> candle_core::Result<Vec<u8>> {
        match self {
            ModelType::StableDiffusion3(m) => m.generate(prompt, config),
        }
    }

    fn generate_stream(
        &mut self,
        prompt: &str,
        config: &RustImageGenerationConfig,
        callback: impl FnMut(crate::image_gen::GenerationProgress),
    ) -> candle_core::Result<Vec<u8>> {
        match self {
            ModelType::StableDiffusion3(m) => m.generate_stream(prompt, config, callback),
        }
    }
    
    fn clear_cache(&mut self) {
        match self {
            ModelType::StableDiffusion3(m) => m.clear_cache(),
        }
    }
    
    fn model_name(&self) -> &str {
        match self {
            ModelType::StableDiffusion3(m) => m.model_name(),
        }
    }
    
    fn device(&self) -> &candle_core::Device {
        match self {
            ModelType::StableDiffusion3(m) => m.device(),
        }
    }
}

#[derive(Clone, Debug)]
#[magnus::wrap(class = "Candle::ImageGenerationConfig", mark, free_immediately)]
pub struct ImageGenerationConfig {
    inner: RustImageGenerationConfig,
}

impl ImageGenerationConfig {
    pub fn new(kwargs: RHash) -> RbResult<Self> {
        let mut config = RustImageGenerationConfig::default();
        
        // Extract values from kwargs
        if let Some(value) = kwargs.get(magnus::Symbol::new("height")) {
            if let Ok(v) = TryConvert::try_convert(value) {
                config.height = v;
            }
        }
        
        if let Some(value) = kwargs.get(magnus::Symbol::new("width")) {
            if let Ok(v) = TryConvert::try_convert(value) {
                config.width = v;
            }
        }
        
        if let Some(value) = kwargs.get(magnus::Symbol::new("num_inference_steps")) {
            if let Ok(v) = TryConvert::try_convert(value) {
                config.num_inference_steps = v;
            }
        }
        
        if let Some(value) = kwargs.get(magnus::Symbol::new("guidance_scale")) {
            if let Ok(v) = TryConvert::try_convert(value) {
                config.guidance_scale = v;
            }
        }
        
        if let Some(value) = kwargs.get(magnus::Symbol::new("negative_prompt")) {
            if let Ok(v) = TryConvert::try_convert(value) {
                config.negative_prompt = Some(v);
            }
        }
        
        if let Some(value) = kwargs.get(magnus::Symbol::new("seed")) {
            if let Ok(v) = TryConvert::try_convert(value) {
                config.seed = Some(v);
            }
        }
        
        if let Some(value) = kwargs.get(magnus::Symbol::new("preview_interval")) {
            if let Ok(v) = TryConvert::try_convert(value) {
                config.preview_interval = Some(v);
            }
        }
        
        if let Some(value) = kwargs.get(magnus::Symbol::new("scheduler")) {
            if let Ok(v) = <String as TryConvert>::try_convert(value) {
                config.scheduler = match v.as_str() {
                    "euler" => SchedulerType::Euler,
                    "euler_a" => SchedulerType::EulerA,
                    "dpm_solver" => SchedulerType::DPMSolver,
                    "ddim" => SchedulerType::DDIM,
                    _ => SchedulerType::Euler,
                };
            }
        }
        
        if let Some(value) = kwargs.get(magnus::Symbol::new("use_t5")) {
            if let Ok(v) = TryConvert::try_convert(value) {
                config.use_t5 = v;
            }
        }
        
        if let Some(value) = kwargs.get(magnus::Symbol::new("clip_skip")) {
            if let Ok(v) = TryConvert::try_convert(value) {
                config.clip_skip = v;
            }
        }
        
        Ok(Self { inner: config })
    }

    pub fn default() -> Self {
        Self {
            inner: RustImageGenerationConfig::default(),
        }
    }

    // Getters
    pub fn height(&self) -> usize {
        self.inner.height
    }

    pub fn width(&self) -> usize {
        self.inner.width
    }

    pub fn num_inference_steps(&self) -> usize {
        self.inner.num_inference_steps
    }

    pub fn guidance_scale(&self) -> f64 {
        self.inner.guidance_scale
    }

    pub fn negative_prompt(&self) -> Option<String> {
        self.inner.negative_prompt.clone()
    }

    pub fn seed(&self) -> Option<u64> {
        self.inner.seed
    }

    pub fn preview_interval(&self) -> Option<usize> {
        self.inner.preview_interval
    }

    pub fn scheduler(&self) -> String {
        match self.inner.scheduler {
            SchedulerType::Euler => "euler".to_string(),
            SchedulerType::EulerA => "euler_a".to_string(),
            SchedulerType::DPMSolver => "dpm_solver".to_string(),
            SchedulerType::DDIM => "ddim".to_string(),
        }
    }

    pub fn use_t5(&self) -> bool {
        self.inner.use_t5
    }

    pub fn clip_skip(&self) -> usize {
        self.inner.clip_skip
    }
}

#[derive(Clone, Debug)]
#[magnus::wrap(class = "Candle::ImageGenerator", mark, free_immediately)]
pub struct ImageGenerator {
    model: std::sync::Arc<std::sync::Mutex<RefCell<ModelType>>>,
    model_id: String,
    device: RbDevice,
}

impl ImageGenerator {
    /// Create a new ImageGenerator from a pretrained model
    pub fn from_pretrained(
        model_id: String,
        device: Option<RbDevice>,
        model_file: Option<String>,
        gguf_file: Option<String>,
        clip_model: Option<String>,
        t5_model: Option<String>,
        config_source: Option<String>,
        tokenizer_source: Option<String>,
    ) -> RbResult<Self> {
        let device = device.unwrap_or(RbDevice::Cpu);
        let candle_device = device.as_device()?;
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to create runtime: {}", e)))?;
        
        // Parse model_id if it contains @ separators
        let (actual_model_id, parsed_gguf, parsed_config) = if model_id.contains('@') {
            let parts: Vec<&str> = model_id.split("@@").collect();
            let model_parts: Vec<&str> = parts[0].split('@').collect();
            (
                model_parts[0].to_string(),
                model_parts.get(1).map(|s| s.to_string()).or(gguf_file),
                parts.get(1).map(|s| s.to_string()).or(config_source),
            )
        } else {
            (model_id.clone(), gguf_file, config_source)
        };
        
        // Determine model type and load
        let model = if actual_model_id.to_lowercase().contains("stable-diffusion-3") || 
                       actual_model_id.to_lowercase().contains("sd3") {
            let sd3 = rt.block_on(async {
                RustStableDiffusion3::from_pretrained(
                    &actual_model_id,
                    candle_device,
                    model_file.as_deref(),
                    parsed_gguf.as_deref(),
                    clip_model.as_deref(),
                    t5_model.as_deref(),
                    parsed_config.as_deref(),
                    tokenizer_source.as_deref(),
                ).await
            })
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to load model: {}", e)))?;
            ModelType::StableDiffusion3(sd3)
        } else {
            return Err(Error::new(
                magnus::exception::runtime_error(),
                format!("Unsupported model type: {}. Currently only Stable Diffusion 3 models are supported.", actual_model_id),
            ));
        };
        
        Ok(Self {
            model: std::sync::Arc::new(std::sync::Mutex::new(RefCell::new(model))),
            model_id: actual_model_id,
            device,
        })
    }

    /// Generate an image from a prompt
    pub fn generate(&self, prompt: String, config: Option<&ImageGenerationConfig>) -> RbResult<Vec<u8>> {
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

    /// Generate an image with streaming progress
    pub fn generate_stream(&self, prompt: String, config: Option<&ImageGenerationConfig>) -> RbResult<Vec<u8>> {
        let config = config
            .map(|c| c.inner.clone())
            .unwrap_or_default();
        
        let ruby = Ruby::get().unwrap();
        let block = ruby.block_proc()
            .map_err(|_| Error::new(magnus::exception::runtime_error(), "No block given"))?;
        
        let model = match self.model.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut model_ref = model.borrow_mut();
        
        let result = model_ref.generate_stream(&prompt, &config, |progress| {
            // Create a hash with progress information
            let hash = ruby.hash_new();
            let _ = hash.aset("step", progress.step);
            let _ = hash.aset("total_steps", progress.total_steps);
            
            if let Some(image_data) = progress.image_data {
                let _ = hash.aset("image_data", image_data);
            }
            
            // Call the Ruby block with progress
            let _ = block.call::<(RHash,), Value>((hash,));
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
    
    /// Clear the model's cache
    pub fn clear_cache(&self) -> RbResult<()> {
        let model = match self.model.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut model_ref = model.borrow_mut();
        model_ref.clear_cache();
        Ok(())
    }
}

// Wrapper function for from_pretrained that handles variable arguments
fn from_pretrained_wrapper(args: &[Value]) -> RbResult<ImageGenerator> {
    let kwargs = args.last()
        .and_then(|v| <RHash as TryConvert>::try_convert(*v).ok());
    
    let model_id: String = TryConvert::try_convert(args[0])?;
    
    let device = kwargs.as_ref()
        .and_then(|h| h.get(magnus::Symbol::new("device")))
        .and_then(|v| <RbDevice as TryConvert>::try_convert(v).ok());
    
    let model_file = kwargs.as_ref()
        .and_then(|h| h.get(magnus::Symbol::new("model_file")))
        .and_then(|v| <String as TryConvert>::try_convert(v).ok());
    
    let gguf_file = kwargs.as_ref()
        .and_then(|h| h.get(magnus::Symbol::new("gguf_file")))
        .and_then(|v| <String as TryConvert>::try_convert(v).ok());
    
    let clip_model = kwargs.as_ref()
        .and_then(|h| h.get(magnus::Symbol::new("clip_model")))
        .and_then(|v| <String as TryConvert>::try_convert(v).ok());
    
    let t5_model = kwargs.as_ref()
        .and_then(|h| h.get(magnus::Symbol::new("t5_model")))
        .and_then(|v| <String as TryConvert>::try_convert(v).ok());
    
    let config_source = kwargs.as_ref()
        .and_then(|h| h.get(magnus::Symbol::new("config_source")))
        .and_then(|v| <String as TryConvert>::try_convert(v).ok());
    
    let tokenizer_source = kwargs.as_ref()
        .and_then(|h| h.get(magnus::Symbol::new("tokenizer_source")))
        .and_then(|v| <String as TryConvert>::try_convert(v).ok());
    
    ImageGenerator::from_pretrained(
        model_id,
        device,
        model_file,
        gguf_file,
        clip_model,
        t5_model,
        config_source,
        tokenizer_source,
    )
}

pub fn init_image_gen(rb_candle: RModule) -> RbResult<()> {
    let rb_generation_config = rb_candle.define_class("ImageGenerationConfig", magnus::class::object())?;
    rb_generation_config.define_singleton_method("new", function!(ImageGenerationConfig::new, 1))?;
    rb_generation_config.define_singleton_method("default", function!(ImageGenerationConfig::default, 0))?;
    
    rb_generation_config.define_method("height", method!(ImageGenerationConfig::height, 0))?;
    rb_generation_config.define_method("width", method!(ImageGenerationConfig::width, 0))?;
    rb_generation_config.define_method("num_inference_steps", method!(ImageGenerationConfig::num_inference_steps, 0))?;
    rb_generation_config.define_method("guidance_scale", method!(ImageGenerationConfig::guidance_scale, 0))?;
    rb_generation_config.define_method("negative_prompt", method!(ImageGenerationConfig::negative_prompt, 0))?;
    rb_generation_config.define_method("seed", method!(ImageGenerationConfig::seed, 0))?;
    rb_generation_config.define_method("preview_interval", method!(ImageGenerationConfig::preview_interval, 0))?;
    rb_generation_config.define_method("scheduler", method!(ImageGenerationConfig::scheduler, 0))?;
    rb_generation_config.define_method("use_t5", method!(ImageGenerationConfig::use_t5, 0))?;
    rb_generation_config.define_method("clip_skip", method!(ImageGenerationConfig::clip_skip, 0))?;
    
    let rb_image_generator = rb_candle.define_class("ImageGenerator", magnus::class::object())?;
    rb_image_generator.define_singleton_method("_from_pretrained", function!(from_pretrained_wrapper, -1))?;
    rb_image_generator.define_method("_generate", method!(ImageGenerator::generate, 2))?;
    rb_image_generator.define_method("_generate_stream", method!(ImageGenerator::generate_stream, 2))?;
    rb_image_generator.define_method("model_name", method!(ImageGenerator::model_name, 0))?;
    rb_image_generator.define_method("device", method!(ImageGenerator::device, 0))?;
    rb_image_generator.define_method("clear_cache", method!(ImageGenerator::clear_cache, 0))?;
    
    Ok(())
}