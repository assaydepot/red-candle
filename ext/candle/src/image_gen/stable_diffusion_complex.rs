use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_core::quantized::gguf_file;
use hf_hub::api::tokio::Api;
use std::path::Path;

use crate::image_gen::{ImageGenerationConfig, ImageGenerator, GenerationProgress, tensor_to_png};
use crate::image_gen::sd3::{SD3Pipeline, ThreadSafeSD3Pipeline, SD3Config};

/// Model file preferences for automatic selection
const MODEL_PREFERENCES: &[(&str, &[&str])] = &[
    ("stabilityai/stable-diffusion-3-medium", &[
        "sd3_medium_incl_clips_t5xxlfp8.safetensors",    // Default: smallest complete
        "sd3_medium_incl_clips_t5xxlfp16.safetensors",   // Higher precision
        "sd3_medium_incl_clips.safetensors",             // Needs T5
        "sd3_medium.safetensors"                         // Needs CLIP + T5
    ]),
    ("stabilityai/stable-diffusion-3.5-large", &[
        "sd3.5_large_incl_clips_t5xxlfp8.safetensors",
        "sd3.5_large_incl_clips_t5xxlfp16.safetensors",
        "sd3.5_large_incl_clips.safetensors",
        "sd3.5_large.safetensors"
    ]),
];

const GGUF_PREFERENCES: &[(&str, &[&str])] = &[
    ("second-state/stable-diffusion-3-medium-GGUF", &[
        "sd3-medium-Q5_0.gguf",    // 5-bit, ~5.53 GB (good balance)
        "sd3-medium-Q4_0.gguf",    // 4-bit, ~4.55 GB (smallest)
        "sd3-medium-Q8_0.gguf",    // 8-bit, ~8.45 GB (high quality)
        "sd3-medium-f16.gguf",     // fp16, ~15.8 GB (best quality)
    ]),
];

#[derive(Debug)]
pub struct StableDiffusion3 {
    model_type: ModelType,
    model_id: String,
    device: Device,
}

#[derive(Debug)]
enum ModelType {
    Safetensors(SafetensorsModel),
    GGUF(GGUFModel),
}

#[derive(Debug)]
struct SafetensorsModel {
    pipeline: ThreadSafeSD3Pipeline,
    config: SD3Config,
}

#[derive(Debug)]
struct GGUFModel {
    content: gguf_file::Content,
    // config: ModelConfig,
    // tokenizers: Tokenizers,
}

impl StableDiffusion3 {
    /// Load a Stable Diffusion 3 model
    pub async fn from_pretrained(
        model_id: &str,
        device: Device,
        model_file: Option<&str>,
        gguf_file: Option<&str>,
        clip_model: Option<&str>,
        t5_model: Option<&str>,
        config_source: Option<&str>,
        tokenizer_source: Option<&str>,
    ) -> CandleResult<Self> {
        // Determine if this is a GGUF model
        let is_gguf = model_id.contains("GGUF") || 
                      gguf_file.is_some() || 
                      model_file.map(|f| f.ends_with(".gguf")).unwrap_or(false);
        
        let model_type = if is_gguf {
            Self::load_gguf_model(
                model_id,
                gguf_file.or(model_file),
                config_source,
                tokenizer_source,
                &device,
            ).await?
        } else {
            Self::load_safetensors_model(
                model_id,
                model_file,
                clip_model,
                t5_model,
                &device,
            ).await?
        };
        
        Ok(Self {
            model_type,
            model_id: model_id.to_string(),
            device,
        })
    }
    
    async fn load_gguf_model(
        model_id: &str,
        gguf_file: Option<&str>,
        config_source: Option<&str>,
        tokenizer_source: Option<&str>,
        _device: &Device,
    ) -> CandleResult<ModelType> {
        let api = Api::new()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create HF API: {}", e)))?;
        
        let repo = api.model(model_id.to_string());
        
        // Determine GGUF file to use
        let gguf_filename = if let Some(file) = gguf_file {
            file
        } else {
            Self::detect_best_gguf_file(model_id)?
        };
        
        // Download GGUF file
        let gguf_path = repo.get(gguf_filename).await
            .map_err(|e| candle_core::Error::Msg(format!(
                "Failed to download GGUF file '{}': {}",
                gguf_filename, e
            )))?;
        
        // Load GGUF content
        let mut file = std::fs::File::open(&gguf_path)?;
        let content = gguf_file::Content::read(&mut file)?;
        
        // Detect model type from GGUF metadata
        let model_arch = Self::detect_model_architecture(&content)?;
        if !model_arch.starts_with("sd3") && model_arch != "stable-diffusion-3" {
            return Err(candle_core::Error::Msg(format!(
                "GGUF file contains unsupported architecture: {}",
                model_arch
            )));
        }
        
        // TODO: Load config and tokenizers from config_source
        let _config_repo = config_source.unwrap_or("stabilityai/stable-diffusion-3-medium");
        let _tokenizer_repo = tokenizer_source.unwrap_or(_config_repo);
        
        Ok(ModelType::GGUF(GGUFModel {
            content,
        }))
    }
    
    async fn load_safetensors_model(
        model_id: &str,
        model_file: Option<&str>,
        clip_model: Option<&str>,
        t5_model: Option<&str>,
        _device: &Device,
    ) -> CandleResult<ModelType> {
        let api = Api::new()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create HF API: {}", e)))?;
        
        let repo = api.model(model_id.to_string());
        
        // Determine model file to use
        let model_filename = if let Some(file) = model_file {
            file
        } else {
            Self::detect_best_model_file(model_id)?
        };
        
        // Download model file
        let _model_path = repo.get(model_filename).await
            .map_err(|e| candle_core::Error::Msg(format!(
                "Failed to download model file '{}': {}",
                model_filename, e
            )))?;
        
        // Determine what additional components are needed
        let needs_clip = !model_filename.contains("incl_clips");
        let needs_t5 = !model_filename.contains("t5xxl");
        
        // TODO: Actually load the models
        if needs_clip {
            let _clip_source = clip_model.unwrap_or("openai/clip-vit-large-patch14");
            // Download and load CLIP models
        }
        
        if needs_t5 {
            let _t5_source = t5_model.unwrap_or("google/t5-v1_1-xxl");
            // Download and load T5 model
        }
        
        // Load the pipeline
        let pipeline = SD3Pipeline::from_single_file(
            &_model_path,
            _device,
            DType::F32,
            false, // flash_attn
        )?;
        
        let thread_safe_pipeline = ThreadSafeSD3Pipeline::new(pipeline);
        let config = SD3Config::default();
        
        Ok(ModelType::Safetensors(SafetensorsModel {
            pipeline: thread_safe_pipeline,
            config,
        }))
    }
    
    fn detect_best_model_file(model_id: &str) -> CandleResult<&'static str> {
        for (id, files) in MODEL_PREFERENCES {
            if model_id == *id {
                return Ok(files[0]); // Return the first (preferred) option
            }
        }
        
        Err(candle_core::Error::Msg(format!(
            "No default model file configured for {}. Please specify model_file parameter.",
            model_id
        )))
    }
    
    fn detect_best_gguf_file(model_id: &str) -> CandleResult<&'static str> {
        for (id, files) in GGUF_PREFERENCES {
            if model_id == *id {
                return Ok(files[0]); // Return the first (preferred) option
            }
        }
        
        Err(candle_core::Error::Msg(format!(
            "No default GGUF file configured for {}. Please specify gguf_file parameter.",
            model_id
        )))
    }
    
    fn detect_model_architecture(content: &gguf_file::Content) -> CandleResult<String> {
        // Check GGUF metadata for model architecture
        if let Some(gguf_file::Value::String(arch)) = content.metadata.get("general.architecture") {
            return Ok(arch.clone());
        }
        
        // Fallback: infer from tensor names
        let has_joint_blocks = content.tensor_infos.keys().any(|k| k.contains("joint_blocks"));
        let has_sd3_tensors = content.tensor_infos.keys().any(|k| k.contains("sd3") || k.contains("mmdit"));
        
        if has_joint_blocks || has_sd3_tensors {
            Ok("stable-diffusion-3".to_string())
        } else {
            Err(candle_core::Error::Msg(
                "Cannot determine model architecture from GGUF metadata or tensor names".to_string()
            ))
        }
    }
}

impl ImageGenerator for StableDiffusion3 {
    fn generate(&mut self, prompt: &str, config: &ImageGenerationConfig) -> CandleResult<Vec<u8>> {
        match &self.model_type {
            ModelType::Safetensors(model) => {
                // Convert config
                let mut sd3_config = model.config.clone();
                sd3_config.width = config.width;
                sd3_config.height = config.height;
                sd3_config.num_inference_steps = config.num_inference_steps;
                sd3_config.cfg_scale = config.guidance_scale;
                sd3_config.use_t5 = config.use_t5;
                sd3_config.clip_skip = config.clip_skip;
                
                // Generate image
                let image_tensor = model.pipeline.generate(
                    prompt,
                    config.negative_prompt.as_deref(),
                    &sd3_config,
                    config.seed,
                    None,
                )?;
                
                // Convert tensor to PNG
                tensor_to_png(&image_tensor)
            }
            ModelType::GGUF(_) => {
                // GGUF generation not yet implemented
                Err(candle_core::Error::Msg(
                    "GGUF model generation not yet implemented".to_string()
                ))
            }
        }
    }
    
    fn generate_stream(
        &mut self,
        prompt: &str,
        config: &ImageGenerationConfig,
        mut callback: impl FnMut(GenerationProgress),
    ) -> CandleResult<Vec<u8>> {
        match &mut self.model_type {
            ModelType::Safetensors(model) => {
                // Convert config
                let mut sd3_config = model.config.clone();
                sd3_config.width = config.width;
                sd3_config.height = config.height;
                sd3_config.num_inference_steps = config.num_inference_steps;
                sd3_config.cfg_scale = config.guidance_scale;
                sd3_config.use_t5 = config.use_t5;
                sd3_config.clip_skip = config.clip_skip;
                
                let preview_interval = config.preview_interval.unwrap_or(5);
                let mut final_image = None;
                
                // Generate with progress callback
                let image_tensor = model.pipeline.generate(
                    prompt,
                    config.negative_prompt.as_deref(),
                    &sd3_config,
                    config.seed,
                    Some(&mut |step, total, preview_latents| {
                        let image_data = if step % preview_interval == 0 {
                            // Decode preview if latents provided
                            if let Some(latents) = preview_latents {
                                // For preview, just use a simplified decode
                                // Real implementation would properly decode through VAE
                                None // TODO: Implement preview decoding
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        
                        callback(GenerationProgress {
                            step,
                            total_steps: total,
                            image_data,
                        });
                    }),
                )?;
                
                // Convert final tensor to PNG
                tensor_to_png(&image_tensor)
            }
            ModelType::GGUF(_) => {
                // GGUF generation not yet implemented
                Err(candle_core::Error::Msg(
                    "GGUF model streaming not yet implemented".to_string()
                ))
            }
        }
    }
    
    fn model_name(&self) -> &str {
        &self.model_id
    }
    
    fn device(&self) -> &Device {
        &self.device
    }
    
    fn clear_cache(&mut self) {
        // TODO: Clear any cached data
    }
}