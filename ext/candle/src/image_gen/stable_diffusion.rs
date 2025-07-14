use candle_core::{DType, Device, Result as CandleResult, Tensor};
use hf_hub::api::tokio::Api;
use std::sync::{Arc, Mutex};
use std::path::PathBuf;

use crate::image_gen::{ImageGenerationConfig, GenerationProgress, tensor_to_png};
use super::sd3::{ThreadSafeSD3Pipeline, SD3Pipeline};
use super::sd3::model::SD3Config;
use super::gguf::{QuantizedSD3Pipeline};

/// Stable Diffusion 3 implementation
pub struct StableDiffusion3 {
    model_id: String,
    device: Device,
    dtype: DType,
    pipeline: Option<PipelineType>,
}

enum PipelineType {
    Standard(Arc<Mutex<ThreadSafeSD3Pipeline>>),
    Quantized(Arc<Mutex<QuantizedSD3Pipeline>>),
}

impl std::fmt::Debug for StableDiffusion3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StableDiffusion3")
            .field("model_id", &self.model_id)
            .field("device", &self.device)
            .field("dtype", &self.dtype)
            .field("pipeline", &self.pipeline.is_some())
            .finish()
    }
}

impl StableDiffusion3 {
    pub async fn from_pretrained(
        model_id: &str,
        device: Device,
        model_file: Option<&str>,
        gguf_file: Option<&str>,
        _clip_model: Option<&str>,
        _t5_model: Option<&str>,
        _config_source: Option<&str>,
        _tokenizer_source: Option<&str>,
    ) -> CandleResult<Self> {
        // For now, verify we can access the model files
        // Full SD3 implementation to be added when the sd3 module is ready
        let api = Api::new()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create HF API: {}", e)))?;
        
        let repo = api.model(model_id.to_string());
        
        // Determine if this is GGUF or safetensors
        let is_gguf = model_id.contains("GGUF") || 
                      gguf_file.is_some() || 
                      model_file.map(|f| f.ends_with(".gguf")).unwrap_or(false);
        
        let pipeline = if is_gguf {
            // Download GGUF file
            let gguf_filename = gguf_file.unwrap_or("sd3-medium-Q5_0.gguf");
            let gguf_path = repo.get(gguf_filename).await
                .map_err(|e| candle_core::Error::Msg(format!(
                    "Failed to download GGUF file '{}': {}",
                    gguf_filename, e
                )))?;
            
            // Try to load the quantized SD3 pipeline
            match QuantizedSD3Pipeline::from_gguf_file(&gguf_path, &device) {
                Ok(quantized_pipeline) => {
                    eprintln!("Successfully loaded quantized SD3 pipeline from GGUF!");
                    Some(PipelineType::Quantized(Arc::new(Mutex::new(quantized_pipeline))))
                }
                Err(e) => {
                    eprintln!("Warning: Failed to load quantized SD3 pipeline: {}. Using placeholder.", e);
                    None
                }
            }
        } else {
            // Download safetensors file
            let model_filename = model_file.unwrap_or("sd3_medium_incl_clips_t5xxlfp8.safetensors");
            let model_path = repo.get(model_filename).await
                .map_err(|e| candle_core::Error::Msg(format!(
                    "Failed to download model file '{}': {}",
                    model_filename, e
                )))?;
            
            // Try to load the actual SD3 pipeline
            match Self::load_sd3_pipeline(&model_path, &device, DType::F32) {
                Ok(pipeline) => {
                    eprintln!("Successfully loaded SD3 pipeline!");
                    Some(PipelineType::Standard(Arc::new(Mutex::new(pipeline))))
                }
                Err(e) => {
                    eprintln!("Warning: Failed to load SD3 pipeline: {}. Using placeholder.", e);
                    if e.to_string().contains("FP8") {
                        eprintln!("\nTo use SD3 with Candle, try one of these options:");
                        eprintln!("1. Use the FP16 model (20GB):");
                        eprintln!("   model_file: \"sd3_medium_incl_clips_t5xxlfp16.safetensors\"");
                        eprintln!("2. Use the base model without text encoders (5GB):");
                        eprintln!("   model_file: \"sd3_medium.safetensors\"");
                    }
                    None
                }
            }
        };
        
        Ok(Self {
            model_id: model_id.to_string(),
            device,
            dtype: DType::F32,
            pipeline,
        })
    }
    
    fn load_sd3_pipeline(model_path: &PathBuf, device: &Device, dtype: DType) -> CandleResult<ThreadSafeSD3Pipeline> {
        // Load the SD3 pipeline from the single file
        let pipeline = SD3Pipeline::from_single_file(
            model_path,
            device,
            dtype,
            false, // use_flash_attn
        )?;
        
        Ok(ThreadSafeSD3Pipeline::new(pipeline))
    }
    
    fn generate_placeholder_image(&self, config: &ImageGenerationConfig) -> CandleResult<Tensor> {
        let height = config.height;
        let width = config.width;
        
        // Create a more sophisticated placeholder based on the seed
        let mut data = vec![0f32; height * width * 3];
        let seed = config.seed.unwrap_or(42);
        
        // Use seed to create variation
        let pattern = (seed % 4) as usize;
        let color_shift = (seed as f32 / 100.0) % 1.0;
        
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                let fx = x as f32 / width as f32;
                let fy = y as f32 / height as f32;
                
                match pattern {
                    0 => {
                        // Gradient pattern
                        data[idx] = fx;
                        data[idx + 1] = fy;
                        data[idx + 2] = (fx + fy) / 2.0;
                    }
                    1 => {
                        // Wave pattern
                        let wave = ((fx * 10.0 + color_shift * 20.0).sin() + 1.0) / 2.0;
                        data[idx] = wave * fx;
                        data[idx + 1] = wave * (1.0 - fx);
                        data[idx + 2] = wave * 0.7;
                    }
                    2 => {
                        // Radial gradient
                        let cx = 0.5;
                        let cy = 0.5;
                        let dist = ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt();
                        let normalized_dist = (dist * 2.0).min(1.0);
                        data[idx] = 1.0 - normalized_dist;
                        data[idx + 1] = normalized_dist * 0.5;
                        data[idx + 2] = 0.3 + normalized_dist * 0.4;
                    }
                    _ => {
                        // Noise-like pattern
                        let noise = ((fx * 7.0 + seed as f32).sin() * (fy * 13.0 + seed as f32).cos() + 1.0) / 2.0;
                        data[idx] = noise * (0.8 + color_shift * 0.2);
                        data[idx + 1] = noise * 0.5;
                        data[idx + 2] = noise * (0.3 + (1.0 - color_shift) * 0.4);
                    }
                }
            }
        }
        
        // Create tensor on CPU
        Tensor::from_vec(
            data,
            &[height, width, 3],
            &Device::Cpu
        )
    }
}

// TODO: Fix Send/Sync issues with MMDiT before re-enabling
impl StableDiffusion3 {
    pub fn generate(&mut self, prompt: &str, config: &ImageGenerationConfig) -> CandleResult<Vec<u8>> {
        // Log the prompt for debugging
        eprintln!("\n=== StableDiffusion3::generate called ===");
        eprintln!("Generating image for prompt: {}", prompt);
        eprintln!("Config: {}x{}, steps: {}", config.width, config.height, config.num_inference_steps);
        eprintln!("Pipeline type: {}", match &self.pipeline {
            Some(PipelineType::Standard(_)) => "Standard",
            Some(PipelineType::Quantized(_)) => "Quantized",
            None => "None (placeholder)",
        });
        
        if let Some(pipeline) = &self.pipeline {
            match pipeline {
                PipelineType::Standard(standard_pipeline) => {
                    // Use the standard SD3 pipeline
                    let sd3_config = SD3Config {
                        width: config.width,
                        height: config.height,
                        num_inference_steps: config.num_inference_steps,
                        cfg_scale: config.guidance_scale,
                        time_shift: 3.0, // Default time shift for SD3
                        use_t5: true,
                        clip_skip: config.clip_skip,
                    };
                    
                    let pipeline = standard_pipeline.lock().unwrap();
                    let image_tensor = pipeline.generate(
                        prompt,
                        config.negative_prompt.as_deref(),
                        &sd3_config,
                        config.seed,
                        None, // No progress callback for non-streaming
                    )?;
                    
                    // Convert to PNG
                    tensor_to_png(&image_tensor)
                }
                PipelineType::Quantized(quantized_pipeline) => {
                    eprintln!("\n>>> Using QUANTIZED pipeline <<<");
                    // Use the quantized SD3 pipeline
                    let mut pipeline = quantized_pipeline.lock().unwrap();
                    eprintln!(">>> Calling quantized generate_image <<<");
                    let result = pipeline.generate_image(prompt, config);
                    eprintln!(">>> Quantized generate_image returned: {:?} <<<", result.is_ok());
                    result
                }
            }
        } else {
            // Fallback to placeholder
            let image_tensor = self.generate_placeholder_image(config)?;
            tensor_to_png(&image_tensor)
        }
    }
    
    pub fn generate_stream(
        &mut self,
        prompt: &str,
        config: &ImageGenerationConfig,
        mut callback: impl FnMut(GenerationProgress),
    ) -> CandleResult<Vec<u8>> {
        if let Some(pipeline) = &self.pipeline {
            match pipeline {
                PipelineType::Standard(standard_pipeline) => {
                    // Use the standard SD3 pipeline with streaming
                    let sd3_config = SD3Config {
                        width: config.width,
                        height: config.height,
                        num_inference_steps: config.num_inference_steps,
                        cfg_scale: config.guidance_scale,
                        time_shift: 3.0,
                        use_t5: true,
                        clip_skip: config.clip_skip,
                    };
                    
                    let mut progress_fn = |step: usize, total_steps: usize, preview: Option<&Tensor>| {
                        callback(GenerationProgress {
                            step,
                            total_steps,
                            image_data: preview.and_then(|t| tensor_to_png(t).ok()),
                        });
                    };
                    
                    let pipeline = standard_pipeline.lock().unwrap();
                    let image_tensor = pipeline.generate(
                        prompt,
                        config.negative_prompt.as_deref(),
                        &sd3_config,
                        config.seed,
                        Some(&mut progress_fn),
                    )?;
                    
                    // Convert to PNG
                    tensor_to_png(&image_tensor)
                }
                PipelineType::Quantized(quantized_pipeline) => {
                    // Use the quantized SD3 pipeline with streaming
                    let mut pipeline = quantized_pipeline.lock().unwrap();
                    pipeline.generate_image_stream(prompt, config, callback)
                }
            }
        } else {
            // Fallback to placeholder with simulated progress
            let total_steps = config.num_inference_steps;
            let preview_interval = config.preview_interval.unwrap_or(5);
            
            for step in 0..total_steps {
                if step % preview_interval == 0 || step == total_steps - 1 {
                    callback(GenerationProgress {
                        step: step + 1,
                        total_steps,
                        image_data: None,
                    });
                }
            }
            
            self.generate(prompt, config)
        }
    }
    
    pub fn model_name(&self) -> &str {
        &self.model_id
    }
    
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    pub fn clear_cache(&mut self) {
        // Nothing to clear yet
    }
}