use candle_core::{DType, Device, Result as CandleResult, Tensor, IndexOp};
use std::path::Path;

use crate::image_gen::{ImageGenerationConfig, GenerationProgress, tensor_to_png};
use super::{GGUFMetadata, ComponentType, QuantizedMMDiT, QuantizedVAE, load_component_tensors};
use crate::image_gen::sd3::{SchedulerConfig, EulerScheduler};
use crate::image_gen::sd3::model::SD3Config;
use crate::image_gen::sd3::model::TextEncoders;

/// Quantized Stable Diffusion 3 pipeline using GGUF models
pub struct QuantizedSD3Pipeline {
    mmdit: QuantizedMMDiT,
    vae: QuantizedVAE,
    text_encoders: TextEncoders,
    scheduler_config: SchedulerConfig,
    device: Device,
    metadata: GGUFMetadata,
}

impl QuantizedSD3Pipeline {
    /// Load a quantized SD3 pipeline from a GGUF file
    pub fn from_gguf_file(path: &Path, device: &Device) -> CandleResult<Self> {
        eprintln!("Loading quantized SD3 pipeline from GGUF: {}", path.display());
        
        // Parse GGUF metadata
        let metadata = GGUFMetadata::from_file(path)?;
        
        // Log model information
        eprintln!("Model: {} ({})", 
            metadata.model_name.as_deref().unwrap_or("Unknown"),
            metadata.total_size_human()
        );
        eprintln!("Architecture: {}", metadata.architecture);
        eprintln!("Quantization version: {}", metadata.quantization_version);
        eprintln!("Components found:");
        
        for (component_type, info) in &metadata.components {
            eprintln!("  - {:?}: {} tensors, {:.1} MB", 
                component_type, 
                info.tensor_count,
                info.size_bytes as f64 / (1024.0 * 1024.0)
            );
        }
        
        // Validate required components
        if !metadata.has_component(&ComponentType::MMDiT) {
            return Err(candle_core::Error::Msg(
                "GGUF file does not contain MMDiT component".to_string()
            ));
        }
        
        if !metadata.has_component(&ComponentType::VAE) {
            return Err(candle_core::Error::Msg(
                "GGUF file does not contain VAE component".to_string()
            ));
        }
        
        // Load MMDiT component
        eprintln!("Loading quantized MMDiT...");
        let (mmdit_content, mut mmdit_file, mmdit_device) = load_component_tensors(path, &ComponentType::MMDiT, device)?;
        let mmdit = QuantizedMMDiT::from_gguf(mmdit_content, &mut mmdit_file, &mmdit_device)?;
        
        // Load VAE component
        eprintln!("Loading quantized VAE...");
        let (vae_content, mut vae_file, vae_device) = load_component_tensors(path, &ComponentType::VAE, device)?;
        let vae = QuantizedVAE::from_gguf(vae_content, &mut vae_file, &vae_device)?;
        
        // Load text encoders (or use dummy ones)
        let text_encoders = Self::load_text_encoders(&metadata, path, device)?;
        
        let scheduler_config = SchedulerConfig::default();
        
        Ok(Self {
            mmdit,
            vae,
            text_encoders,
            scheduler_config,
            device: device.clone(),
            metadata,
        })
    }
    
    /// Load text encoders from GGUF file or create dummy ones
    fn load_text_encoders(
        metadata: &GGUFMetadata,
        _path: &Path,
        device: &Device,
    ) -> CandleResult<TextEncoders> {
        // Check which text encoders are available
        let has_clip_g = metadata.has_component(&ComponentType::CLIPTextG);
        let has_clip_l = metadata.has_component(&ComponentType::CLIPTextL);
        let has_t5 = metadata.has_component(&ComponentType::T5Text);
        
        if !has_clip_g && !has_clip_l && !has_t5 {
            eprintln!("No text encoders found in GGUF file. Using dummy encoders.");
            eprintln!("For full text conditioning, use a GGUF model that includes text encoders");
            eprintln!("or load text encoders separately.");
            return TextEncoders::dummy(device);
        }
        
        // For now, use dummy encoders even if text encoders are present
        // TODO: Implement quantized text encoder loading
        eprintln!("Text encoders found but quantized text encoder loading not yet implemented.");
        eprintln!("Available encoders:");
        if has_clip_g {
            if let Some(info) = metadata.get_component(&ComponentType::CLIPTextG) {
                eprintln!("  - CLIP-G: {} tensors", info.tensor_count);
            }
        }
        if has_clip_l {
            if let Some(info) = metadata.get_component(&ComponentType::CLIPTextL) {
                eprintln!("  - CLIP-L: {} tensors", info.tensor_count);
            }
        }
        if has_t5 {
            if let Some(info) = metadata.get_component(&ComponentType::T5Text) {
                eprintln!("  - T5-XXL: {} tensors", info.tensor_count);
            }
        }
        eprintln!("Using dummy encoders for now.");
        
        TextEncoders::dummy(device)
    }
    
    /// Generate an image using the quantized pipeline
    pub fn generate(
        &mut self,
        prompt: &str,
        negative_prompt: Option<&str>,
        config: &SD3Config,
        seed: Option<u64>,
        mut progress_callback: Option<&mut dyn FnMut(usize, usize, Option<&Tensor>)>,
    ) -> CandleResult<Tensor> {
        eprintln!("Generating with quantized SD3 pipeline...");
        eprintln!("Prompt: {}", prompt);
        eprintln!("Config: {}x{}, {} steps", config.width, config.height, config.num_inference_steps);
        
        // Text encoding
        let (context_cond, context_uncond, pooled) = self.text_encoders.encode(
            prompt,
            negative_prompt,
            config.use_t5,
        )?;
        
        // Combine for classifier-free guidance
        let context = Tensor::cat(&[&context_uncond, &context_cond], 0)?;
        let y = Tensor::cat(&[&pooled, &pooled], 0)?;
        
        // Initialize scheduler
        let scheduler = EulerScheduler::new(config.num_inference_steps, self.scheduler_config.clone())?;
        
        // Initialize latents
        let mut latents = scheduler.init_noise(
            1, // batch size
            16, // SD3 latent channels
            config.height,
            config.width,
            &self.device,
            DType::F32,
            seed,
        )?;
        
        // Scale initial noise
        latents = scheduler.scale_noise(&latents, 0)?;
        
        // Diffusion loop
        let timesteps = scheduler.timesteps();
        for (i, &timestep) in timesteps.iter().enumerate() {
            // Expand latents for CFG
            let latent_model_input = Tensor::cat(&[&latents, &latents], 0)?;
            
            // Create timestep tensor
            let t = Tensor::new(&[timestep, timestep], &self.device)?;
            
            // Predict noise using quantized MMDiT
            let noise_pred = self.mmdit.forward(
                &latent_model_input,
                &t,
                &context,
                &y,
            )?;
            
            // Perform CFG
            let chunks = noise_pred.chunk(2, 0)?;
            let noise_pred_uncond = &chunks[0];
            let noise_pred_cond = &chunks[1];
            let noise_pred = (noise_pred_uncond + (noise_pred_cond - noise_pred_uncond)? * config.cfg_scale)?;
            
            // Scheduler step
            latents = scheduler.step(&noise_pred, i, &latents)?;
            
            // Progress callback
            if let Some(callback) = progress_callback.as_mut() {
                let preview = if i % 5 == 0 {
                    Some(&latents)
                } else {
                    None
                };
                callback(i + 1, config.num_inference_steps, preview);
            }
        }
        
        // Decode latents using quantized VAE
        eprintln!("Decoding latents with quantized VAE...");
        let images = self.vae.decode(&latents)?;
        
        // Convert from [B, C, H, W] to [H, W, C] for single image
        let image = images.i(0)?; // Get first image
        let image = image.permute((1, 2, 0))?; // [C, H, W] -> [H, W, C]
        
        Ok(image)
    }
    
    /// Get model metadata
    pub fn metadata(&self) -> &GGUFMetadata {
        &self.metadata
    }
    
    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Convert QuantizedSD3Pipeline to be compatible with ImageGenerationConfig
impl QuantizedSD3Pipeline {
    pub fn generate_image(&mut self, prompt: &str, config: &ImageGenerationConfig) -> CandleResult<Vec<u8>> {
        let sd3_config = SD3Config {
            width: config.width,
            height: config.height,
            num_inference_steps: config.num_inference_steps,
            cfg_scale: config.guidance_scale,
            time_shift: 3.0, // Default time shift for SD3
            use_t5: true,
            clip_skip: config.clip_skip,
        };
        
        let image_tensor = self.generate(
            prompt,
            config.negative_prompt.as_deref(),
            &sd3_config,
            config.seed,
            None, // No progress callback for simple generation
        )?;
        
        // Convert to PNG
        tensor_to_png(&image_tensor)
    }
    
    pub fn generate_image_stream(
        &mut self,
        prompt: &str,
        config: &ImageGenerationConfig,
        mut callback: impl FnMut(GenerationProgress),
    ) -> CandleResult<Vec<u8>> {
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
        
        let image_tensor = self.generate(
            prompt,
            config.negative_prompt.as_deref(),
            &sd3_config,
            config.seed,
            Some(&mut progress_fn),
        )?;
        
        // Convert to PNG
        tensor_to_png(&image_tensor)
    }
}