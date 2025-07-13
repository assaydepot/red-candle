use candle_core::{DType, Device, Result as CandleResult, Tensor, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::models::mmdit;
use candle_transformers::models::clip;
use candle_transformers::models::t5;
use tokenizers::Tokenizer;
use std::path::Path;

use super::{
    model::{MMDiT, TextEncoders, CLIPTextEncoder, T5TextEncoder, SD3Config},
    vae::{AutoEncoderKL, VAEConfig},
    scheduler::{EulerScheduler, SchedulerConfig},
};

pub struct SD3Pipeline {
    mmdit: MMDiT,
    vae: AutoEncoderKL,
    text_encoders: TextEncoders,
    scheduler_config: SchedulerConfig,
    device: Device,
}

impl SD3Pipeline {
    /// Load from a model file that includes all components
    pub fn from_single_file(
        model_path: &Path,
        device: &Device,
        dtype: DType,
        _use_flash_attn: bool,
    ) -> CandleResult<Self> {
        // Load the safetensors file with FP8 awareness
        let vb = crate::image_gen::fp8_loader::create_fp8_aware_varbuilder(
            &[model_path.to_path_buf()],
            dtype,
            device
        )?;
        
        // Create configs based on model variant
        let (mmdit_config, vae_config) = Self::detect_model_config(&vb)?;
        
        // Load MMDiT
        let mmdit = MMDiT::new(vb.pp("model.diffusion_model"), &mmdit_config, device)?;
        
        // Load VAE - try different possible prefixes
        let vae = match AutoEncoderKL::new(vb.pp("first_stage_model"), vae_config.clone()) {
            Ok(vae) => vae,
            Err(_) => match AutoEncoderKL::new(vb.pp("vae"), vae_config) {
                Ok(vae) => vae,
                Err(e) => return Err(candle_core::Error::Msg(
                    format!("Failed to load VAE: {}. VAE weights not found under 'first_stage_model' or 'vae' prefix", e)
                )),
            }
        };
        
        // Load text encoders from the model file if available
        let text_encoders = Self::load_text_encoders(&vb, device)?;
        
        let scheduler_config = SchedulerConfig::default();
        
        Ok(Self {
            mmdit,
            vae,
            text_encoders,
            scheduler_config,
            device: device.clone(),
        })
    }
    
    /// Load from separate component files
    pub fn from_components(
        _mmdit_path: &Path,
        _vae_path: Option<&Path>,
        _clip_g_path: Option<&Path>,
        _clip_l_path: Option<&Path>,
        _t5_path: Option<&Path>,
        _device: &Device,
        _dtype: DType,
    ) -> CandleResult<Self> {
        // This would load each component separately
        // For now, return an error as this is complex
        Err(candle_core::Error::Msg(
            "Component-based loading not yet implemented".to_string()
        ))
    }
    
    /// Detect model configuration from loaded weights
    fn detect_model_config(_vb: &VarBuilder) -> CandleResult<(mmdit::model::Config, VAEConfig)> {
        // This is a simplified detection - real implementation would check tensor shapes
        // For now, assume SD3-medium configuration
        
        let mmdit_config = mmdit::model::Config::sd3_medium();
        let vae_config = VAEConfig::default();
        
        Ok((mmdit_config, vae_config))
    }
    
    pub fn generate(
        &mut self,
        prompt: &str,
        negative_prompt: Option<&str>,
        config: &SD3Config,
        seed: Option<u64>,
        mut progress_callback: Option<&mut dyn FnMut(usize, usize, Option<&Tensor>)>,
    ) -> CandleResult<Tensor> {
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
            
            // Predict noise
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
        
        // Decode latents
        let images = self.vae.decode(&latents)?;
        
        // Convert from [B, C, H, W] to [H, W, C] for single image
        let image = images.i(0)?; // Get first image
        let image = image.permute((1, 2, 0))?; // [C, H, W] -> [H, W, C]
        
        Ok(image)
    }
    
    /// Load text encoders from the model file
    fn load_text_encoders(vb: &VarBuilder, device: &Device) -> CandleResult<TextEncoders> {
        // Try to load text encoders and track what's available
        let mut encoders_found = false;
        
        // Try to load CLIP-G
        let clip_g = match Self::load_clip_g(vb, device) {
            Ok(encoder) => {
                encoders_found = true;
                eprintln!("Loaded CLIP-G text encoder");
                Some(encoder)
            }
            Err(_) => {
                None
            }
        };
        
        // Try to load CLIP-L
        let clip_l = match Self::load_clip_l(vb, device) {
            Ok(encoder) => {
                encoders_found = true;
                eprintln!("Loaded CLIP-L text encoder");
                Some(encoder)
            }
            Err(_) => {
                None
            }
        };
        
        // Try to load T5
        let t5 = match Self::load_t5(vb, device) {
            Ok(encoder) => {
                encoders_found = true;
                eprintln!("Loaded T5-XXL text encoder");
                Some(encoder)
            }
            Err(_) => {
                None
            }
        };
        
        // If no encoders loaded successfully, return dummy ones
        if !encoders_found {
            eprintln!("No text encoders found in model file. Using dummy encoders.");
            eprintln!("For full functionality, use a model file that includes text encoders:");
            eprintln!("  - sd3_medium_incl_clips_t5xxlfp16.safetensors (includes all text encoders)");
            eprintln!("  - sd3_medium_incl_clips.safetensors (includes CLIP encoders only)");
            return TextEncoders::dummy(device);
        }
        
        Ok(TextEncoders {
            clip_g,
            clip_l,
            t5,
        })
    }
    
    /// Load CLIP-G encoder
    fn load_clip_g(vb: &VarBuilder, device: &Device) -> CandleResult<CLIPTextEncoder> {
        // Try different prefixes
        let prefixes = ["conditioner.embedders.0", "text_encoder", "cond_stage_model.0"];
        
        for prefix in &prefixes {
            if let Ok(encoder) = Self::try_load_clip_g_from_prefix(vb, device, prefix) {
                return Ok(encoder);
            }
        }
        
        Err(candle_core::Error::Msg("CLIP-G encoder not found under any known prefix".to_string()))
    }
    
    fn try_load_clip_g_from_prefix(vb: &VarBuilder, device: &Device, prefix: &str) -> CandleResult<CLIPTextEncoder> {
        
        // CLIP-G config (OpenCLIP ViT-bigG-14)
        let config = clip::text_model::ClipTextConfig {
            vocab_size: 49408,
            embed_dim: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_attention_heads: 20,
            max_position_embeddings: 77,
            activation: clip::text_model::Activation::QuickGelu,
            projection_dim: 1280,
            pad_with: None,
        };
        
        // Load tokenizer (would need to download separately)
        let tokenizer = Self::load_default_tokenizer()?;
        
        CLIPTextEncoder::new(
            vb.pp(format!("{}.transformer", prefix)),
            &config,
            tokenizer,
            device,
        )
    }
    
    /// Load CLIP-L encoder
    fn load_clip_l(vb: &VarBuilder, device: &Device) -> CandleResult<CLIPTextEncoder> {
        // Try different prefixes
        let prefixes = ["conditioner.embedders.1", "text_encoder_2", "cond_stage_model.1"];
        
        for prefix in &prefixes {
            if let Ok(encoder) = Self::try_load_clip_l_from_prefix(vb, device, prefix) {
                return Ok(encoder);
            }
        }
        
        Err(candle_core::Error::Msg("CLIP-L encoder not found under any known prefix".to_string()))
    }
    
    fn try_load_clip_l_from_prefix(vb: &VarBuilder, device: &Device, prefix: &str) -> CandleResult<CLIPTextEncoder> {
        
        // CLIP-L config (OpenAI CLIP ViT-L/14)
        let config = clip::text_model::ClipTextConfig {
            vocab_size: 49408,
            embed_dim: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            max_position_embeddings: 77,
            activation: clip::text_model::Activation::QuickGelu,
            projection_dim: 768,
            pad_with: None,
        };
        
        // Load tokenizer
        let tokenizer = Self::load_default_tokenizer()?;
        
        CLIPTextEncoder::new(
            vb.pp(format!("{}.transformer", prefix)),
            &config,
            tokenizer,
            device,
        )
    }
    
    /// Load T5-XXL encoder
    fn load_t5(vb: &VarBuilder, device: &Device) -> CandleResult<T5TextEncoder> {
        // Try different prefixes
        let prefixes = ["conditioner.embedders.2.transformer", "text_encoder_3", "t5"];
        
        for prefix in &prefixes {
            if let Ok(encoder) = Self::try_load_t5_from_prefix(vb, device, prefix) {
                return Ok(encoder);
            }
        }
        
        Err(candle_core::Error::Msg("T5 encoder not found under any known prefix".to_string()))
    }
    
    fn try_load_t5_from_prefix(vb: &VarBuilder, device: &Device, prefix: &str) -> CandleResult<T5TextEncoder> {
        
        // T5-XXL config
        let config = t5::Config {
            vocab_size: 32128,
            d_ff: 10240,
            d_kv: 64,
            d_model: 4096,
            num_heads: 64,
            num_layers: 24,
            num_decoder_layers: None,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: t5::ActivationWithOptionalGating {
                gated: true,
                activation: candle_nn::Activation::NewGelu,
            },
            tie_word_embeddings: false,
            is_decoder: false,
            is_encoder_decoder: false,
            use_cache: false,
            decoder_start_token_id: Some(0),
            eos_token_id: 1,
            pad_token_id: 0,
        };
        
        // Load tokenizer
        let tokenizer = Self::load_t5_tokenizer()?;
        
        T5TextEncoder::new(
            vb.pp(prefix),
            &config,
            tokenizer,
            device,
        )
    }
    
    /// Load default CLIP tokenizer
    fn load_default_tokenizer() -> CandleResult<Tokenizer> {
        // For now, create a simple tokenizer
        // In production, this would download the actual tokenizer
        use tokenizers::{Tokenizer, models::bpe::BPE};
        
        let tokenizer = Tokenizer::new(BPE::default());
        Ok(tokenizer)
    }
    
    /// Load T5 tokenizer
    fn load_t5_tokenizer() -> CandleResult<Tokenizer> {
        // For now, create a simple tokenizer
        // In production, this would download the T5 tokenizer
        use tokenizers::{Tokenizer, models::bpe::BPE};
        
        let tokenizer = Tokenizer::new(BPE::default());
        Ok(tokenizer)
    }
}