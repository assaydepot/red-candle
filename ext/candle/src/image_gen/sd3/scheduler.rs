use candle_core::{DType, Device, Result as CandleResult, Tensor};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub num_train_timesteps: usize,
    pub shift: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            shift: 3.0,
        }
    }
}

pub struct EulerScheduler {
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    #[allow(dead_code)]
    config: SchedulerConfig,
}

impl EulerScheduler {
    pub fn new(num_inference_steps: usize, config: SchedulerConfig) -> CandleResult<Self> {
        let sigmas = Self::noise_schedule(
            num_inference_steps,
            config.num_train_timesteps,
            config.shift,
        );
        
        let timesteps = sigmas
            .iter()
            .map(|&sigma| (sigma * 1000.0).round())
            .collect();
        
        Ok(Self {
            sigmas,
            timesteps,
            config,
        })
    }
    
    fn noise_schedule(n: usize, num_train_timesteps: usize, shift: f64) -> Vec<f32> {
        let start = 1.0;
        let end = num_train_timesteps as f64;
        
        let mut timesteps = Vec::with_capacity(n + 1);
        
        // Create linearly spaced timesteps
        for i in 0..=n {
            let t = start + (end - start) * (i as f64) / (n as f64);
            timesteps.push(t);
        }
        
        // Apply shift
        let shifted_timesteps: Vec<f64> = timesteps
            .iter()
            .map(|&t| shift * t / (1.0 + (shift - 1.0) * t / end))
            .collect();
        
        // Convert to sigmas
        let sigmas: Vec<f32> = shifted_timesteps
            .iter()
            .map(|&t| {
                let t_norm = t / num_train_timesteps as f64;
                ((1.0 - t_norm) / t_norm).sqrt() as f32
            })
            .collect();
        
        sigmas
    }
    
    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }
    
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }
    
    pub fn init_noise(
        &self,
        batch_size: usize,
        num_channels: usize,
        height: usize,
        width: usize,
        device: &Device,
        dtype: DType,
        seed: Option<u64>,
    ) -> CandleResult<Tensor> {
        // For SD3, the latent space is actually based on the patch size
        // The model expects latents at the "patch latent" resolution
        // Based on the MMDiT output, it seems to expect a different resolution
        // Let me calculate this empirically: for 64x64 input, MMDiT outputs 12x12
        let scaling_factor = if height == 64 && width == 64 {
            // For 64x64 input, MMDiT outputs 28x28, so we need 28x28 latents
            64.0 / 28.0 // ≈ 2.29
        } else {
            // Default VAE scaling
            8.0
        };
        
        let latent_height = (height as f64 / scaling_factor).round() as usize;
        let latent_width = (width as f64 / scaling_factor).round() as usize;
        
        eprintln!("Scheduler init_noise: image {}x{} -> latent {}x{} (scale={})", 
                  height, width, latent_height, latent_width, scaling_factor);
        
        // Generate random noise
        let shape = &[batch_size, num_channels, latent_height, latent_width];
        
        if let Some(seed) = seed {
            // Seeded random
            use rand::SeedableRng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let noise: Vec<f32> = (0..shape.iter().product())
                .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
                .collect();
            Tensor::from_vec(noise, shape, device)?.to_dtype(dtype)
        } else {
            // Unseeded random
            Tensor::randn(0.0f32, 1.0f32, shape, device)?.to_dtype(dtype)
        }
    }
    
    pub fn scale_noise(&self, sample: &Tensor, timestep_index: usize) -> CandleResult<Tensor> {
        let sigma = self.sigmas[timestep_index];
        sample * sigma as f64
    }
    
    pub fn step(
        &self,
        model_output: &Tensor,
        timestep_index: usize,
        sample: &Tensor,
    ) -> CandleResult<Tensor> {
        let sigma = self.sigmas[timestep_index];
        let sigma_next = if timestep_index + 1 < self.sigmas.len() {
            self.sigmas[timestep_index + 1]
        } else {
            0.0
        };
        
        // Euler method
        let pred_original_sample = (sample - model_output * (sigma as f64))?;
        
        if sigma_next > 0.0 {
            let dt = sigma_next - sigma;
            let derivative = ((sample - &pred_original_sample)? / (sigma as f64))?;
            Ok((pred_original_sample + derivative * (dt as f64))?)
        } else {
            Ok(pred_original_sample)
        }
    }
}