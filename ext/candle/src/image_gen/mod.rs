use candle_core::{Device, Result as CandleResult};
use serde::{Deserialize, Serialize};

pub mod stable_diffusion;
pub mod sd3;

pub use stable_diffusion::StableDiffusion3;

/// Configuration for image generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationConfig {
    /// Height of the generated image
    pub height: usize,
    /// Width of the generated image
    pub width: usize,
    /// Number of denoising steps
    pub num_inference_steps: usize,
    /// Guidance scale for classifier-free guidance
    pub guidance_scale: f64,
    /// Negative prompt to avoid certain features
    pub negative_prompt: Option<String>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Interval between preview images (for streaming)
    pub preview_interval: Option<usize>,
    /// Scheduler type
    pub scheduler: SchedulerType,
    /// Whether to use T5 text encoder
    pub use_t5: bool,
    /// Number of CLIP layers to skip
    pub clip_skip: usize,
}

impl Default for ImageGenerationConfig {
    fn default() -> Self {
        Self {
            height: 1024,
            width: 1024,
            num_inference_steps: 28,
            guidance_scale: 7.0,
            negative_prompt: None,
            seed: None,
            preview_interval: None,
            scheduler: SchedulerType::Euler,
            use_t5: true,
            clip_skip: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerType {
    Euler,
    EulerA,
    DPMSolver,
    DDIM,
}

/// Progress information for streaming generation
#[derive(Debug, Clone)]
pub struct GenerationProgress {
    pub step: usize,
    pub total_steps: usize,
    pub image_data: Option<Vec<u8>>, // PNG bytes
}

/// Trait for image generation models
pub trait ImageGenerator: Send + Sync {
    /// Generate an image from a text prompt
    fn generate(&mut self, prompt: &str, config: &ImageGenerationConfig) -> CandleResult<Vec<u8>>;
    
    /// Generate an image with streaming progress updates
    fn generate_stream(
        &mut self,
        prompt: &str,
        config: &ImageGenerationConfig,
        callback: impl FnMut(GenerationProgress),
    ) -> CandleResult<Vec<u8>>;
    
    /// Get the model name
    fn model_name(&self) -> &str;
    
    /// Get the device the model is running on
    fn device(&self) -> &Device;
    
    /// Clear any cached data
    fn clear_cache(&mut self);
}

/// Convert a tensor to PNG bytes
pub fn tensor_to_png(tensor: &candle_core::Tensor) -> CandleResult<Vec<u8>> {
    use image::{ImageBuffer, Rgb};
    
    // Ensure tensor is on CPU and has the right shape [height, width, 3]
    let tensor = tensor.to_device(&Device::Cpu)?;
    let dims = tensor.dims();
    
    if dims.len() != 3 || dims[2] != 3 {
        return Err(candle_core::Error::Msg(format!(
            "Expected tensor shape [height, width, 3], got {:?}",
            dims
        )));
    }
    
    let height = dims[0];
    let width = dims[1];
    
    // Convert tensor to f32 and get values
    let tensor = tensor.to_dtype(candle_core::DType::F32)?;
    let data = tensor.flatten_all()?.to_vec1::<f32>()?;
    
    // Create image buffer
    let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width as u32, height as u32);
    
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let r = (data[idx] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (data[idx + 1] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (data[idx + 2] * 255.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    
    // Encode to PNG
    let mut png_data = Vec::new();
    {
        use std::io::Cursor;
        let mut cursor = Cursor::new(&mut png_data);
        img.write_to(&mut cursor, image::ImageFormat::Png)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to encode PNG: {}", e)))?;
    }
    
    Ok(png_data)
}