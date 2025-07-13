use candle_core::{DType, Device, Result as CandleResult, Tensor, IndexOp};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::mmdit;
use candle_transformers::models::clip;
use candle_transformers::models::t5;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct SD3Config {
    pub width: usize,
    pub height: usize,
    pub num_inference_steps: usize,
    pub cfg_scale: f64,
    pub time_shift: f64,
    pub use_t5: bool,
    pub clip_skip: usize,
}

impl Default for SD3Config {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            num_inference_steps: 28,
            cfg_scale: 7.0,
            time_shift: 3.0,
            use_t5: true,
            clip_skip: 0,
        }
    }
}

pub struct MMDiT {
    model: mmdit::model::MMDiT,
    device: Device,
}

impl MMDiT {
    pub fn new(vb: VarBuilder, config: &mmdit::model::Config, device: &Device) -> CandleResult<Self> {
        let model = mmdit::model::MMDiT::new(config, false, vb)?; // use_flash_attn = false
        Ok(Self {
            model,
            device: device.clone(),
        })
    }
    
    pub fn forward(&self, x: &Tensor, timestep: &Tensor, context: &Tensor, y: &Tensor) -> CandleResult<Tensor> {
        self.model.forward(x, timestep, context, y, None) // No image_rotary_emb
    }
}

pub struct CLIPTextEncoder {
    model: clip::text_model::ClipTextTransformer,
    tokenizer: Tokenizer,
    max_length: usize,
    device: Device,
}

impl CLIPTextEncoder {
    pub fn new(vb: VarBuilder, config: &clip::text_model::ClipTextConfig, tokenizer: Tokenizer, device: &Device) -> CandleResult<Self> {
        let model = clip::text_model::ClipTextTransformer::new(vb, config)?;
        Ok(Self {
            model,
            tokenizer,
            max_length: 77, // Standard CLIP max length
            device: device.clone(),
        })
    }
    
    pub fn encode(&self, text: &str) -> CandleResult<Tensor> {
        // Tokenize
        let encoding = self.tokenizer
            .encode(text, true)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenization error: {}", e)))?;
        
        let tokens = encoding.get_ids();
        let mut padded_tokens = tokens.to_vec();
        
        // Pad or truncate to max_length
        padded_tokens.resize(self.max_length, 0);
        
        // Convert to tensor
        let input_ids = Tensor::new(padded_tokens.as_slice(), &self.device)?
            .unsqueeze(0)?;
        
        // Get embeddings
        self.model.forward(&input_ids)
    }
}

pub struct T5TextEncoder {
    model: t5::T5EncoderModel,
    tokenizer: Tokenizer,
    max_length: usize,
    device: Device,
}

impl T5TextEncoder {
    pub fn new(vb: VarBuilder, config: &t5::Config, tokenizer: Tokenizer, device: &Device) -> CandleResult<Self> {
        let model = t5::T5EncoderModel::load(vb, config)?;
        Ok(Self {
            model,
            tokenizer,
            max_length: 512, // T5 can handle longer sequences
            device: device.clone(),
        })
    }
    
    pub fn encode(&mut self, text: &str) -> CandleResult<Tensor> {
        // Tokenize
        let encoding = self.tokenizer
            .encode(text, true)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenization error: {}", e)))?;
        
        let tokens = encoding.get_ids();
        let mut padded_tokens = tokens.to_vec();
        
        // Pad or truncate to max_length  
        padded_tokens.resize(self.max_length.min(tokens.len()), 0);
        
        // Convert to tensor
        let input_ids = Tensor::new(padded_tokens.as_slice(), &self.device)?
            .unsqueeze(0)?;
        
        // Get embeddings
        self.model.forward(&input_ids)
    }
}

pub struct TextEncoders {
    pub clip_g: Option<CLIPTextEncoder>,
    pub clip_l: Option<CLIPTextEncoder>,
    pub t5: Option<T5TextEncoder>,
}

impl TextEncoders {
    pub fn encode(&mut self, prompt: &str, negative_prompt: Option<&str>, use_t5: bool) -> CandleResult<(Tensor, Tensor, Tensor)> {
        let negative = negative_prompt.unwrap_or("");
        
        // Encode with CLIP-G
        let (clip_g_pos, clip_g_neg) = if let Some(encoder) = &self.clip_g {
            (encoder.encode(prompt)?, encoder.encode(negative)?)
        } else {
            return Err(candle_core::Error::Msg("CLIP-G encoder not loaded".to_string()));
        };
        
        // Encode with CLIP-L
        let (clip_l_pos, clip_l_neg) = if let Some(encoder) = &self.clip_l {
            (encoder.encode(prompt)?, encoder.encode(negative)?)
        } else {
            return Err(candle_core::Error::Msg("CLIP-L encoder not loaded".to_string()));
        };
        
        // Concatenate CLIP embeddings
        let clip_pos = Tensor::cat(&[&clip_g_pos, &clip_l_pos], 2)?;
        let clip_neg = Tensor::cat(&[&clip_g_neg, &clip_l_neg], 2)?;
        
        // Encode with T5 if requested
        let (t5_pos, t5_neg) = if use_t5 {
            if let Some(encoder) = &mut self.t5 {
                (encoder.encode(prompt)?, encoder.encode(negative)?)
            } else {
                // Return empty tensors if T5 not available
                let device = clip_pos.device();
                let zeros = Tensor::zeros(&[1, 1, 4096], DType::F32, device)?;
                (zeros.clone(), zeros)
            }
        } else {
            let device = clip_pos.device();
            let zeros = Tensor::zeros(&[1, 1, 4096], DType::F32, device)?;
            (zeros.clone(), zeros)
        };
        
        // Combine all embeddings
        let context_pos = Tensor::cat(&[&clip_pos, &t5_pos], 1)?;
        let context_neg = Tensor::cat(&[&clip_neg, &t5_neg], 1)?;
        
        // Also need pooled output for conditioning
        let pooled = clip_g_pos.i((.., 0, ..))?; // Use first token as pooled output
        
        Ok((context_pos, context_neg, pooled))
    }
}