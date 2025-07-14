use candle_core::{Device, Result as CandleResult, Tensor, D, quantized::{gguf_file, GgmlDType}};
use std::collections::HashMap;

/// Quantized text encoder for GGUF SD3
pub struct QuantizedTextEncoder {
    tensors: HashMap<String, QuantizedTextTensor>,
    device: Device,
    config: TextEncoderConfig,
    encoder_type: TextEncoderType,
}

/// Type of text encoder
#[derive(Debug, Clone)]
pub enum TextEncoderType {
    ClipL,
    ClipG,
    T5XXL,
}

/// Configuration for text encoders
#[derive(Debug, Clone)]
pub struct TextEncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
}

impl TextEncoderConfig {
    /// Default config for CLIP-L
    pub fn clip_l() -> Self {
        Self {
            vocab_size: 49408,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 77,
        }
    }
    
    /// Default config for CLIP-G
    pub fn clip_g() -> Self {
        Self {
            vocab_size: 49408,
            hidden_size: 1280,
            num_layers: 32,
            num_heads: 20,
            intermediate_size: 5120,
            max_position_embeddings: 77,
        }
    }
    
    /// Default config for T5-XXL
    pub fn t5_xxl() -> Self {
        Self {
            vocab_size: 32128,
            hidden_size: 4096,
            num_layers: 24,
            num_heads: 64,
            intermediate_size: 10240,
            max_position_embeddings: 512,
        }
    }
}

/// A quantized text encoder tensor
pub struct QuantizedTextTensor {
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: GgmlDType,
    device: Device,
}

impl QuantizedTextTensor {
    pub fn new(
        data: Vec<u8>,
        shape: Vec<usize>,
        dtype: GgmlDType,
        device: &Device,
    ) -> Self {
        Self {
            data,
            shape,
            dtype,
            device: device.clone(),
        }
    }
    
    pub fn dequantize(&self) -> CandleResult<Tensor> {
        crate::image_gen::gguf::ggml_quant::dequantize_ggml(
            &self.data,
            &self.shape,
            self.dtype,
            &self.device,
        )
    }
}

/// Helper to get a tensor by name and dequantize it
fn get_tensor(tensors: &HashMap<String, QuantizedTextTensor>, name: &str) -> CandleResult<Tensor> {
    tensors.get(name)
        .ok_or_else(|| candle_core::Error::Msg(format!("Tensor {} not found", name)))?
        .dequantize()
}

/// Apply layer normalization
fn layer_norm(
    x: &Tensor,
    tensors: &HashMap<String, QuantizedTextTensor>,
    prefix: &str,
    eps: f64,
) -> CandleResult<Tensor> {
    let weight = get_tensor(tensors, &format!("{}.weight", prefix))?;
    let bias = get_tensor(tensors, &format!("{}.bias", prefix))?;
    
    // Compute mean and variance
    let mean = x.mean_keepdim(D::Minus1)?;
    let x_centered = x.broadcast_sub(&mean)?;
    let var = (&x_centered * &x_centered)?.mean_keepdim(D::Minus1)?;
    let std = (var + eps)?.sqrt()?;
    
    // Normalize and apply affine transformation
    let normalized = x_centered.broadcast_div(&std)?;
    let scaled = normalized.broadcast_mul(&weight)?;
    scaled.broadcast_add(&bias)
}

/// Apply a linear layer
fn linear(
    x: &Tensor,
    tensors: &HashMap<String, QuantizedTextTensor>,
    prefix: &str,
) -> CandleResult<Tensor> {
    let weight = get_tensor(tensors, &format!("{}.weight", prefix))?;
    let bias = tensors.get(&format!("{}.bias", prefix))
        .map(|t| t.dequantize())
        .transpose()?;
    
    let out = x.matmul(&weight.t()?)?;
    match bias {
        Some(b) => out.broadcast_add(&b),
        None => Ok(out),
    }
}

impl QuantizedTextEncoder {
    /// Create a new QuantizedTextEncoder from GGUF content
    pub fn from_gguf(
        content: gguf_file::Content,
        file: &mut std::fs::File,
        device: &Device,
        encoder_type: TextEncoderType,
    ) -> CandleResult<Self> {
        eprintln!("Loading quantized {:?} text encoder from GGUF...", encoder_type);
        
        // Extract configuration based on encoder type
        let config = Self::extract_config(&content, &encoder_type)?;
        eprintln!("Detected text encoder config: {:?}", config);
        
        let mut tensors = HashMap::new();
        let mut encoder_tensor_count = 0;
        
        // Load all text encoder tensors
        for (name, info) in &content.tensor_infos {
            if Self::is_text_encoder_tensor(name, &encoder_type) {
                eprintln!("Loading text encoder tensor: {} (shape: {:?}, dtype: {:?})", 
                    name, info.shape.dims(), info.ggml_dtype);
                
                // Read tensor data from file
                let tensor_data = Self::read_tensor_data(file, info)?;
                
                // Create quantized tensor
                let quantized_tensor = QuantizedTextTensor::new(
                    tensor_data,
                    info.shape.dims().to_vec(),
                    info.ggml_dtype,
                    device,
                );
                
                tensors.insert(name.clone(), quantized_tensor);
                encoder_tensor_count += 1;
            }
        }
        
        eprintln!("Loaded {} {:?} text encoder tensors from GGUF", encoder_tensor_count, encoder_type);
        
        if encoder_tensor_count == 0 {
            return Err(candle_core::Error::Msg(
                format!("No {:?} text encoder tensors found in GGUF file", encoder_type)
            ));
        }
        
        Ok(Self {
            tensors,
            device: device.clone(),
            config,
            encoder_type,
        })
    }
    
    /// Extract text encoder configuration from GGUF metadata
    fn extract_config(content: &gguf_file::Content, encoder_type: &TextEncoderType) -> CandleResult<TextEncoderConfig> {
        let metadata = &content.metadata;
        
        // Try to extract from metadata, otherwise use defaults
        let prefix = match encoder_type {
            TextEncoderType::ClipL => "clip_l",
            TextEncoderType::ClipG => "clip_g", 
            TextEncoderType::T5XXL => "t5_xxl",
        };
        
        let hidden_size = metadata.get(&format!("{}.hidden_size", prefix))
            .and_then(|v| match v {
                gguf_file::Value::U32(n) => Some(*n as usize),
                _ => None,
            });
        
        let num_layers = metadata.get(&format!("{}.num_layers", prefix))
            .and_then(|v| match v {
                gguf_file::Value::U32(n) => Some(*n as usize),
                _ => None,
            });
        
        // Use defaults if not found in metadata
        let config = match encoder_type {
            TextEncoderType::ClipL => TextEncoderConfig::clip_l(),
            TextEncoderType::ClipG => TextEncoderConfig::clip_g(),
            TextEncoderType::T5XXL => TextEncoderConfig::t5_xxl(),
        };
        
        Ok(TextEncoderConfig {
            hidden_size: hidden_size.unwrap_or(config.hidden_size),
            num_layers: num_layers.unwrap_or(config.num_layers),
            ..config
        })
    }
    
    /// Check if tensor belongs to this text encoder type
    fn is_text_encoder_tensor(name: &str, encoder_type: &TextEncoderType) -> bool {
        match encoder_type {
            TextEncoderType::ClipL => {
                name.starts_with("cond_stage_model.") ||
                name.starts_with("text_model.") ||
                name.starts_with("clip_l.") ||
                (name.contains("text_encoder") && !name.contains("text_encoder_2"))
            },
            TextEncoderType::ClipG => {
                name.starts_with("text_encoder_2.") ||
                name.starts_with("clip_g.") ||
                name.contains("text_model_g")
            },
            TextEncoderType::T5XXL => {
                name.starts_with("text_encoder_3.") ||
                name.starts_with("t5xxl.") ||
                name.contains("t5_xxl")
            },
        }
    }
    
    /// Read tensor data from GGUF file
    fn read_tensor_data(
        file: &mut std::fs::File,
        info: &gguf_file::TensorInfo,
    ) -> CandleResult<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        
        let elem_count = info.shape.elem_count();
        let type_size = info.ggml_dtype.type_size();
        let data_size = elem_count * type_size;
        
        file.seek(SeekFrom::Start(info.offset))
            .map_err(|e| candle_core::Error::Msg(format!("Failed to seek: {}", e)))?;
        
        let mut data = vec![0u8; data_size];
        file.read_exact(&mut data)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read: {}", e)))?;
        
        Ok(data)
    }
    
    /// Encode text tokens to embeddings
    pub fn encode(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        eprintln!("Encoding with quantized {:?} text encoder", self.encoder_type);
        eprintln!("Input shape: {:?}", input_ids.shape());
        
        // Get token embeddings
        let embeddings = self.get_embeddings(input_ids)?;
        
        // Add positional embeddings if available
        let embeddings = self.add_positional_embeddings(embeddings)?;
        
        // Process through transformer layers
        let mut hidden = embeddings;
        for i in 0..self.config.num_layers {
            hidden = self.apply_encoder_layer(&hidden, i)?;
            
            if i % 4 == 0 {
                eprintln!("Processed encoder layer {}/{}", i + 1, self.config.num_layers);
            }
        }
        
        // Final layer norm
        hidden = self.apply_final_layer_norm(&hidden)?;
        
        eprintln!("Encoded output shape: {:?}", hidden.shape());
        Ok(hidden)
    }
    
    /// Get token embeddings
    fn get_embeddings(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        // Different encoders have different embedding table names
        let embed_name = match self.encoder_type {
            TextEncoderType::ClipL | TextEncoderType::ClipG => "text_model.embeddings.token_embedding.weight",
            TextEncoderType::T5XXL => "shared.weight",
        };
        
        let embed_weight = get_tensor(&self.tensors, embed_name)?;
        
        // Gather embeddings for input tokens
        let embeddings = embed_weight.index_select(input_ids, 0)?;
        Ok(embeddings)
    }
    
    /// Add positional embeddings
    fn add_positional_embeddings(&self, embeddings: Tensor) -> CandleResult<Tensor> {
        // CLIP uses learned position embeddings, T5 uses relative position bias
        match self.encoder_type {
            TextEncoderType::ClipL | TextEncoderType::ClipG => {
                let pos_embed = get_tensor(&self.tensors, "text_model.embeddings.position_embedding.weight")?;
                embeddings.broadcast_add(&pos_embed)
            },
            TextEncoderType::T5XXL => {
                // T5 uses relative position bias in attention, not absolute embeddings
                Ok(embeddings)
            },
        }
    }
    
    /// Apply a single encoder layer
    fn apply_encoder_layer(&self, x: &Tensor, layer_idx: usize) -> CandleResult<Tensor> {
        let layer_prefix = match self.encoder_type {
            TextEncoderType::ClipL | TextEncoderType::ClipG => 
                format!("text_model.encoder.layers.{}", layer_idx),
            TextEncoderType::T5XXL => 
                format!("encoder.block.{}.layer", layer_idx),
        };
        
        // Self-attention
        let attn_output = self.apply_self_attention(x, &layer_prefix)?;
        let x = (x + attn_output)?;
        
        // Feed-forward
        let ff_output = self.apply_feed_forward(&x, &layer_prefix)?;
        let x = (x + ff_output)?;
        
        Ok(x)
    }
    
    /// Apply self-attention
    fn apply_self_attention(&self, x: &Tensor, layer_prefix: &str) -> CandleResult<Tensor> {
        // Layer norm
        let x_norm = layer_norm(x, &self.tensors, &format!("{}.0.layer_norm", layer_prefix), 1e-5)?;
        
        // Query, key, value projections
        let q = linear(&x_norm, &self.tensors, &format!("{}.0.SelfAttention.q", layer_prefix))?;
        let k = linear(&x_norm, &self.tensors, &format!("{}.0.SelfAttention.k", layer_prefix))?;
        let v = linear(&x_norm, &self.tensors, &format!("{}.0.SelfAttention.v", layer_prefix))?;
        
        // Attention computation (simplified)
        let d_k = (self.config.hidden_size / self.config.num_heads) as f64;
        let scores = (q.matmul(&k.t()?)? / d_k.sqrt())?;
        let weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let attn_out = weights.matmul(&v)?;
        
        // Output projection
        linear(&attn_out, &self.tensors, &format!("{}.0.SelfAttention.o", layer_prefix))
    }
    
    /// Apply feed-forward network
    fn apply_feed_forward(&self, x: &Tensor, layer_prefix: &str) -> CandleResult<Tensor> {
        // Layer norm
        let x_norm = layer_norm(x, &self.tensors, &format!("{}.1.layer_norm", layer_prefix), 1e-5)?;
        
        // FFN layers
        let hidden = linear(&x_norm, &self.tensors, &format!("{}.1.DenseReluDense.wi", layer_prefix))?;
        let hidden = hidden.relu()?;
        linear(&hidden, &self.tensors, &format!("{}.1.DenseReluDense.wo", layer_prefix))
    }
    
    /// Apply final layer normalization
    fn apply_final_layer_norm(&self, x: &Tensor) -> CandleResult<Tensor> {
        let norm_name = match self.encoder_type {
            TextEncoderType::ClipL | TextEncoderType::ClipG => "text_model.final_layer_norm",
            TextEncoderType::T5XXL => "encoder.final_layer_norm",
        };
        
        layer_norm(x, &self.tensors, norm_name, 1e-5)
    }
    
    /// Get the encoder type
    pub fn encoder_type(&self) -> &TextEncoderType {
        &self.encoder_type
    }
    
    /// Get configuration
    pub fn config(&self) -> &TextEncoderConfig {
        &self.config
    }
    
    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl std::fmt::Debug for QuantizedTextEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantizedTextEncoder")
            .field("encoder_type", &self.encoder_type)
            .field("num_tensors", &self.tensors.len())
            .field("config", &self.config)
            .field("device", &self.device)
            .finish()
    }
}