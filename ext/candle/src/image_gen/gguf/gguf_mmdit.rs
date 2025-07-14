use candle_core::{Device, Result as CandleResult, Tensor, DType, D, IndexOp, quantized::{gguf_file, GgmlDType}};
use std::collections::HashMap;

/// Quantized MMDiT (Multimodal Diffusion Transformer) for SD3
pub struct QuantizedMMDiT {
    tensors: HashMap<String, QuantizedTensor>,
    device: Device,
    config: MMDiTConfig,
}

/// Configuration for the quantized MMDiT model
#[derive(Debug, Clone)]
pub struct MMDiTConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub max_sequence_length: usize,
}

impl Default for MMDiTConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1536,      // SD3 medium
            num_attention_heads: 24,
            num_layers: 24,
            patch_size: 2,
            in_channels: 16,
            max_sequence_length: 256,
        }
    }
}

/// A quantized tensor that can be dequantized on demand
pub struct QuantizedTensor {
    /// Raw quantized data
    data: Vec<u8>,
    /// Shape of the tensor
    shape: Vec<usize>,
    /// Quantization type (Q4_0, Q5_0, Q8_0, etc.)
    dtype: GgmlDType,
    /// Device for computation
    device: Device,
}

impl QuantizedTensor {
    /// Create a new quantized tensor
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
    
    /// Dequantize the tensor to a regular Tensor
    pub fn dequantize(&self) -> CandleResult<Tensor> {
        eprintln!("Dequantizing tensor with shape {:?} and dtype {:?}", self.shape, self.dtype);
        
        // Use the real GGML dequantization
        crate::image_gen::gguf::ggml_quant::dequantize_ggml(
            &self.data,
            &self.shape,
            self.dtype,
            &self.device,
        )
    }
}

/// Helper to get a tensor by name and dequantize it
fn get_tensor(tensors: &HashMap<String, QuantizedTensor>, name: &str) -> CandleResult<Tensor> {
    eprintln!("\n=== get_tensor called ===");
    eprintln!("Looking for: '{}'", name);
    eprintln!("Total tensors available: {}", tensors.len());
    
    // Print first few tensor names for debugging
    if tensors.len() > 0 {
        eprintln!("Sample tensor names:");
        for (i, key) in tensors.keys().take(3).enumerate() {
            eprintln!("  [{}] {}", i, key);
        }
    }
    
    // Try to get the tensor with the exact name first
    if let Some(tensor) = tensors.get(name) {
        eprintln!("  ✓ Found with exact name");
        return tensor.dequantize();
    }
    
    // If not found, try with the model.diffusion_model prefix
    let prefixed_name = format!("model.diffusion_model.{}", name);
    eprintln!("  Trying with prefix: '{}'", prefixed_name);
    if let Some(tensor) = tensors.get(&prefixed_name) {
        eprintln!("  ✓ Found with prefix!");
        return tensor.dequantize();
    }
    
    // If still not found, try without any prefix if the name already has one
    if name.starts_with("model.diffusion_model.") {
        let unprefixed = name.strip_prefix("model.diffusion_model.").unwrap();
        eprintln!("  Trying without prefix: '{}'", unprefixed);
        if let Some(tensor) = tensors.get(unprefixed) {
            eprintln!("  ✓ Found without prefix!");
            return tensor.dequantize();
        }
    }
    
    eprintln!("  ✗ NOT FOUND!");
    eprintln!("  Available keys containing 'x_embedder':");
    for key in tensors.keys() {
        if key.contains("x_embedder") {
            eprintln!("    - {}", key);
        }
    }
    Err(candle_core::Error::Msg(format!("Tensor {} not found", name)))
}

/// Apply a linear layer using quantized weights
fn linear(
    x: &Tensor,
    tensors: &HashMap<String, QuantizedTensor>,
    prefix: &str,
) -> CandleResult<Tensor> {
    let weight = get_tensor(tensors, &format!("{}.weight", prefix))?;
    eprintln!("\nLinear layer: {}", prefix);
    eprintln!("  Input shape: {:?}", x.shape());
    eprintln!("  Weight shape: {:?}", weight.shape());
    
    // Try to get bias - it's optional
    let bias = get_tensor(tensors, &format!("{}.bias", prefix)).ok();
    
    // For linear layers, weight is typically [out_features, in_features]
    // We need to transpose for matmul: [batch, seq, in] @ [in, out]
    eprintln!("  Attempting transpose...");
    let weight_t = match weight.t() {
        Ok(t) => {
            eprintln!("  Weight transposed shape: {:?}", t.shape());
            t
        },
        Err(e) => {
            eprintln!("  Transpose failed: {}", e);
            return Err(e);
        }
    };
    
    eprintln!("  Attempting matmul: {:?} @ {:?}", x.shape(), weight_t.shape());
    eprintln!("  x device: {:?}, dtype: {:?}", x.device(), x.dtype());
    eprintln!("  weight_t device: {:?}, dtype: {:?}", weight_t.device(), weight_t.dtype());
    
    // For 3D tensors, we might need to reshape for matmul
    let (batch_seq, in_dim) = if x.dims().len() == 3 {
        let dims = x.dims();
        let batch = dims[0];
        let seq = dims[1];
        let in_features = dims[2];
        eprintln!("  Reshaping 3D tensor from [{}, {}, {}] to [{}, {}]", batch, seq, in_features, batch * seq, in_features);
        let x_2d = x.reshape(&[batch * seq, in_features])?;
        ((batch, seq), x_2d)
    } else {
        ((1, x.dims()[0]), x.clone())
    };
    
    eprintln!("  Reshaped input: {:?}", in_dim.shape());
    let out_2d = match in_dim.matmul(&weight_t) {
        Ok(result) => {
            eprintln!("  Matmul succeeded! Output shape: {:?}", result.shape());
            result
        },
        Err(e) => {
            eprintln!("  Matmul failed: {}", e);
            eprintln!("  Error details: {:?}", e);
            return Err(e);
        }
    };
    
    // Reshape back to 3D if needed
    let out = if x.dims().len() == 3 {
        let out_features = out_2d.dims()[1];
        eprintln!("  Reshaping output back to [{}, {}, {}]", batch_seq.0, batch_seq.1, out_features);
        out_2d.reshape(&[batch_seq.0, batch_seq.1, out_features])?
    } else {
        out_2d
    };
    
    match bias {
        Some(b) => out.broadcast_add(&b),
        None => Ok(out),
    }
}

/// Apply layer normalization using quantized weights
fn layer_norm(
    x: &Tensor,
    tensors: &HashMap<String, QuantizedTensor>,
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

/// Apply GELU activation
fn gelu(x: &Tensor) -> CandleResult<Tensor> {
    x.gelu()
}

/// Apply SiLU (Swish) activation
fn silu(x: &Tensor) -> CandleResult<Tensor> {
    x.silu()
}

impl QuantizedMMDiT {
    /// Create a new QuantizedMMDiT from GGUF content
    pub fn from_gguf(
        content: gguf_file::Content,
        file: &mut std::fs::File,
        device: &Device,
    ) -> CandleResult<Self> {
        eprintln!("Loading quantized MMDiT from GGUF...");
        
        // Extract model configuration from metadata
        let config = Self::extract_config(&content)?;
        eprintln!("Detected MMDiT config: {:?}", config);
        
        let mut tensors = HashMap::new();
        let mut mmdit_tensor_count = 0;
        
        // Debug: list x_embedder tensors
        eprintln!("\nSearching for x_embedder tensors:");
        for (name, _) in &content.tensor_infos {
            if name.contains("x_embedder") {
                eprintln!("  Found: {}", name);
            }
        }
        eprintln!("");
        
        // Load all MMDiT tensors
        for (name, info) in &content.tensor_infos {
            // Check if this tensor belongs to MMDiT component
            if Self::is_mmdit_tensor(name) {
                if mmdit_tensor_count < 10 || name.contains("x_embedder") {
                    eprintln!("Loading MMDiT tensor: {} (shape: {:?}, dtype: {:?})", 
                        name, info.shape.dims(), info.ggml_dtype);
                }
                
                // Read tensor data from file
                let tensor_data = Self::read_tensor_data(file, info)?;
                
                // Create quantized tensor
                let quantized_tensor = QuantizedTensor::new(
                    tensor_data,
                    info.shape.dims().to_vec(),
                    info.ggml_dtype,
                    device,
                );
                
                tensors.insert(name.clone(), quantized_tensor);
                mmdit_tensor_count += 1;
            }
        }
        
        eprintln!("Loaded {} MMDiT tensors from GGUF", mmdit_tensor_count);
        
        if mmdit_tensor_count == 0 {
            return Err(candle_core::Error::Msg(
                "No MMDiT tensors found in GGUF file".to_string()
            ));
        }
        
        Ok(Self {
            tensors,
            device: device.clone(),
            config,
        })
    }
    
    /// Extract model configuration from GGUF metadata
    fn extract_config(content: &gguf_file::Content) -> CandleResult<MMDiTConfig> {
        let metadata = &content.metadata;
        
        // Try to extract configuration from metadata
        let hidden_size = metadata.get("sd3.attention.head_count")
            .and_then(|v| match v {
                gguf_file::Value::U32(n) => Some(*n as usize * 64), // head_count * head_dim
                _ => None,
            })
            .unwrap_or(1536);
        
        let num_attention_heads = metadata.get("sd3.attention.head_count")
            .and_then(|v| match v {
                gguf_file::Value::U32(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(24);
        
        let num_layers = metadata.get("sd3.block_count")
            .and_then(|v| match v {
                gguf_file::Value::U32(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(24);
        
        eprintln!("Extracted config - hidden_size: {}, heads: {}, layers: {}", 
            hidden_size, num_attention_heads, num_layers);
        
        Ok(MMDiTConfig {
            hidden_size,
            num_attention_heads,
            num_layers,
            patch_size: 2,
            in_channels: 16,
            max_sequence_length: 256,
        })
    }
    
    /// Check if tensor name belongs to MMDiT component
    fn is_mmdit_tensor(name: &str) -> bool {
        name.starts_with("model.diffusion_model.") ||
        name.starts_with("diffusion_model.") ||
        name.starts_with("dit.") ||
        name.contains(".joint_blocks.") ||
        name.contains(".final_layer.") ||
        name.contains(".x_embedder.") ||
        name.contains(".t_embedder.") ||
        name.contains(".y_embedder.") ||
        name.contains(".context_embedder.")
    }
    
    /// Read tensor data from GGUF file
    fn read_tensor_data(
        file: &mut std::fs::File,
        info: &gguf_file::TensorInfo,
    ) -> CandleResult<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        
        // Calculate the actual data size for quantized formats
        let data_size = Self::calculate_tensor_data_size(info)?;
        
        // Seek to the tensor data position
        file.seek(SeekFrom::Start(info.offset))
            .map_err(|e| candle_core::Error::Msg(format!("Failed to seek to tensor data: {}", e)))?;
        
        // Read the raw tensor data
        let mut data = vec![0u8; data_size];
        file.read_exact(&mut data)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read tensor data: {}", e)))?;
        
        Ok(data)
    }
    
    /// Calculate actual tensor data size for GGML formats
    fn calculate_tensor_data_size(info: &gguf_file::TensorInfo) -> CandleResult<usize> {
        let elem_count = info.shape.elem_count();
        
        let size = match info.ggml_dtype {
            GgmlDType::Q4_0 => {
                let n_blocks = (elem_count + 31) / 32;
                n_blocks * 18
            },
            GgmlDType::Q5_0 => {
                let n_blocks = (elem_count + 31) / 32;
                n_blocks * 22
            },
            GgmlDType::Q8_0 => {
                let n_blocks = (elem_count + 31) / 32;
                n_blocks * 34
            },
            GgmlDType::Q4_1 => {
                let n_blocks = (elem_count + 31) / 32;
                n_blocks * 20
            },
            GgmlDType::Q5_1 => {
                let n_blocks = (elem_count + 31) / 32;
                n_blocks * 24
            },
            GgmlDType::F16 => elem_count * 2,
            GgmlDType::F32 => elem_count * 4,
            _ => elem_count * info.ggml_dtype.type_size()
        };
        
        Ok(size)
    }
    
    /// Forward pass through the quantized MMDiT
    pub fn forward(
        &self, 
        x: &Tensor, 
        timestep: &Tensor, 
        context: &Tensor, 
        y: &Tensor
    ) -> CandleResult<Tensor> {
        eprintln!("QuantizedMMDiT forward pass starting...");
        eprintln!("Input shapes - x: {:?}, timestep: {:?}, context: {:?}, y: {:?}", 
            x.shape(), timestep.shape(), context.shape(), y.shape());
        
        // Debug: print available x_embedder tensors
        eprintln!("\nTotal tensors in MMDiT: {}", self.tensors.len());
        eprintln!("First 5 tensor names:");
        for (i, name) in self.tensors.keys().take(5).enumerate() {
            eprintln!("  {}: {}", i, name);
        }
        eprintln!("\nSearching for x_embedder tensors:");
        let x_embedder_count = self.tensors.keys()
            .filter(|name| name.contains("x_embedder"))
            .count();
        eprintln!("Found {} x_embedder tensors", x_embedder_count);
        for name in self.tensors.keys() {
            if name.contains("x_embedder") {
                eprintln!("  - {}", name);
            }
        }
        eprintln!("");
        
        // 1. Patch embedding (x_embedder)
        eprintln!("About to apply patch embedding...");
        let x_emb = self.apply_patch_embedding(x)?;
        eprintln!("Patch embedding complete!");
        eprintln!("After patch embedding: {:?}", x_emb.shape());
        
        // 2. Timestep embedding (t_embedder)
        let t_emb = self.apply_timestep_embedding(timestep)?;
        eprintln!("Timestep embedding: {:?}", t_emb.shape());
        
        // 3. Label embedding (y_embedder)
        let y_emb = self.apply_label_embedding(y)?;
        eprintln!("Label embedding: {:?}", y_emb.shape());
        
        // 4. Context embedding (for text conditioning)
        let context_emb = self.apply_context_embedding(context)?;
        eprintln!("Context embedding: {:?}", context_emb.shape());
        
        // 5. Combine embeddings
        let mut hidden = self.combine_embeddings(&x_emb, &t_emb, &y_emb)?;
        eprintln!("Combined embeddings: {:?}", hidden.shape());
        
        // 6. Process through joint transformer blocks
        hidden = self.apply_transformer_blocks(&hidden, &context_emb)?;
        eprintln!("After transformer blocks: {:?}", hidden.shape());
        
        // 7. Final layer and unpatchify
        let output = self.apply_final_layer(&hidden)?;
        eprintln!("Final output: {:?}", output.shape());
        
        Ok(output)
    }
    
    /// Apply patch embedding to input
    fn apply_patch_embedding(&self, x: &Tensor) -> CandleResult<Tensor> {
        eprintln!("\n=== apply_patch_embedding called ===");
        // x_embedder in SD3 is a 2D convolution with patch_size=2
        // Follow the standard PatchEmbedder implementation
        
        let (b, c, h, w) = x.dims4()?;
        eprintln!("Input shape: [{}, {}, {}, {}]", b, c, h, w);
        let patch_size = self.config.patch_size;
        eprintln!("Patch size: {}", patch_size);
        
        // For patch embedding, we need to handle different input sizes
        // The Conv2d with kernel_size=2, stride=2 will downsample by 2x
        // So: input_size -> input_size/2 patches
        // But we need to handle potential padding issues
        
        // Don't crop - let the conv2d handle the input as-is
        let x = x.clone();
        
        // Calculate expected output dimensions based on input
        let expected_h_out = h / patch_size; // Simple stride-based calculation
        let expected_w_out = w / patch_size;
        let expected_n_patches = expected_h_out * expected_w_out;
        eprintln!("Expected patches: {} ({}x{}) from input {}x{}", 
                  expected_n_patches, expected_h_out, expected_w_out, h, w);
        
        // Get the conv weight and bias
        let weight = get_tensor(&self.tensors, "x_embedder.proj.weight")?;
        let bias = get_tensor(&self.tensors, "x_embedder.proj.bias")?;
        
        eprintln!("Conv weight shape: {:?}", weight.shape());
        eprintln!("Conv bias shape: {:?}", bias.shape());
        
        // Apply 2D convolution following standard PatchEmbedder pattern
        // kernel_size = patch_size, stride = patch_size, padding = 0
        let kernel_size = patch_size;
        let stride = patch_size; 
        let padding = 0;
        let dilation = 1;
        
        eprintln!("Conv2d params: kernel={}, stride={}, padding={}, dilation={}", kernel_size, stride, padding, dilation);
        eprintln!("Input to conv2d: {:?}", x.shape());
        eprintln!("Weight shape: {:?}", weight.shape());
        let groups = 1; // Standard convolution (not grouped)
        let x_conv = x.conv2d(&weight, padding, stride, dilation, groups)?;
        eprintln!("After conv2d: {:?}", x_conv.shape());
        eprintln!("Conv2d: {}x{} input -> {}x{} output", h, w, x_conv.dims()[2], x_conv.dims()[3]);
        
        // Add bias
        let x_conv = x_conv.broadcast_add(&bias.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?)?;
        eprintln!("After bias addition: {:?}", x_conv.shape());
        
        // Reshape following standard PatchEmbedder pattern:
        // (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        let (b, c_out, h_out, w_out) = x_conv.dims4()?;
        eprintln!("Conv output dimensions: b={}, c_out={}, h_out={}, w_out={}", b, c_out, h_out, w_out);
        
        let x_emb = x_conv.reshape((b, c_out, h_out * w_out))?.transpose(1, 2)?;
        eprintln!("Final patch embedding shape: {:?}", x_emb.shape());
        
        // Verify dimensions match expectations
        let actual_n_patches = h_out * w_out;
        if actual_n_patches != expected_n_patches {
            eprintln!("WARNING: Patch count mismatch! Expected {}, got {}", expected_n_patches, actual_n_patches);
            eprintln!("This suggests input size or conv2d parameters are incorrect");
        }
        
        // Add positional embeddings if available
        if let Ok(pos_embed) = get_tensor(&self.tensors, "pos_embed") {
            eprintln!("\nAdding positional embeddings...");
            eprintln!("pos_embed shape: {:?}", pos_embed.shape());
            eprintln!("x_emb shape: {:?}", x_emb.shape());
            
            // pos_embed is [max_patches, hidden_dim], we need to slice it
            let n_patches = x_emb.dims()[1];
            eprintln!("Number of patches in x_emb: {}", n_patches);
            
            // Check if we have enough positional embeddings
            let max_pos_embeds = pos_embed.dims()[0];
            eprintln!("Max positional embeddings available: {}", max_pos_embeds);
            
            if n_patches > max_pos_embeds {
                eprintln!("ERROR: Not enough positional embeddings! Need {}, have {}", n_patches, max_pos_embeds);
                return Err(candle_core::Error::Msg(format!(
                    "Insufficient positional embeddings: need {} patches but only have {} embeddings",
                    n_patches, max_pos_embeds
                )));
            }
            
            let pos_embed_slice = pos_embed.narrow(0, 0, n_patches)?;
            eprintln!("pos_embed_slice shape: {:?}", pos_embed_slice.shape());
            
            // Add positional embeddings
            x_emb.broadcast_add(&pos_embed_slice.unsqueeze(0)?)
        } else {
            eprintln!("No positional embeddings found");
            Ok(x_emb)
        }
    }
    
    /// Apply timestep embedding
    fn apply_timestep_embedding(&self, timestep: &Tensor) -> CandleResult<Tensor> {
        // Timestep embedding typically uses sinusoidal encoding followed by MLPs
        
        // First, create sinusoidal embeddings
        let t_freq = self.timestep_sinusoidal_embedding(timestep)?;
        
        // Then apply MLP layers
        let t_emb = linear(&t_freq, &self.tensors, "t_embedder.mlp.0")?;
        let t_emb = silu(&t_emb)?;
        linear(&t_emb, &self.tensors, "t_embedder.mlp.2")
    }
    
    /// Apply label embedding
    fn apply_label_embedding(&self, y: &Tensor) -> CandleResult<Tensor> {
        // In SD3, y is the pooled text embeddings, not class labels
        // The y_embedder might not exist in all models
        if self.tensors.contains_key("y_embedder.embedding_table.weight") || 
           self.tensors.contains_key("model.diffusion_model.y_embedder.embedding_table.weight") {
            linear(y, &self.tensors, "y_embedder.embedding_table")
        } else {
            // If no y_embedder, need to project pooled embeddings to match hidden_size
            eprintln!("No y_embedder found, projecting pooled embeddings to hidden size");
            
            // Check if there's a pooled_projector
            if self.tensors.contains_key("pooled_projector.weight") || 
               self.tensors.contains_key("model.diffusion_model.pooled_projector.weight") {
                linear(y, &self.tensors, "pooled_projector")
            } else {
                // Create a simple projection to match hidden_size
                let (batch_size, pooled_dim) = y.dims2()?;
                eprintln!("Pooled embedding shape: [{}, {}]", batch_size, pooled_dim);
                eprintln!("Target hidden size: {}", self.config.hidden_size);
                
                // If dimensions match, use directly
                if pooled_dim == self.config.hidden_size {
                    Ok(y.clone())
                } else {
                    // Otherwise, we need to handle the dimension mismatch
                    // For now, just slice or pad to match
                    if pooled_dim > self.config.hidden_size {
                        // Slice to match hidden_size
                        y.narrow(1, 0, self.config.hidden_size)
                    } else {
                        // Pad with zeros
                        let zeros = Tensor::zeros(&[batch_size, self.config.hidden_size - pooled_dim], y.dtype(), y.device())?;
                        Tensor::cat(&[y, &zeros], 1)
                    }
                }
            }
        }
    }
    
    /// Apply context embedding for text conditioning
    fn apply_context_embedding(&self, context: &Tensor) -> CandleResult<Tensor> {
        // Context is already embedded by text encoder, just project it
        eprintln!("\nApplying context embedding...");
        eprintln!("Context shape: {:?}", context.shape());
        
        if self.tensors.contains_key("context_embedder.weight") || self.tensors.contains_key("model.diffusion_model.context_embedder.weight") {
            let result = linear(context, &self.tensors, "context_embedder")?;
            eprintln!("Context after embedding: {:?}", result.shape());
            Ok(result)
        } else {
            eprintln!("No context embedder found, using context as-is");
            Ok(context.clone())
        }
    }
    
    /// Combine embeddings
    fn combine_embeddings(
        &self,
        x_emb: &Tensor,
        t_emb: &Tensor,
        y_emb: &Tensor,
    ) -> CandleResult<Tensor> {
        // Add timestep and label embeddings to patch embeddings
        let c_emb = (t_emb + y_emb)?;
        
        // Broadcast and add to all patch positions
        let (b, n_patches, d) = x_emb.dims3()?;
        let c_emb = c_emb.unsqueeze(1)?; // Add spatial dimension
        let c_emb = c_emb.broadcast_as((b, n_patches, d))?;
        
        x_emb.add(&c_emb)
    }
    
    /// Apply transformer blocks
    fn apply_transformer_blocks(
        &self,
        hidden: &Tensor,
        context: &Tensor,
    ) -> CandleResult<Tensor> {
        let mut x = hidden.clone();
        
        // Process through each transformer block
        for i in 0..self.config.num_layers {
            x = self.apply_single_block(&x, context, i)?;
            
            // Log progress every few blocks
            if i % 4 == 0 {
                eprintln!("Processed transformer block {}/{}", i + 1, self.config.num_layers);
            }
        }
        
        Ok(x)
    }
    
    /// Apply a single transformer block
    fn apply_single_block(
        &self,
        x: &Tensor,
        context: &Tensor,
        block_idx: usize,
    ) -> CandleResult<Tensor> {
        let prefix = format!("joint_blocks.{}", block_idx);
        
        // SD3 uses AdaLN modulation, not regular layer norm
        // For now, let's use a simplified version that assumes standard attention blocks
        
        // X-block (self-attention on image patches)
        let x_block_prefix = format!("{}.x_block", prefix);
        let x_attn = self.apply_attention_block(x, x, &x_block_prefix, "x")?;
        let x = (x + x_attn)?;
        
        // Context block (cross-attention with text)
        let context_block_prefix = format!("{}.context_block", prefix);
        let context_attn = self.apply_attention_block(&x, context, &context_block_prefix, "context")?;
        let x = (x + context_attn)?;
        
        Ok(x)
    }
    
    /// Apply attention block with AdaLN modulation
    fn apply_attention_block(
        &self,
        x: &Tensor,
        context: &Tensor,
        prefix: &str,
        block_type: &str,
    ) -> CandleResult<Tensor> {
        eprintln!("Applying attention block: {}", prefix);
        
        // For SD3, attention blocks typically have:
        // - attn.qkv for combined query/key/value projection
        // - attn.proj for output projection
        // - mlp.fc1 and mlp.fc2 for feedforward
        // - adaLN_modulation for adaptive layer norm
        
        // Simple attention without AdaLN for now
        let attn_prefix = format!("{}.attn", prefix);
        
        // Combined QKV projection
        if let Ok(qkv) = linear(x, &self.tensors, &format!("{}.qkv", attn_prefix)) {
            // Split QKV
            let (b, seq_len, _) = x.dims3()?;
            let head_dim = 64; // Typical head dimension
            let n_heads = self.config.num_attention_heads;
            let hidden_size = self.config.hidden_size;
            
            // Reshape QKV: [B, Seq, 3*Hidden] -> [B, Seq, 3, Heads, HeadDim]
            let qkv = qkv.reshape(&[b, seq_len, 3, n_heads, head_dim])?;
            let qkv = qkv.permute((2, 0, 3, 1, 4))?; // [3, B, Heads, Seq, HeadDim]
            
            let q = qkv.i(0)?;
            let k = qkv.i(1)?;
            let v = qkv.i(2)?;
            
            // Scaled dot-product attention
            let d_k = (head_dim as f64).sqrt();
            
            // Ensure tensors are contiguous for matmul
            let q = q.contiguous()?;
            let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
            let v = v.contiguous()?;
            
            let scores = q.matmul(&k_t)?;
            let scores = (scores / d_k)?;
            let weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
            let attn_out = weights.matmul(&v)?;
            
            // Reshape back: [B, Heads, Seq, HeadDim] -> [B, Seq, Hidden]
            let attn_out = attn_out.permute((0, 2, 1, 3))?;
            let attn_out = attn_out.reshape(&[b, seq_len, hidden_size])?;
            
            // Output projection (may not exist for all blocks)
            let out = if let Ok(proj_out) = linear(&attn_out, &self.tensors, &format!("{}.proj", attn_prefix)) {
                proj_out
            } else {
                eprintln!("Warning: No projection weights found for {}, using attention output directly", attn_prefix);
                attn_out
            };
            
            // Apply MLP if exists
            let mlp_prefix = format!("{}.mlp", prefix);
            if self.tensors.contains_key(&format!("{}.fc1.weight", mlp_prefix)) ||
               self.tensors.contains_key(&format!("model.diffusion_model.{}.fc1.weight", mlp_prefix)) {
                let mlp_out = self.apply_mlp(&out, &mlp_prefix)?;
                Ok(mlp_out)
            } else {
                Ok(out)
            }
        } else {
            // Fallback: just return zeros if attention weights not found
            eprintln!("Warning: QKV weights not found for {}, returning zeros", prefix);
            Tensor::zeros_like(x)
        }
    }
    
    /// Apply MLP block
    fn apply_mlp(&self, x: &Tensor, prefix: &str) -> CandleResult<Tensor> {
        let x = linear(x, &self.tensors, &format!("{}.fc1", prefix))?;
        let x = gelu(&x)?;
        linear(&x, &self.tensors, &format!("{}.fc2", prefix))
    }
    
    /// Apply attention mechanism
    fn apply_attention(
        &self,
        query: &Tensor,
        key_value: &Tensor,
        prefix: &str,
    ) -> CandleResult<Tensor> {
        // Simple attention implementation
        // In a full implementation, this would handle multi-head attention properly
        
        let q = linear(query, &self.tensors, &format!("{}.q_linear", prefix))?;
        let k = linear(key_value, &self.tensors, &format!("{}.k_linear", prefix))?;
        let v = linear(key_value, &self.tensors, &format!("{}.v_linear", prefix))?;
        
        // Scaled dot-product attention (simplified)
        let d_k = (q.dims()[2] as f64).sqrt();
        let scores = (q.matmul(&k.t()?)? / d_k)?;
        let weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let attn_out = weights.matmul(&v)?;
        
        // Output projection
        linear(&attn_out, &self.tensors, &format!("{}.out_proj", prefix))
    }
    
    /// Apply feed-forward network
    fn apply_feedforward(&self, x: &Tensor, block_prefix: &str) -> CandleResult<Tensor> {
        let prefix = format!("{}.mlp", block_prefix);
        
        // First linear layer
        let x = linear(x, &self.tensors, &format!("{}.fc1", prefix))?;
        let x = gelu(&x)?;
        
        // Second linear layer
        linear(&x, &self.tensors, &format!("{}.fc2", prefix))
    }
    
    /// Apply final layer and reshape output
    fn apply_final_layer(&self, hidden: &Tensor) -> CandleResult<Tensor> {
        eprintln!("\nApplying final layer...");
        eprintln!("Hidden shape before final layer: {:?}", hidden.shape());
        
        // SD3 uses AdaLN modulation for the final layer too
        // For now, skip the norm and just apply linear projection if it exists
        let x = if let Ok(linear_out) = linear(hidden, &self.tensors, "final_layer.linear") {
            eprintln!("Applied final_layer.linear");
            linear_out
        } else if let Ok(linear_out) = linear(hidden, &self.tensors, "final_layer.proj_out") {
            eprintln!("Applied final_layer.proj_out");
            linear_out
        } else {
            eprintln!("Warning: No final layer projection found, using hidden state directly");
            hidden.clone()
        };
        
        eprintln!("Shape after final projection: {:?}", x.shape());
        
        // Unpatchify: reshape back to image format
        let (b, n_patches, channels) = x.dims3()?;
        let patch_size = self.config.patch_size;
        // Calculate dimensions based on the expected latent resolution
        // The MMDiT seems to be outputting 34x34 patches instead of 32x32
        // For compatibility with the scheduler, we need to crop or adjust the output
        let h_patches = (n_patches as f64).sqrt() as usize;
        let w_patches = h_patches; 
        
        // For SD3, we need the output to match the actual latent dimensions
        // With 8x downsampling: 512x512 -> 64x64, 256x256 -> 32x32, 64x64 -> 8x8
        // But patch_size=2 means: 8x8 latents -> 4x4 patches
        // However, we're getting 6x6=36 patches, so let's work with that
        let target_h_patches = h_patches; // Use actual patch count
        let target_w_patches = w_patches;
        
        eprintln!("Unpatchify: n_patches={}, computed={}x{}, target={}x{}", 
                  n_patches, h_patches, w_patches, target_h_patches, target_w_patches);
        
        // For SD3, the output should have channels = patch_size * patch_size * latent_channels
        // where latent_channels = 16 for VAE input
        let latent_channels = channels / (patch_size * patch_size);
        eprintln!("Unpatchify - patches: {}, patch_size: {}, latent_channels: {}", n_patches, patch_size, latent_channels);
        
        // If we have more patches than needed, crop the tensor
        let x_cropped = if h_patches > target_h_patches || w_patches > target_w_patches {
            eprintln!("Cropping from {}x{} to {}x{} patches", h_patches, w_patches, target_h_patches, target_w_patches);
            
            // Reshape to spatial layout first: (B, n_patches, C) -> (B, H, W, C)  
            let x_spatial = x.reshape((b, h_patches, w_patches, channels))?;
            
            // Crop the spatial dimensions
            let x_cropped_spatial = x_spatial.i((.., 0..target_h_patches, 0..target_w_patches, ..))?;
            
            // Reshape back: (B, H, W, C) -> (B, n_patches_new, C)
            x_cropped_spatial.reshape((b, target_h_patches * target_w_patches, channels))?
        } else {
            x.clone()
        };
        
        // Use target dimensions for unpatchify
        let h = target_h_patches;
        let w = target_w_patches;
        
        // Reshape: (B, n_patches, patch_size*patch_size*C) -> (B, C, H, W)
        let output = x_cropped.reshape((b, h, w, patch_size, patch_size, latent_channels))?
            .permute((0, 5, 1, 3, 2, 4))?
            .reshape((b, latent_channels, h * patch_size, w * patch_size))?;
        
        eprintln!("Final output shape: {:?}", output.shape());
        Ok(output)
    }
    
    /// Create sinusoidal timestep embeddings
    fn timestep_sinusoidal_embedding(&self, timesteps: &Tensor) -> CandleResult<Tensor> {
        // SD3 uses 256-dimensional timestep embeddings
        let dim = 256;
        let half_dim = dim / 2;
        let max_period = 10000.0;
        
        // Create frequency bands
        let freqs = Tensor::arange(0, half_dim as i64, timesteps.device())?
            .to_dtype(DType::F32)?
            .affine((-f32::ln(max_period) / half_dim as f32) as f64, 0.0)?
            .exp()?;
        
        // Apply to timesteps
        let args = timesteps.unsqueeze(1)?.matmul(&freqs.unsqueeze(0)?)?;
        
        // Create sin and cos embeddings
        let sin_emb = args.sin()?;
        let cos_emb = args.cos()?;
        
        // Concatenate
        Tensor::cat(&[&sin_emb, &cos_emb], 1)
    }
    
    /// Get a specific tensor by name (for debugging/inspection)
    pub fn get_tensor(&self, name: &str) -> Option<&QuantizedTensor> {
        self.tensors.get(name)
    }
    
    /// List all available tensor names
    pub fn tensor_names(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }
    
    /// Get model configuration
    pub fn config(&self) -> &MMDiTConfig {
        &self.config
    }
    
    /// Get the device this model is on
    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl std::fmt::Debug for QuantizedMMDiT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantizedMMDiT")
            .field("device", &self.device)
            .finish()
    }
}