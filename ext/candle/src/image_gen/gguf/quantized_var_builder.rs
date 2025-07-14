use candle_core::{Device, Result as CandleResult, Tensor, Shape, quantized::{gguf_file, GgmlDType}};
use std::collections::HashMap;
use std::sync::Arc;

/// A quantized tensor that can be dequantized on demand
#[derive(Clone)]
pub struct QuantizedTensor {
    /// Raw quantized data
    data: Arc<Vec<u8>>,
    /// Shape of the tensor
    shape: Shape,
    /// Quantization type
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
            data: Arc::new(data),
            shape: Shape::from_dims(&shape),
            dtype,
            device: device.clone(),
        }
    }
    
    /// Dequantize the tensor to a regular Tensor
    pub fn dequantize(&self) -> CandleResult<Tensor> {
        crate::image_gen::gguf::ggml_quant::dequantize_ggml(
            &self.data,
            self.shape.dims(),
            self.dtype,
            &self.device,
        )
    }
    
    /// Get the shape of the tensor
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    
    /// Get the quantization type
    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }
    
    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// A builder for accessing quantized tensors with a path-based API
#[derive(Clone)]
pub struct QuantizedVarBuilder {
    tensors: Arc<HashMap<String, QuantizedTensor>>,
    path: Vec<String>,
    device: Device,
}

impl QuantizedVarBuilder {
    /// Create a new QuantizedVarBuilder from GGUF content
    pub fn from_gguf(
        content: gguf_file::Content,
        file: &mut std::fs::File,
        device: &Device,
    ) -> CandleResult<Self> {
        use std::io::{Read, Seek, SeekFrom};
        
        let mut tensors = HashMap::new();
        
        // Load all tensors
        for (name, info) in &content.tensor_infos {
            // Calculate tensor data size
            let elem_count = info.shape.elem_count();
            let type_size = info.ggml_dtype.type_size();
            let data_size = elem_count * type_size;
            
            // Seek to tensor data
            file.seek(SeekFrom::Start(info.offset))
                .map_err(|e| candle_core::Error::Msg(format!("Failed to seek: {}", e)))?;
            
            // Read tensor data
            let mut data = vec![0u8; data_size];
            file.read_exact(&mut data)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to read: {}", e)))?;
            
            // Create quantized tensor
            let tensor = QuantizedTensor::new(
                data,
                info.shape.dims().to_vec(),
                info.ggml_dtype,
                device,
            );
            
            tensors.insert(name.clone(), tensor);
        }
        
        Ok(Self {
            tensors: Arc::new(tensors),
            path: Vec::new(),
            device: device.clone(),
        })
    }
    
    /// Create a new builder with a sub-path
    pub fn push<S: Into<String>>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.into());
        Self {
            tensors: self.tensors.clone(),
            path,
            device: self.device.clone(),
        }
    }
    
    /// Prepend a prefix to the current path
    pub fn prepend<S: Into<String>>(mut self, s: S) -> Self {
        self.path.insert(0, s.into());
        self
    }
    
    /// Get the current path as a string
    fn path(&self) -> String {
        self.path.join(".")
    }
    
    /// Get a tensor with the given name
    pub fn get<S: AsRef<str>>(&self, name: S) -> CandleResult<Tensor> {
        let full_name = if self.path.is_empty() {
            name.as_ref().to_string()
        } else {
            format!("{}.{}", self.path(), name.as_ref())
        };
        
        self.get_quantized(&full_name)?
            .dequantize()
    }
    
    /// Get a quantized tensor without dequantizing
    pub fn get_quantized<S: AsRef<str>>(&self, name: S) -> CandleResult<&QuantizedTensor> {
        let name = name.as_ref();
        self.tensors.get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("Tensor '{}' not found", name)))
    }
    
    /// Get a tensor with shape checking
    pub fn get_with_shape<S: AsRef<str>>(&self, name: S, expected_shape: &[usize]) -> CandleResult<Tensor> {
        let tensor = self.get(name.as_ref())?;
        let shape = tensor.shape();
        
        if shape.dims() != expected_shape {
            return Err(candle_core::Error::Msg(format!(
                "Shape mismatch for '{}': expected {:?}, got {:?}",
                name.as_ref(), expected_shape, shape.dims()
            )));
        }
        
        Ok(tensor)
    }
    
    /// Check if a tensor exists
    pub fn contains<S: AsRef<str>>(&self, name: S) -> bool {
        let full_name = if self.path.is_empty() {
            name.as_ref().to_string()
        } else {
            format!("{}.{}", self.path(), name.as_ref())
        };
        
        self.tensors.contains_key(&full_name)
    }
    
    /// List all tensor names with the current prefix
    pub fn list_tensors(&self) -> Vec<String> {
        let prefix = self.path();
        let prefix_with_dot = if prefix.is_empty() {
            String::new()
        } else {
            format!("{}.", prefix)
        };
        
        self.tensors.keys()
            .filter(|k| k.starts_with(&prefix_with_dot) || (prefix.is_empty() && !k.contains('.')))
            .cloned()
            .collect()
    }
    
    /// Get all tensor names in the builder
    pub fn all_tensors(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }
    
    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Apply a function to get a tensor (useful for error handling)
    pub fn get_or_init<S: AsRef<str>, F>(&self, name: S, f: F) -> CandleResult<Tensor>
    where
        F: FnOnce() -> CandleResult<Tensor>,
    {
        match self.get(name) {
            Ok(tensor) => Ok(tensor),
            Err(_) => f(),
        }
    }
    
    /// Get multiple tensors at once
    pub fn get_many<S: AsRef<str>>(&self, names: &[S]) -> CandleResult<Vec<Tensor>> {
        names.iter()
            .map(|name| self.get(name))
            .collect()
    }
}

/// Extension trait for using QuantizedVarBuilder with model components
pub trait QuantizedVarBuilderExt {
    /// Get a linear layer's weight and bias
    fn get_linear(&self, name: &str) -> CandleResult<(Tensor, Option<Tensor>)>;
    
    /// Get layer normalization parameters
    fn get_layer_norm(&self, name: &str) -> CandleResult<(Tensor, Tensor)>;
    
    /// Get convolution parameters
    fn get_conv2d(&self, name: &str) -> CandleResult<(Tensor, Option<Tensor>)>;
}

impl QuantizedVarBuilderExt for QuantizedVarBuilder {
    fn get_linear(&self, name: &str) -> CandleResult<(Tensor, Option<Tensor>)> {
        let weight = self.get(&format!("{}.weight", name))?;
        let bias = self.get(&format!("{}.bias", name)).ok();
        Ok((weight, bias))
    }
    
    fn get_layer_norm(&self, name: &str) -> CandleResult<(Tensor, Tensor)> {
        let weight = self.get(&format!("{}.weight", name))?;
        let bias = self.get(&format!("{}.bias", name))?;
        Ok((weight, bias))
    }
    
    fn get_conv2d(&self, name: &str) -> CandleResult<(Tensor, Option<Tensor>)> {
        let weight = self.get(&format!("{}.weight", name))?;
        let bias = self.get(&format!("{}.bias", name)).ok();
        Ok((weight, bias))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_path_building() {
        // This would require actual GGUF data to test properly
        // For now, just test the path building logic
        let tensors = Arc::new(HashMap::new());
        let device = Device::Cpu;
        
        let vb = QuantizedVarBuilder {
            tensors,
            path: vec![],
            device,
        };
        
        let sub_vb = vb.push("model").push("layer1");
        assert_eq!(sub_vb.path(), "model.layer1");
        
        let prefixed_vb = vb.prepend("prefix");
        assert_eq!(prefixed_vb.path(), "prefix");
    }
}