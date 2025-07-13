use candle_core::{DType, Device, Result as CandleResult, Tensor};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

/// FP8 E4M3 format (used in SD3 models)
/// Sign: 1 bit, Exponent: 4 bits, Mantissa: 3 bits
#[derive(Debug, Clone, Copy)]
pub struct Fp8E4M3(u8);

impl Fp8E4M3 {
    /// Convert FP8 E4M3 to f32
    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        
        // Extract components
        let sign = (bits >> 7) & 1;
        let exponent = (bits >> 3) & 0xF; // 4 bits
        let mantissa = bits & 0x7; // 3 bits
        
        // Handle special cases
        if exponent == 0xF {
            // Infinity or NaN
            if mantissa == 0 {
                return if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY };
            } else {
                return f32::NAN;
            }
        }
        
        if exponent == 0 {
            // Zero or subnormal
            if mantissa == 0 {
                return 0.0;
            }
            // Subnormal number
            let mantissa_f32 = mantissa as f32 / 8.0; // 2^-3
            let value = mantissa_f32 * 2.0f32.powi(-6); // Bias adjusted
            return if sign == 1 { -value } else { value };
        }
        
        // Normal number
        let bias = 7; // E4M3 bias
        let unbiased_exp = exponent as i32 - bias;
        let mantissa_f32 = 1.0 + (mantissa as f32 / 8.0); // 1.mantissa
        let value = mantissa_f32 * 2.0f32.powi(unbiased_exp);
        
        if sign == 1 { -value } else { value }
    }
}

/// Convert a buffer of FP8 E4M3 values to f32
pub fn fp8_e4m3_to_f32(data: &[u8]) -> Vec<f32> {
    data.iter()
        .map(|&byte| Fp8E4M3(byte).to_f32())
        .collect()
}

/// Load FP8 safetensors and convert to FP16/FP32
pub fn load_fp8_safetensors(
    path: &std::path::Path,
    device: &Device,
    dtype: DType,
) -> CandleResult<HashMap<String, Tensor>> {
    use std::fs::File;
    use memmap2::MmapOptions;
    
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    
    let buffer = &mmap[..];
    let safetensors = SafeTensors::deserialize(buffer)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to deserialize safetensors: {}", e)))?;
    
    let mut tensors = HashMap::new();
    
    for (name, view) in safetensors.tensors() {
        // Check dtype as string since F8_E4M3 might not be in the enum
        let dtype_str = format!("{:?}", view.dtype());
        let tensor = if dtype_str.contains("F8_E4M3") || dtype_str.contains("F8E4M3") {
                // Convert FP8 to f32 first
                let data = view.data();
                let f32_data = fp8_e4m3_to_f32(data);
                
                // Create tensor from f32 data
                let shape = view.shape();
                let tensor = Tensor::from_vec(f32_data, shape, device)?;
                
                // Convert to target dtype if needed
                if dtype != DType::F32 {
                    tensor.to_dtype(dtype)?
                } else {
                    tensor
                }
            } else {
                match view.dtype() {
                    safetensors::Dtype::F32 => {
                // Standard f32 tensor
                let data = view.data();
                let f32_data: Vec<f32> = data.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                
                let shape = view.shape();
                Tensor::from_vec(f32_data, shape, device)?
            }
            safetensors::Dtype::F16 => {
                // F16 tensor
                let data = view.data();
                let f16_data: Vec<half::f16> = data.chunks_exact(2)
                    .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                
                let shape = view.shape();
                let f32_data: Vec<f32> = f16_data.iter().map(|&x| x.to_f32()).collect();
                Tensor::from_vec(f32_data, shape, device)?
            }
                    _ => {
                        eprintln!("Skipping tensor {} with unsupported dtype {:?}", name, view.dtype());
                        continue;
                    }
                }
            }
        };
        
        tensors.insert(name.to_string(), tensor);
    }
    
    Ok(tensors)
}

/// Create a VarBuilder from FP8 safetensors
pub fn varbuilder_from_fp8_safetensors(
    paths: &[std::path::PathBuf],
    dtype: DType,
    device: &Device,
) -> CandleResult<candle_nn::VarBuilder> {
    let mut all_tensors = HashMap::new();
    
    for path in paths {
        let tensors = load_fp8_safetensors(path, device, dtype)?;
        all_tensors.extend(tensors);
    }
    
    Ok(candle_nn::VarBuilder::from_tensors(all_tensors, dtype, device))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fp8_conversion() {
        // Test some known FP8 E4M3 values
        assert_eq!(Fp8E4M3(0x00).to_f32(), 0.0); // Zero
        assert_eq!(Fp8E4M3(0x38).to_f32(), 1.0); // One (0 01110 00)
        assert_eq!(Fp8E4M3(0xB8).to_f32(), -1.0); // Minus one (1 01110 00)
        
        // Test infinity
        assert_eq!(Fp8E4M3(0x78).to_f32(), f32::INFINITY); // +Inf (0 11110 00)
        assert_eq!(Fp8E4M3(0xF8).to_f32(), f32::NEG_INFINITY); // -Inf (1 11110 00)
    }
}