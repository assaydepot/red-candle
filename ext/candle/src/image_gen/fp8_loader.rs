use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::path::Path;

/// Check if a safetensors file contains FP8 tensors
pub fn is_fp8_safetensors(path: &Path) -> CandleResult<bool> {
    // For now, just check if the filename contains "fp8"
    // A more robust check would inspect the file metadata
    Ok(path.to_string_lossy().contains("fp8"))
}

/// Create a VarBuilder that handles FP8 models
pub fn create_fp8_aware_varbuilder(
    paths: &[std::path::PathBuf],
    dtype: DType,
    device: &Device,
) -> CandleResult<VarBuilder<'static>> {
    // Check if any path contains FP8
    let has_fp8 = paths.iter().any(|p| p.to_string_lossy().contains("fp8"));
    
    if has_fp8 {
        // For FP8 models, we need special handling
        // For now, return an error with a helpful message
        Err(candle_core::Error::Msg(
            "FP8 models require conversion to FP16/FP32. The model contains 8-bit floating point tensors which need to be converted for use with Candle.".to_string()
        ))
    } else {
        // Standard loading for non-FP8 models
        unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device) }
    }
}

/// Placeholder for FP8 to F32 conversion
/// In a full implementation, this would:
/// 1. Read the raw safetensors file
/// 2. Parse FP8 E4M3 values
/// 3. Convert to F32
/// 4. Create new tensors
pub fn convert_fp8_to_f32(fp8_value: u8) -> f32 {
    // FP8 E4M3 format: 1 sign, 4 exponent, 3 mantissa
    let sign = (fp8_value >> 7) & 1;
    let exponent = (fp8_value >> 3) & 0xF;
    let mantissa = fp8_value & 0x7;
    
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
        // Subnormal
        let mantissa_f32 = mantissa as f32 / 8.0;
        let value = mantissa_f32 * 2.0f32.powi(-6);
        return if sign == 1 { -value } else { value };
    }
    
    // Normal number
    let bias = 7;
    let unbiased_exp = exponent as i32 - bias;
    let mantissa_f32 = 1.0 + (mantissa as f32 / 8.0);
    let value = mantissa_f32 * 2.0f32.powi(unbiased_exp);
    
    if sign == 1 { -value } else { value }
}