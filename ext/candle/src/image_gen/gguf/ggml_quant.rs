use candle_core::{Device, Result as CandleResult, Tensor};
use candle_core::quantized::GgmlDType;
use half::f16;

/// GGML quantization block sizes
const QK4_0: usize = 32;
const QK5_0: usize = 32;
const QK8_0: usize = 32;

/// Dequantize GGML tensor data to f32
pub fn dequantize_ggml(
    data: &[u8],
    shape: &[usize],
    dtype: GgmlDType,
    device: &Device,
) -> CandleResult<Tensor> {
    let elem_count: usize = shape.iter().product();
    
    let f32_data = match dtype {
        GgmlDType::Q4_0 => dequantize_q4_0(data, elem_count)?,
        GgmlDType::Q5_0 => dequantize_q5_0(data, elem_count)?,
        GgmlDType::Q8_0 => dequantize_q8_0(data, elem_count)?,
        GgmlDType::Q4_1 => dequantize_q4_1(data, elem_count)?,
        GgmlDType::Q5_1 => dequantize_q5_1(data, elem_count)?,
        GgmlDType::F16 => dequantize_f16(data, elem_count)?,
        GgmlDType::F32 => dequantize_f32(data, elem_count)?,
        _ => {
            eprintln!("Warning: Unsupported quantization type {:?}, using placeholder", dtype);
            vec![0.0f32; elem_count]
        }
    };
    
    Tensor::from_vec(f32_data, shape, device)
}

/// Q4_0 quantization format:
/// - 32 weights per block
/// - Each block: 2 bytes (f16 scale) + 16 bytes (32 x 4-bit weights)
/// - Total: 18 bytes per block
fn dequantize_q4_0(data: &[u8], elem_count: usize) -> CandleResult<Vec<f32>> {
    let block_size = 18; // 2 bytes scale + 16 bytes data
    let n_blocks = data.len() / block_size;
    
    if n_blocks * QK4_0 != elem_count {
        return Err(candle_core::Error::Msg(format!(
            "Q4_0: element count mismatch. Expected {}, got {}",
            n_blocks * QK4_0, elem_count
        )));
    }
    
    let mut output = Vec::with_capacity(elem_count);
    
    for block_idx in 0..n_blocks {
        let block_start = block_idx * block_size;
        
        // Read scale as f16
        let scale_bytes = [data[block_start], data[block_start + 1]];
        let scale = f16::from_le_bytes(scale_bytes).to_f32();
        
        // Read 4-bit quantized values (16 bytes = 32 x 4-bit values)
        let quants = &data[block_start + 2..block_start + 18];
        
        // Dequantize 32 values
        for i in 0..QK4_0 {
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 {
                quants[byte_idx] & 0x0F
            } else {
                (quants[byte_idx] >> 4) & 0x0F
            };
            
            // Q4_0 uses symmetric quantization with offset of 8
            let dequantized = scale * (nibble as f32 - 8.0);
            output.push(dequantized);
        }
    }
    
    Ok(output)
}

/// Q5_0 quantization format:
/// - 32 weights per block  
/// - Each block: 2 bytes (f16 scale) + 4 bytes (32 5th bits) + 16 bytes (32 x 4-bit low bits)
/// - Total: 22 bytes per block
fn dequantize_q5_0(data: &[u8], elem_count: usize) -> CandleResult<Vec<f32>> {
    let block_size = 22; // 2 bytes scale + 4 bytes high bits + 16 bytes low bits
    let n_blocks = data.len() / block_size;
    
    if n_blocks * QK5_0 != elem_count {
        return Err(candle_core::Error::Msg(format!(
            "Q5_0: element count mismatch. Expected {}, got {}",
            n_blocks * QK5_0, elem_count
        )));
    }
    
    let mut output = Vec::with_capacity(elem_count);
    
    for block_idx in 0..n_blocks {
        let block_start = block_idx * block_size;
        
        // Read scale as f16
        let scale_bytes = [data[block_start], data[block_start + 1]];
        let scale = f16::from_le_bytes(scale_bytes).to_f32();
        
        // Read high bits (5th bits for 32 values packed into 4 bytes)
        let high_bits = &data[block_start + 2..block_start + 6];
        
        // Read low 4 bits (16 bytes = 32 x 4-bit values)
        let low_bits = &data[block_start + 6..block_start + 22];
        
        // Dequantize 32 values
        for i in 0..QK5_0 {
            // Extract low 4 bits
            let byte_idx = i / 2;
            let low_nibble = if i % 2 == 0 {
                low_bits[byte_idx] & 0x0F
            } else {
                (low_bits[byte_idx] >> 4) & 0x0F
            };
            
            // Extract 5th bit
            let high_bit_byte_idx = i / 8;
            let high_bit_pos = i % 8;
            let high_bit = ((high_bits[high_bit_byte_idx] >> high_bit_pos) & 1) << 4;
            
            // Combine to get 5-bit value
            let quant_value = low_nibble | high_bit;
            
            // Q5_0 uses symmetric quantization with offset of 16
            let dequantized = scale * (quant_value as f32 - 16.0);
            output.push(dequantized);
        }
    }
    
    Ok(output)
}

/// Q8_0 quantization format:
/// - 32 weights per block
/// - Each block: 2 bytes (f16 scale) + 32 bytes (32 x 8-bit weights)
/// - Total: 34 bytes per block
fn dequantize_q8_0(data: &[u8], elem_count: usize) -> CandleResult<Vec<f32>> {
    let block_size = 34; // 2 bytes scale + 32 bytes data
    let n_blocks = data.len() / block_size;
    
    if n_blocks * QK8_0 != elem_count {
        return Err(candle_core::Error::Msg(format!(
            "Q8_0: element count mismatch. Expected {}, got {}",
            n_blocks * QK8_0, elem_count
        )));
    }
    
    let mut output = Vec::with_capacity(elem_count);
    
    for block_idx in 0..n_blocks {
        let block_start = block_idx * block_size;
        
        // Read scale as f16
        let scale_bytes = [data[block_start], data[block_start + 1]];
        let scale = f16::from_le_bytes(scale_bytes).to_f32();
        
        // Read 8-bit quantized values
        let quants = &data[block_start + 2..block_start + 34];
        
        // Dequantize 32 values
        for i in 0..QK8_0 {
            // Q8_0 uses signed 8-bit integers
            let quant_value = quants[i] as i8;
            let dequantized = scale * (quant_value as f32);
            output.push(dequantized);
        }
    }
    
    Ok(output)
}

/// Q4_1 quantization format:
/// - 32 weights per block
/// - Each block: 2 bytes (f16 scale) + 2 bytes (f16 min) + 16 bytes (32 x 4-bit weights)
/// - Total: 20 bytes per block
fn dequantize_q4_1(data: &[u8], elem_count: usize) -> CandleResult<Vec<f32>> {
    let block_size = 20; // 2 bytes scale + 2 bytes min + 16 bytes data
    let n_blocks = data.len() / block_size;
    
    if n_blocks * QK4_0 != elem_count {
        return Err(candle_core::Error::Msg(format!(
            "Q4_1: element count mismatch. Expected {}, got {}",
            n_blocks * QK4_0, elem_count
        )));
    }
    
    let mut output = Vec::with_capacity(elem_count);
    
    for block_idx in 0..n_blocks {
        let block_start = block_idx * block_size;
        
        // Read scale and min as f16
        let scale_bytes = [data[block_start], data[block_start + 1]];
        let scale = f16::from_le_bytes(scale_bytes).to_f32();
        
        let min_bytes = [data[block_start + 2], data[block_start + 3]];
        let min_val = f16::from_le_bytes(min_bytes).to_f32();
        
        // Read 4-bit quantized values
        let quants = &data[block_start + 4..block_start + 20];
        
        // Dequantize 32 values
        for i in 0..QK4_0 {
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 {
                quants[byte_idx] & 0x0F
            } else {
                (quants[byte_idx] >> 4) & 0x0F
            };
            
            // Q4_1 uses scale and min
            let dequantized = scale * (nibble as f32) + min_val;
            output.push(dequantized);
        }
    }
    
    Ok(output)
}

/// Q5_1 quantization format:
/// - 32 weights per block  
/// - Each block: 2 bytes (f16 scale) + 2 bytes (f16 min) + 4 bytes (32 5th bits) + 16 bytes (32 x 4-bit low bits)
/// - Total: 24 bytes per block
fn dequantize_q5_1(data: &[u8], elem_count: usize) -> CandleResult<Vec<f32>> {
    let block_size = 24; // 2 bytes scale + 2 bytes min + 4 bytes high bits + 16 bytes low bits
    let n_blocks = data.len() / block_size;
    
    if n_blocks * QK5_0 != elem_count {
        return Err(candle_core::Error::Msg(format!(
            "Q5_1: element count mismatch. Expected {}, got {}",
            n_blocks * QK5_0, elem_count
        )));
    }
    
    let mut output = Vec::with_capacity(elem_count);
    
    for block_idx in 0..n_blocks {
        let block_start = block_idx * block_size;
        
        // Read scale and min as f16
        let scale_bytes = [data[block_start], data[block_start + 1]];
        let scale = f16::from_le_bytes(scale_bytes).to_f32();
        
        let min_bytes = [data[block_start + 2], data[block_start + 3]];
        let min_val = f16::from_le_bytes(min_bytes).to_f32();
        
        // Read high bits
        let high_bits = &data[block_start + 4..block_start + 8];
        
        // Read low 4 bits
        let low_bits = &data[block_start + 8..block_start + 24];
        
        // Dequantize 32 values
        for i in 0..QK5_0 {
            // Extract low 4 bits
            let byte_idx = i / 2;
            let low_nibble = if i % 2 == 0 {
                low_bits[byte_idx] & 0x0F
            } else {
                (low_bits[byte_idx] >> 4) & 0x0F
            };
            
            // Extract 5th bit
            let high_bit_byte_idx = i / 8;
            let high_bit_pos = i % 8;
            let high_bit = ((high_bits[high_bit_byte_idx] >> high_bit_pos) & 1) << 4;
            
            // Combine to get 5-bit value
            let quant_value = low_nibble | high_bit;
            
            // Q5_1 uses scale and min
            let dequantized = scale * (quant_value as f32) + min_val;
            output.push(dequantized);
        }
    }
    
    Ok(output)
}

/// Dequantize f16 data
fn dequantize_f16(data: &[u8], elem_count: usize) -> CandleResult<Vec<f32>> {
    if data.len() != elem_count * 2 {
        return Err(candle_core::Error::Msg(format!(
            "F16: data size mismatch. Expected {} bytes, got {}",
            elem_count * 2, data.len()
        )));
    }
    
    let mut output = Vec::with_capacity(elem_count);
    
    for i in 0..elem_count {
        let byte_idx = i * 2;
        let bytes = [data[byte_idx], data[byte_idx + 1]];
        let value = f16::from_le_bytes(bytes).to_f32();
        output.push(value);
    }
    
    Ok(output)
}

/// Dequantize f32 data (no quantization)
fn dequantize_f32(data: &[u8], elem_count: usize) -> CandleResult<Vec<f32>> {
    if data.len() != elem_count * 4 {
        return Err(candle_core::Error::Msg(format!(
            "F32: data size mismatch. Expected {} bytes, got {}",
            elem_count * 4, data.len()
        )));
    }
    
    let mut output = Vec::with_capacity(elem_count);
    
    for i in 0..elem_count {
        let byte_idx = i * 4;
        let bytes = [
            data[byte_idx],
            data[byte_idx + 1],
            data[byte_idx + 2],
            data[byte_idx + 3],
        ];
        let value = f32::from_le_bytes(bytes);
        output.push(value);
    }
    
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_q4_0_block_size() {
        // Q4_0: 2 bytes scale + 16 bytes data = 18 bytes per 32 elements
        let data = vec![0u8; 18];
        let result = dequantize_q4_0(&data, 32).unwrap();
        assert_eq!(result.len(), 32);
    }
    
    #[test]
    fn test_q5_0_block_size() {
        // Q5_0: 2 bytes scale + 4 bytes high + 16 bytes low = 22 bytes per 32 elements
        let data = vec![0u8; 22];
        let result = dequantize_q5_0(&data, 32).unwrap();
        assert_eq!(result.len(), 32);
    }
    
    #[test]
    fn test_q8_0_block_size() {
        // Q8_0: 2 bytes scale + 32 bytes data = 34 bytes per 32 elements
        let data = vec![0u8; 34];
        let result = dequantize_q8_0(&data, 32).unwrap();
        assert_eq!(result.len(), 32);
    }
}