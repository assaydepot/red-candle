// Test module for GGML quantization
use super::*;
use candle_core::Device;
use half::f16;

#[test]
fn test_q4_0_dequantization() {
        // Create test Q4_0 block with known values
        // Scale = 1.0, values = [0, 1, 2, ..., 15, -8, -7, ..., -1]
        let scale = f16::from_f32(1.0);
        let scale_bytes = scale.to_le_bytes();
        
        // Pack 32 4-bit values into 16 bytes
        let mut data = vec![scale_bytes[0], scale_bytes[1]];
        
        // First 16 values: 0-15 (subtract 8 for actual values: -8 to 7)
        for i in 0..8 {
            let low = i * 2;
            let high = i * 2 + 1;
            data.push((high << 4) | low);
        }
        
        // Next 16 values: repeat pattern
        for i in 0..8 {
            let low = i * 2;
            let high = i * 2 + 1;
            data.push((high << 4) | low);
        }
        
        // Dequantize
        let device = Device::Cpu;
        let shape = vec![32];
        let tensor = dequantize_ggml(&data, &shape, candle_core::quantized::GgmlDType::Q4_0, &device).unwrap();
        
        // Check shape
        assert_eq!(tensor.dims(), &[32]);
        
        // Convert to vec and check some values
        let values: Vec<f32> = tensor.to_vec1().unwrap();
        
        // First value should be 0 - 8 = -8
        assert_eq!(values[0], -8.0);
        // Second value should be 1 - 8 = -7
        assert_eq!(values[1], -7.0);
        // Value at index 8 should be 8 - 8 = 0
        assert_eq!(values[8], 0.0);
        // Last value should be 15 - 8 = 7
        assert_eq!(values[15], 7.0);
        
        println!("Q4_0 dequantization test passed!");
        println!("First 8 values: {:?}", &values[0..8]);
    }
    
    #[test]
    fn test_q8_0_dequantization() {
        // Create test Q8_0 block
        // Scale = 0.5, values = [-128, -64, 0, 64, 127, ...]
        let scale = f16::from_f32(0.5);
        let scale_bytes = scale.to_le_bytes();
        
        let mut data = vec![scale_bytes[0], scale_bytes[1]];
        
        // Add 32 signed 8-bit values
        let test_values: Vec<i8> = vec![
            -128, -64, -32, -16, -8, -4, -2, -1,
            0, 1, 2, 4, 8, 16, 32, 64,
            127, 100, 50, 25, 10, 5, 2, 1,
            0, -1, -2, -5, -10, -25, -50, -100,
        ];
        
        for val in test_values.iter() {
            data.push(*val as u8);
        }
        
        // Dequantize
        let device = Device::Cpu;
        let shape = vec![32];
        let tensor = dequantize_ggml(&data, &shape, candle_core::quantized::GgmlDType::Q8_0, &device).unwrap();
        
        // Check shape
        assert_eq!(tensor.dims(), &[32]);
        
        // Convert to vec and check values
        let values: Vec<f32> = tensor.to_vec1().unwrap();
        
        // Check some values (scaled by 0.5)
        assert_eq!(values[0], -128.0 * 0.5); // -64.0
        assert_eq!(values[8], 0.0);           // 0 * 0.5 = 0
        assert_eq!(values[16], 127.0 * 0.5);  // 63.5
        
        println!("Q8_0 dequantization test passed!");
        println!("First 8 values: {:?}", &values[0..8]);
    }
    
    #[test]
    fn test_f16_dequantization() {
        // Test F16 dequantization
        let test_values = vec![1.0f32, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0, 3.14159];
        let mut data = Vec::new();
        
        for val in test_values.iter() {
            let f16_val = f16::from_f32(*val);
            data.extend_from_slice(&f16_val.to_le_bytes());
        }
        
        // Dequantize
        let device = Device::Cpu;
        let shape = vec![test_values.len()];
        let tensor = dequantize_ggml(&data, &shape, candle_core::quantized::GgmlDType::F16, &device).unwrap();
        
        // Check values
        let values: Vec<f32> = tensor.to_vec1().unwrap();
        
        for (i, (expected, actual)) in test_values.iter().zip(values.iter()).enumerate() {
            let diff = (expected - actual).abs();
            assert!(diff < 0.001, "Value {} mismatch: expected {}, got {}", i, expected, actual);
        }
        
        println!("F16 dequantization test passed!");
    }
    
    #[test]
    fn test_multiple_blocks() {
        // Test with multiple Q4_0 blocks (64 values = 2 blocks)
        let scale1 = f16::from_f32(2.0);
        let scale2 = f16::from_f32(0.5);
        
        let mut data = Vec::new();
        
        // First block with scale 2.0
        data.extend_from_slice(&scale1.to_le_bytes());
        for i in 0..16 {
            data.push(0x88); // All nibbles = 8 (value 0 after offset)
        }
        
        // Second block with scale 0.5
        data.extend_from_slice(&scale2.to_le_bytes());
        for i in 0..16 {
            data.push(0xFF); // All nibbles = 15 (value 7 after offset)
        }
        
        // Dequantize
        let device = Device::Cpu;
        let shape = vec![64];
        let tensor = dequantize_ggml(&data, &shape, candle_core::quantized::GgmlDType::Q4_0, &device).unwrap();
        
        let values: Vec<f32> = tensor.to_vec1().unwrap();
        
        // First block values should all be 0
        for i in 0..32 {
            assert_eq!(values[i], 0.0, "First block value {} should be 0", i);
        }
        
        // Second block values should all be 7 * 0.5 = 3.5
        for i in 32..64 {
            assert_eq!(values[i], 3.5, "Second block value {} should be 3.5", i);
        }
        
        println!("Multiple blocks test passed!");
    }

