use candle_core::{Tensor, Device, DType};

#[test]
fn test_tensor_creation() {
    let device = Device::Cpu;
    
    // Test tensor creation from slice
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(&data[..], &device).unwrap();
    assert_eq!(tensor.dims(), &[4]);
    assert_eq!(tensor.dtype(), DType::F32);
    
    // Test zeros
    let zeros = Tensor::zeros(&[2, 3], DType::F32, &device).unwrap();
    assert_eq!(zeros.dims(), &[2, 3]);
    
    // Test ones
    let ones = Tensor::ones(&[3, 2], DType::F32, &device).unwrap();
    assert_eq!(ones.dims(), &[3, 2]);
}

#[test]
fn test_tensor_arithmetic() {
    let device = Device::Cpu;
    
    let a = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
    let b = Tensor::new(&[4.0f32, 5.0, 6.0], &device).unwrap();
    
    // Addition
    let sum = a.add(&b).unwrap();
    let sum_vec: Vec<f32> = sum.to_vec1().unwrap();
    assert_eq!(sum_vec, vec![5.0, 7.0, 9.0]);
    
    // Subtraction
    let diff = a.sub(&b).unwrap();
    let diff_vec: Vec<f32> = diff.to_vec1().unwrap();
    assert_eq!(diff_vec, vec![-3.0, -3.0, -3.0]);
    
    // Multiplication
    let prod = a.mul(&b).unwrap();
    let prod_vec: Vec<f32> = prod.to_vec1().unwrap();
    assert_eq!(prod_vec, vec![4.0, 10.0, 18.0]);
}

#[test]
fn test_tensor_reshape() {
    let device = Device::Cpu;
    
    let tensor = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &device).unwrap();
    
    // Reshape to 2x3
    let reshaped = tensor.reshape(&[2, 3]).unwrap();
    assert_eq!(reshaped.dims(), &[2, 3]);
    
    // Reshape to 3x2
    let reshaped = tensor.reshape(&[3, 2]).unwrap();
    assert_eq!(reshaped.dims(), &[3, 2]);
}

#[test]
fn test_tensor_transpose() {
    let device = Device::Cpu;
    
    let tensor = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();
    
    let transposed = tensor.transpose(0, 1).unwrap();
    assert_eq!(transposed.dims(), &[2, 2]);
    
    let values: Vec<f32> = transposed.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(values, vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_tensor_reduction() {
    let device = Device::Cpu;
    
    let tensor = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device).unwrap();
    
    // Sum
    let sum = tensor.sum_all().unwrap();
    let sum_val: f32 = sum.to_scalar().unwrap();
    assert_eq!(sum_val, 10.0);
    
    // Mean
    let mean = tensor.mean_all().unwrap();
    let mean_val: f32 = mean.to_scalar().unwrap();
    assert_eq!(mean_val, 2.5);
}

#[test]
fn test_tensor_indexing() {
    let device = Device::Cpu;
    
    let tensor = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], &device).unwrap();
    
    // Get element at index 0
    let elem = tensor.get(0).unwrap();
    let val: f32 = elem.to_scalar().unwrap();
    assert_eq!(val, 10.0);
    
    // Get element at index 2
    let elem = tensor.get(2).unwrap();
    let val: f32 = elem.to_scalar().unwrap();
    assert_eq!(val, 30.0);
}

#[test]
fn test_tensor_matmul() {
    let device = Device::Cpu;
    
    // 2x3 matrix
    let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &device)
        .unwrap()
        .reshape(&[2, 3])
        .unwrap();
    
    // 3x2 matrix
    let b = Tensor::new(&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], &device)
        .unwrap()
        .reshape(&[3, 2])
        .unwrap();
    
    // Matrix multiplication
    let result = a.matmul(&b).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    
    let values: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
    // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12, 4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]
    // = [58, 64, 139, 154]
    assert_eq!(values, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_tensor_where() {
    let device = Device::Cpu;
    
    // Create a condition tensor where values > 0 are treated as true
    let cond_values = Tensor::new(&[1.0f32, 0.0, 1.0], &device).unwrap();
    let cond = cond_values.gt(&Tensor::zeros(cond_values.shape(), DType::F32, &device).unwrap()).unwrap();
    
    let on_true = Tensor::new(&[10.0f32, 20.0, 30.0], &device).unwrap();
    let on_false = Tensor::new(&[100.0f32, 200.0, 300.0], &device).unwrap();
    
    let result = cond.where_cond(&on_true, &on_false).unwrap();
    let values: Vec<f32> = result.to_vec1().unwrap();
    assert_eq!(values, vec![10.0, 200.0, 30.0]);
}

#[test]
fn test_tensor_narrow() {
    let device = Device::Cpu;
    
    let tensor = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &device).unwrap();
    
    // Narrow from index 1, length 3
    let narrowed = tensor.narrow(0, 1, 3).unwrap();
    let values: Vec<f32> = narrowed.to_vec1().unwrap();
    assert_eq!(values, vec![2.0, 3.0, 4.0]);
}