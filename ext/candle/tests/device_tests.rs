use candle_core::Device as CoreDevice;

#[test]
fn test_device_creation() {
    // CPU device should always work
    let cpu = CoreDevice::Cpu;
    assert!(matches!(cpu, CoreDevice::Cpu));
    
    // Test device display
    assert_eq!(format!("{:?}", cpu), "Cpu");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA hardware"]
fn test_cuda_device_creation() {
    // This might fail if no CUDA device is available
    match CoreDevice::new_cuda(0) {
        Ok(device) => assert!(matches!(device, CoreDevice::Cuda(_))),
        Err(_) => println!("No CUDA device available for testing"),
    }
}

#[cfg(feature = "metal")] 
#[test]
#[ignore = "requires Metal hardware"]
fn test_metal_device_creation() {
    // This might fail if no Metal device is available
    match CoreDevice::new_metal(0) {
        Ok(device) => assert!(matches!(device, CoreDevice::Metal(_))),
        Err(_) => println!("No Metal device available for testing"),
    }
}

#[test]
fn test_device_matching() {
    let cpu1 = CoreDevice::Cpu;
    let cpu2 = CoreDevice::Cpu;
    
    // Same device types should match
    assert!(matches!(cpu1, CoreDevice::Cpu));
    assert!(matches!(cpu2, CoreDevice::Cpu));
}