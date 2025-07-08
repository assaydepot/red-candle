use magnus::Error;

use crate::ruby::errors::wrap_candle_err;
use ::candle_core::Device as CoreDevice;
use crate::ruby::Result as RbResult;

static CUDA_DEVICE: std::sync::Mutex<Option<CoreDevice>> = std::sync::Mutex::new(None);
static METAL_DEVICE: std::sync::Mutex<Option<CoreDevice>> = std::sync::Mutex::new(None);

/// Get list of available devices based on compile-time features
pub fn available_devices() -> Vec<String> {
    let mut devices = vec!["cpu".to_string()];
    
    #[cfg(all(feature = "cuda", not(force_cpu)))]
    {
        devices.push("cuda".to_string());
    }
    
    #[cfg(all(feature = "metal", not(force_cpu)))]
    {
        devices.push("metal".to_string());
    }
    
    devices
}

/// Get the default device based on what's available
pub fn default_device() -> Device {
    // Use compile-time defaults
    #[cfg(all(has_metal, not(force_cpu)))]
    {
        return Device::Metal;
    }
    
    #[cfg(all(has_cuda, not(has_metal), not(force_cpu)))]
    {
        return Device::Cuda;
    }
    
    Device::Cpu
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[magnus::wrap(class = "Candle::Device")]
pub enum Device {
    Cpu,
    Cuda,
    Metal,
}

impl Device {
    /// Create a CPU device
    pub fn cpu() -> Self {
        Self::Cpu
    }

    /// Create a CUDA device (GPU)
    pub fn cuda() -> Self {
        Self::Cuda
    }

    /// Create a Metal device (Apple GPU)
    pub fn metal() -> Self {
        Self::Metal
    }

    pub fn from_device(device: &CoreDevice) -> Self {
        match device {
            CoreDevice::Cpu => Self::Cpu,
            CoreDevice::Cuda(_) => Self::Cuda,
            CoreDevice::Metal(_) => Self::Metal,
        }
    }

    pub fn as_device(&self) -> RbResult<CoreDevice> {
        match self {
            Self::Cpu => Ok(CoreDevice::Cpu),
            Self::Cuda => {
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(Error::new(
                        magnus::exception::runtime_error(),
                        "CUDA support not compiled in. Rebuild with CUDA available.",
                    ));
                }
                
                #[cfg(feature = "cuda")]
                {
                    let mut device = CUDA_DEVICE.lock().unwrap();
                    if let Some(device) = device.as_ref() {
                        return Ok(device.clone());
                    };
                    // Note: new_cuda() is used here (not cuda_if_available) because
                    // we want to fail if CUDA isn't available at runtime, not fall back to CPU
                    let d = CoreDevice::new_cuda(0).map_err(wrap_candle_err)?;
                    *device = Some(d.clone());
                    Ok(d)
                }
            }
            Self::Metal => {
                #[cfg(not(feature = "metal"))]
                {
                    return Err(Error::new(
                        magnus::exception::runtime_error(),
                        "Metal support not compiled in. Rebuild on macOS.",
                    ));
                }
                
                #[cfg(feature = "metal")]
                {
                    let mut device = METAL_DEVICE.lock().unwrap();
                    if let Some(device) = device.as_ref() {
                        return Ok(device.clone());
                    };
                    let d = CoreDevice::new_metal(0).map_err(wrap_candle_err)?;
                    *device = Some(d.clone());
                    Ok(d)
                }
            }
        }
    }

    pub fn __repr__(&self) -> String {
        match self {
            Self::Cpu => "cpu".to_string(),
            Self::Cuda => "cuda".to_string(),
            Self::Metal => "metal".to_string(),
        }
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl magnus::TryConvert for Device {
    fn try_convert(val: magnus::Value) -> RbResult<Self> {
        // First check if it's already a wrapped Device object
        if let Ok(device) = <magnus::typed_data::Obj<Device> as magnus::TryConvert>::try_convert(val) {
            return Ok(*device);
        }

        // Otherwise try to convert from string
        let device = magnus::RString::try_convert(val)?;
        let device = unsafe { device.as_str() }.unwrap();
        let device = match device {
            "cpu" => Device::Cpu,
            "cuda" => Device::Cuda,
            "metal" => Device::Metal,
            _ => return Err(Error::new(magnus::exception::arg_error(), "invalid device")),
        };
        Ok(device)
    }
}
