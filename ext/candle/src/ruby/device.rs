use magnus::Error;
use magnus::{function, method, class, RModule, Module, Object};

use ::candle_core::Device as CoreDevice;
use crate::ruby::Result;

#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::ruby::errors::wrap_candle_err;

#[cfg(feature = "cuda")]
static CUDA_DEVICE: std::sync::Mutex<Option<CoreDevice>> = std::sync::Mutex::new(None);

#[cfg(feature = "metal")]
static METAL_DEVICE: std::sync::Mutex<Option<CoreDevice>> = std::sync::Mutex::new(None);

/// Get list of available devices based on compile-time features
pub fn available_devices() -> Vec<String> {
    let devices = vec!["cpu".to_string()];
    
    #[cfg(all(feature = "cuda", not(force_cpu)))]
    let devices = {
        let mut devices = devices;
        devices.push("cuda".to_string());
        devices
    };
    
    #[cfg(all(feature = "metal", not(force_cpu)))]
    let devices = {
        let mut devices = devices;
        devices.push("metal".to_string());
        devices
    };
    
    devices
}

/// Get the default device based on what's available
pub fn default_device() -> Device {
    // Return based on compiled features, not detection
    #[cfg(all(feature = "metal", not(force_cpu)))]
    {
        Device::Metal
    }
    
    #[cfg(all(feature = "cuda", not(feature = "metal"), not(force_cpu)))]
    {
        Device::Cuda
    }
    
    #[cfg(not(any(all(feature = "metal", not(force_cpu)), all(feature = "cuda", not(feature = "metal"), not(force_cpu)))))]
    {
        Device::Cpu
    }
}

/// Get the best available device by checking runtime availability
pub fn best_device() -> Device {
    // Try devices in order of preference
    
    #[cfg(feature = "metal")]
    {
        // Check if Metal is actually available at runtime
        if CoreDevice::new_metal(0).is_ok() {
            return Device::Metal;
        }
    }
    
    #[cfg(feature = "cuda")]
    {
        // Check if CUDA is actually available at runtime
        if CoreDevice::new_cuda(0).is_ok() {
            return Device::Cuda;
        }
    }
    
    // Always fall back to CPU
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
    pub fn cuda() -> Result<Self> {
        #[cfg(not(feature = "cuda"))]
        {
            return Err(Error::new(
                magnus::exception::runtime_error(),
                "CUDA support not compiled in. Rebuild with CUDA available.",
            ));
        }
        
        #[cfg(feature = "cuda")]
        Ok(Self::Cuda)
    }

    /// Create a Metal device (Apple GPU)
    pub fn metal() -> Result<Self> {
        #[cfg(not(feature = "metal"))]
        {
            return Err(Error::new(
                magnus::exception::runtime_error(),
                "Metal support not compiled in. Rebuild on macOS.",
            ));
        }
        
        #[cfg(feature = "metal")]
        Ok(Self::Metal)
    }

    pub fn from_device(device: &CoreDevice) -> Self {
        match device {
            CoreDevice::Cpu => Self::Cpu,
            CoreDevice::Cuda(_) => Self::Cuda,
            CoreDevice::Metal(_) => Self::Metal,
        }
    }

    pub fn as_device(&self) -> Result<CoreDevice> {
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

    pub fn __eq__(&self, other: &Device) -> bool {
        self == other
    }
}

impl magnus::TryConvert for Device {
    fn try_convert(val: magnus::Value) -> Result<Self> {
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

pub fn init(rb_candle: RModule) -> Result<()> {
    let rb_device = rb_candle.define_class("Device", class::object())?;
    rb_device.define_singleton_method("cpu", function!(Device::cpu, 0))?;
    rb_device.define_singleton_method("cuda", function!(Device::cuda, 0))?;
    rb_device.define_singleton_method("metal", function!(Device::metal, 0))?;
    rb_device.define_singleton_method("available_devices", function!(available_devices, 0))?;
    rb_device.define_singleton_method("default", function!(default_device, 0))?;
    rb_device.define_singleton_method("best", function!(best_device, 0))?;
    rb_device.define_method("to_s", method!(Device::__str__, 0))?;
    rb_device.define_method("inspect", method!(Device::__repr__, 0))?;
    rb_device.define_method("==", method!(Device::__eq__, 1))?;
    Ok(())
}

