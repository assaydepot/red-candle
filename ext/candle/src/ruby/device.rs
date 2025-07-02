use magnus::Error;

use crate::ruby::errors::wrap_candle_err;
use ::candle_core::Device as CoreDevice;
use crate::ruby::Result as RbResult;

static CUDA_DEVICE: std::sync::Mutex<Option<CoreDevice>> = std::sync::Mutex::new(None);
static METAL_DEVICE: std::sync::Mutex<Option<CoreDevice>> = std::sync::Mutex::new(None);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[magnus::wrap(class = "Candle::Device")]
pub enum Device {
    Cpu,
    Cuda,
    Metal,
}

impl Device {
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
                let mut device = CUDA_DEVICE.lock().unwrap();
                if let Some(device) = device.as_ref() {
                    return Ok(device.clone());
                };
                let d = CoreDevice::new_cuda(0).map_err(wrap_candle_err)?;
                *device = Some(d.clone());
                Ok(d)
            }
            Self::Metal => {
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
        let device = magnus::RString::try_convert(val)?;
        let device = unsafe { device.as_str() }.unwrap();
        let device = match device {
            "cpu" => Device::Cpu,
            "cuda" => Device::Cuda,
            _ => return Err(Error::new(magnus::exception::arg_error(), "invalid device")),
        };
        Ok(device)
    }
}
