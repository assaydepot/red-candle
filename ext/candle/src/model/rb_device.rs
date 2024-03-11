use magnus::Error;

use ::candle_core::Device;
use crate::model::errors::wrap_candle_err;

type RbResult<T> = Result<T, Error>;

static CUDA_DEVICE: std::sync::Mutex<Option<Device>> = std::sync::Mutex::new(None);
static METAL_DEVICE: std::sync::Mutex<Option<Device>> = std::sync::Mutex::new(None);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[magnus::wrap(class = "Candle::Device")]
pub enum RbDevice {
    Cpu,
    Cuda,
    Metal,
}

impl RbDevice {
    pub fn from_device(device: &Device) -> Self {
        match device {
            Device::Cpu => Self::Cpu,
            Device::Cuda(_) => Self::Cuda,
            Device::Metal(_) => Self::Metal,
        }
    }

    pub fn as_device(&self) -> RbResult<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Cuda => {
                let mut device = CUDA_DEVICE.lock().unwrap();
                if let Some(device) = device.as_ref() {
                    return Ok(device.clone());
                };
                let d = Device::new_cuda(0).map_err(wrap_candle_err)?;
                *device = Some(d.clone());
                Ok(d)
            }
            Self::Metal => {
                let mut device = METAL_DEVICE.lock().unwrap();
                if let Some(device) = device.as_ref() {
                    return Ok(device.clone());
                };
                let d = Device::new_metal(0).map_err(wrap_candle_err)?;
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

impl magnus::TryConvert for RbDevice {
    fn try_convert(val: magnus::Value) -> RbResult<Self> {
        let device = magnus::RString::try_convert(val)?;
        let device = unsafe { device.as_str() }.unwrap();
        let device = match device {
            "cpu" => RbDevice::Cpu,
            "cuda" => RbDevice::Cuda,
            _ => return Err(Error::new(magnus::exception::arg_error(), "invalid device")),
        };
        Ok(device)
    }
}
