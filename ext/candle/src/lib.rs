use magnus::{function, method, prelude::*, Error, Ruby};
use std::sync::Arc;

use ::candle_core::{quantized::QTensor, DType, Device, Tensor, WithDType};

type RbResult<T> = Result<T, Error>;
struct RbCandleErr {}

impl RbCandleErr {
    pub fn from(e: candle_core::Error) -> Error {
        Error::new(magnus::exception::runtime_error(), e.to_string())
    }
}

#[derive(Clone, Debug)]
struct RbShape(Vec<usize>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[magnus::wrap(class = "Candle::DType", free_immediately, size)]
struct RbDType(DType);

impl RbDType {
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

static CUDA_DEVICE: std::sync::Mutex<Option<Device>> = std::sync::Mutex::new(None);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[magnus::wrap(class = "Candle::Device")]
enum RbDevice {
    Cpu,
    Cuda,
}

impl RbDevice {
    fn from_device(device: &Device) -> Self {
        match device {
            Device::Cpu => Self::Cpu,
            Device::Cuda(_) => Self::Cuda,
        }
    }

    fn as_device(&self) -> RbResult<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Cuda => {
                let mut device = CUDA_DEVICE.lock().unwrap();
                if let Some(device) = device.as_ref() {
                    return Ok(device.clone());
                };
                let d = Device::new_cuda(0).map_err(RbCandleErr::from)?;
                *device = Some(d.clone());
                Ok(d)
            }
        }
    }
}

fn actual_index(t: &Tensor, dim: usize, index: i64) -> candle_core::Result<usize> {
    let dim = t.dim(dim)?;
    if 0 <= index {
        let index = index as usize;
        if dim <= index {
            candle_core::bail!("index {index} is too large for tensor dimension {dim}")
        }
        Ok(index)
    } else {
        if (dim as i64) < -index {
            candle_core::bail!("index {index} is too low for tensor dimension {dim}")
        }
        Ok((dim as i64 + index) as usize)
    }
}

fn actual_dim(t: &Tensor, dim: i64) -> candle_core::Result<usize> {
    let rank = t.rank();
    if 0 <= dim {
        let dim = dim as usize;
        if rank <= dim {
            candle_core::bail!("dimension index {dim} is too large for tensor rank {rank}")
        }
        Ok(dim)
    } else {
        if (rank as i64) < -dim {
            candle_core::bail!("dimension index {dim} is too low for tensor rank {rank}")
        }
        Ok((rank as i64 + dim) as usize)
    }
}

#[derive(Clone, Debug)]
#[magnus::wrap(class = "Candle::Tensor", free_immediately, size)]
struct RbTensor(Tensor);


impl RbTensor {
    fn new(array: Vec<f32>) -> Self {
        use Device::Cpu;
        Self(Tensor::new(array.as_slice(), &Cpu).unwrap())
    }

    fn shape(&self) -> Vec<usize> {
        self.0.dims().to_vec()
    }

    fn stride(&self) -> Vec<usize> {
        self.0.stride().to_vec()
    }

    fn dtype(&self) -> RbDType {
        RbDType(self.0.dtype())
    }

    fn rank(&self) -> usize {
        self.0.rank()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn sin(&self) -> RbResult<Self> {
        Ok(Self(self.0.sin().map_err(RbCandleErr::from)?))
    }

    fn cos(&self) -> RbResult<Self> {
        Ok(Self(self.0.cos().map_err(RbCandleErr::from)?))
    }

    fn log(&self) -> RbResult<Self> {
        Ok(Self(self.0.log().map_err(RbCandleErr::from)?))
    }

    fn sqr(&self) -> RbResult<Self> {
        Ok(Self(self.0.sqr().map_err(RbCandleErr::from)?))
    }

    fn sqrt(&self) -> RbResult<Self> {
        Ok(Self(self.0.sqrt().map_err(RbCandleErr::from)?))
    }

    fn recip(&self) -> RbResult<Self> {
        Ok(Self(self.0.recip().map_err(RbCandleErr::from)?))
    }

    fn exp(&self) -> RbResult<Self> {
        Ok(Self(self.0.exp().map_err(RbCandleErr::from)?))
    }

    fn powf(&self, n: f64) -> RbResult<Self> {
        Ok(Self(self.0.powf(n).map_err(RbCandleErr::from)?))
    }

    fn matmul(&self, other: &RbTensor) -> RbResult<Self> {
        Ok(Self(self.0.matmul(&other.0).map_err(RbCandleErr::from)?))
    }

    fn where_cond(&self, on_true: &RbTensor, on_false: &RbTensor) -> RbResult<Self> {
        Ok(Self(
            self.0
                .where_cond(&on_true.0, &on_false.0)
                .map_err(RbCandleErr::from)?,
        ))
    }

    fn __add__(&self, rhs: &RbTensor) -> RbResult<Self> {
        Ok(Self(self.0.add(&rhs.0).map_err(RbCandleErr::from)?))
    }

    fn __mul__(&self, rhs: &RbTensor) -> RbResult<Self> {
        Ok(Self(self.0.mul(&rhs.0).map_err(RbCandleErr::from)?))
    }

    fn __sub__(&self, rhs: &RbTensor) -> RbResult<Self> {
        Ok(Self(self.0.sub(&rhs.0).map_err(RbCandleErr::from)?))
    }

    fn reshape(&self, shape: Vec<usize>) -> RbResult<Self> {
        Ok(Self(self.0.reshape(shape).map_err(RbCandleErr::from)?))
    }

    fn broadcast_as(&self, shape: Vec<usize>) -> RbResult<Self> {
        Ok(Self(self.0.broadcast_as(shape).map_err(RbCandleErr::from)?))
    }

    fn broadcast_left(&self, shape: Vec<usize>) -> RbResult<Self> {
        Ok(Self(
            self.0.broadcast_left(shape).map_err(RbCandleErr::from)?,
        ))
    }

    fn squeeze(&self, dim: usize) -> RbResult<Self> {
        let dim = actual_dim(&self.0, dim as i64).map_err(RbCandleErr::from)?;
        Ok(Self(self.0.squeeze(dim).map_err(RbCandleErr::from)?))
    }

    fn unsqueeze(&self, dim: usize) -> RbResult<Self> {
        Ok(Self(self.0.unsqueeze(dim).map_err(RbCandleErr::from)?))
    }

    fn get(&self, index: usize) -> RbResult<Self> {
        let index = actual_index(&self.0, 0, index as i64).map_err(RbCandleErr::from)?;
        Ok(Self(self.0.get(index).map_err(RbCandleErr::from)?))
    }

    fn transpose(&self, dim1: usize, dim2: usize) -> RbResult<Self> {
        Ok(Self(
            self.0.transpose(dim1, dim2).map_err(RbCandleErr::from)?,
        ))
    }

    fn narrow(&self, dim: usize, start: usize, len: usize) -> RbResult<Self> {
        let dim = actual_dim(&self.0, dim as i64).map_err(RbCandleErr::from)?;
        let start = actual_index(&self.0, dim, start as i64).map_err(RbCandleErr::from)?;
        Ok(Self(
            self.0.narrow(dim, start, len).map_err(RbCandleErr::from)?,
        ))
    }

    fn argmax_keepdim(&self, dim: i64) -> RbResult<Self> {
        let dim = actual_dim(&self.0, dim).map_err(RbCandleErr::from)?;
        Ok(Self(self.0.argmax_keepdim(dim).map_err(RbCandleErr::from)?))
    }

    fn argmin_keepdim(&self, dim: i64) -> RbResult<Self> {
        let dim = actual_dim(&self.0, dim).map_err(RbCandleErr::from)?;
        Ok(Self(self.0.argmin_keepdim(dim).map_err(RbCandleErr::from)?))
    }

    fn max_keepdim(&self, dim: i64) -> RbResult<Self> {
        let dim = actual_dim(&self.0, dim).map_err(RbCandleErr::from)?;
        Ok(Self(self.0.max_keepdim(dim).map_err(RbCandleErr::from)?))
    }

    fn min_keepdim(&self, dim: i64) -> RbResult<Self> {
        let dim = actual_dim(&self.0, dim).map_err(RbCandleErr::from)?;
        Ok(Self(self.0.min_keepdim(dim).map_err(RbCandleErr::from)?))
    }

    fn sum_all(&self) -> RbResult<Self> {
        Ok(Self(self.0.sum_all().map_err(RbCandleErr::from)?))
    }

    fn mean_all(&self) -> RbTensor {
        let elements = self.0.elem_count();
        let sum = self.0.sum_all().unwrap();
        let mean = (sum / elements as f64).unwrap();
        RbTensor(mean)
    }

    fn flatten_from(&self, dim: i64) -> RbResult<Self> {
        let dim = actual_dim(&self.0, dim).map_err(RbCandleErr::from)?;
        Ok(Self(self.0.flatten_from(dim).map_err(RbCandleErr::from)?))
    }

    fn flatten_to(&self, dim: i64) -> RbResult<Self> {
        let dim = actual_dim(&self.0, dim).map_err(RbCandleErr::from)?;
        Ok(Self(self.0.flatten_to(dim).map_err(RbCandleErr::from)?))
    }

    fn flatten_all(&self) -> RbResult<Self> {
        Ok(Self(self.0.flatten_all().map_err(RbCandleErr::from)?))
    }

    fn t(&self) -> RbResult<Self> {
        Ok(Self(self.0.t().map_err(RbCandleErr::from)?))
    }

    fn contiguous(&self) -> RbResult<Self> {
        Ok(Self(self.0.contiguous().map_err(RbCandleErr::from)?))
    }

    fn is_contiguous(&self) -> bool {
        self.0.is_contiguous()
    }

    fn is_fortran_contiguous(&self) -> bool {
        self.0.is_fortran_contiguous()
    }

    fn detach(&self) -> RbResult<Self> {
        Ok(Self(self.0.detach().map_err(RbCandleErr::from)?))
    }

    fn copy(&self) -> RbResult<Self> {
        Ok(Self(self.0.copy().map_err(RbCandleErr::from)?))
    }

    fn to_dtype(&self, dtype: &RbDType) -> RbResult<Self> {
        Ok(Self(self.0.to_dtype(dtype.0).map_err(RbCandleErr::from)?))
    }
}

fn candle_utils(rb_candle: magnus::RModule) -> Result<(), Error> {
    let rb_utils = rb_candle.define_module("Utils")?;
    rb_utils.define_singleton_method(
        "cuda_is_available",
        function!(candle_core::utils::cuda_is_available, 0),
    )?;
    rb_utils.define_singleton_method(
        "get_num_threads",
        function!(candle_core::utils::get_num_threads, 0),
    )?;
    rb_utils.define_singleton_method(
        "has_accelerate",
        function!(candle_core::utils::has_accelerate, 0),
    )?;
    rb_utils.define_singleton_method("has_mkl", function!(candle_core::utils::has_mkl, 0))?;
    Ok(())
}

#[magnus::init]
fn init(ruby: &Ruby) -> RbResult<()> {
    let rb_candle = ruby.define_module("Candle")?;
    candle_utils(rb_candle)?;
    let rb_tensor = rb_candle.define_class("Tensor", Ruby::class_object(ruby))?;
    rb_tensor.define_singleton_method("new", function!(RbTensor::new, 1))?;
    rb_tensor.define_method("shape", method!(RbTensor::shape, 0))?;
    rb_tensor.define_method("stride", method!(RbTensor::stride, 0))?;
    rb_tensor.define_method("dtype", method!(RbTensor::dtype, 0))?;
    rb_tensor.define_method("rank", method!(RbTensor::rank, 0))?;
    rb_tensor.define_method("sin", method!(RbTensor::sin, 0))?;
    rb_tensor.define_method("cos", method!(RbTensor::cos, 0))?;
    rb_tensor.define_method("log", method!(RbTensor::log, 0))?;
    rb_tensor.define_method("sqr", method!(RbTensor::sqr, 0))?;
    rb_tensor.define_method("sqrt", method!(RbTensor::sqrt, 0))?;
    rb_tensor.define_method("recip", method!(RbTensor::recip, 0))?;
    rb_tensor.define_method("exp", method!(RbTensor::exp, 0))?;
    rb_tensor.define_method("powf", method!(RbTensor::powf, 1))?;
    rb_tensor.define_method("matmul", method!(RbTensor::matmul, 1))?;
    rb_tensor.define_method("where_cond", method!(RbTensor::where_cond, 2))?;
    rb_tensor.define_method("+", method!(RbTensor::__add__, 1))?;
    rb_tensor.define_method("*", method!(RbTensor::__mul__, 1))?;
    rb_tensor.define_method("-", method!(RbTensor::__sub__, 1))?;
    rb_tensor.define_method("reshape", method!(RbTensor::reshape, 1))?;
    rb_tensor.define_method("broadcast_as", method!(RbTensor::broadcast_as, 1))?;
    rb_tensor.define_method("broadcast_left", method!(RbTensor::broadcast_left, 1))?;
    rb_tensor.define_method("squeeze", method!(RbTensor::squeeze, 1))?;
    rb_tensor.define_method("unsqueeze", method!(RbTensor::unsqueeze, 1))?;
    rb_tensor.define_method("get", method!(RbTensor::get, 1))?;
    rb_tensor.define_method("transpose", method!(RbTensor::transpose, 2))?;
    rb_tensor.define_method("narrow", method!(RbTensor::narrow, 3))?;
    rb_tensor.define_method("argmax_keepdim", method!(RbTensor::argmax_keepdim, 1))?;
    rb_tensor.define_method("argmin_keepdim", method!(RbTensor::argmin_keepdim, 1))?;
    rb_tensor.define_method("max_keepdim", method!(RbTensor::max_keepdim, 1))?;
    rb_tensor.define_method("min_keepdim", method!(RbTensor::min_keepdim, 1))?;
    rb_tensor.define_method("sum_all", method!(RbTensor::sum_all, 0))?;
    rb_tensor.define_method("mean_all", method!(RbTensor::mean_all, 0))?;
    rb_tensor.define_method("flatten_from", method!(RbTensor::flatten_from, 1))?;
    rb_tensor.define_method("flatten_to", method!(RbTensor::flatten_to, 1))?;
    rb_tensor.define_method("flatten_all", method!(RbTensor::flatten_all, 0))?;
    rb_tensor.define_method("t", method!(RbTensor::t, 0))?;
    rb_tensor.define_method("contiguous", method!(RbTensor::contiguous, 0))?;
    rb_tensor.define_method("is_contiguous", method!(RbTensor::is_contiguous, 0))?;
    rb_tensor.define_method(
        "is_fortran_contiguous",
        method!(RbTensor::is_fortran_contiguous, 0),
    )?;
    rb_tensor.define_method("detach", method!(RbTensor::detach, 0))?;
    rb_tensor.define_method("copy", method!(RbTensor::copy, 0))?;
    rb_tensor.define_method("to_s", method!(RbTensor::__str__, 0))?;
    rb_tensor.define_method("inspect", method!(RbTensor::__repr__, 0))?;
    let rb_dtype = rb_candle.define_class("DType", Ruby::class_object(ruby))?;
    rb_dtype.define_method("to_s", method!(RbDType::__str__, 0))?;
    rb_dtype.define_method("inspect", method!(RbDType::__repr__, 0))?;
    Ok(())
}
