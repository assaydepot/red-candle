use magnus::{function, method, prelude::*, Error, Ruby};
use std::sync::Arc;

use ::candle_core::{quantized::QTensor, DType, Device, Tensor, WithDType};

type PyResult<T> = Result<T, Error>;

pub fn wrap_err(err: candle_core::Error) -> Error {
    Error::new(magnus::exception::runtime_error(), err.to_string())
}

#[derive(Clone, Debug)]
struct RbShape(Vec<usize>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[magnus::wrap(class = "Candle::DType", free_immediately, size)]
struct PyDType(DType);

impl PyDType {
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
enum PyDevice {
    Cpu,
    Cuda,
}

impl PyDevice {
    fn from_device(device: &Device) -> Self {
        match device {
            Device::Cpu => Self::Cpu,
            Device::Cuda(_) => Self::Cuda,
        }
    }

    fn as_device(&self) -> PyResult<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Cuda => {
                let mut device = CUDA_DEVICE.lock().unwrap();
                if let Some(device) = device.as_ref() {
                    return Ok(device.clone());
                };
                let d = Device::new_cuda(0).map_err(wrap_err)?;
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
struct PyTensor(Tensor);

impl PyTensor {
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

    fn dtype(&self) -> PyDType {
        PyDType(self.0.dtype())
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

    fn sin(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.sin().map_err(wrap_err)?))
    }

    fn cos(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.cos().map_err(wrap_err)?))
    }

    fn log(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.log().map_err(wrap_err)?))
    }

    fn sqr(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.sqr().map_err(wrap_err)?))
    }

    fn sqrt(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.sqrt().map_err(wrap_err)?))
    }

    fn recip(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.recip().map_err(wrap_err)?))
    }

    fn exp(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.exp().map_err(wrap_err)?))
    }

    fn powf(&self, n: f64) -> PyResult<Self> {
        Ok(PyTensor(self.0.powf(n).map_err(wrap_err)?))
    }

    fn matmul(&self, other: &PyTensor) -> PyResult<Self> {
        Ok(PyTensor(self.0.matmul(&other.0).map_err(wrap_err)?))
    }

    fn where_cond(&self, on_true: &PyTensor, on_false: &PyTensor) -> PyResult<Self> {
        Ok(Self(
            self.0
                .where_cond(&on_true.0, &on_false.0)
                .map_err(wrap_err)?,
        ))
    }

    fn __add__(&self, rhs: &PyTensor) -> PyResult<Self> {
        Ok(Self(self.0.add(&rhs.0).map_err(wrap_err)?))
    }

    fn __mul__(&self, rhs: &PyTensor) -> PyResult<Self> {
        Ok(Self(self.0.mul(&rhs.0).map_err(wrap_err)?))
    }

    fn __sub__(&self, rhs: &PyTensor) -> PyResult<Self> {
        Ok(Self(self.0.sub(&rhs.0).map_err(wrap_err)?))
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self(self.0.reshape(shape).map_err(wrap_err)?))
    }

    fn broadcast_as(&self, shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self(self.0.broadcast_as(shape).map_err(wrap_err)?))
    }

    fn broadcast_left(&self, shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self(self.0.broadcast_left(shape).map_err(wrap_err)?))
    }

    fn squeeze(&self, dim: usize) -> PyResult<Self> {
        let dim = actual_dim(&self.0, dim as i64).map_err(wrap_err)?;
        Ok(Self(self.0.squeeze(dim).map_err(wrap_err)?))
    }

    fn unsqueeze(&self, dim: usize) -> PyResult<Self> {
        Ok(Self(self.0.unsqueeze(dim).map_err(wrap_err)?))
    }

    fn get(&self, index: usize) -> PyResult<Self> {
        let index = actual_index(&self.0, 0, index as i64).map_err(wrap_err)?;
        Ok(Self(self.0.get(index).map_err(wrap_err)?))
    }

    fn transpose(&self, dim1: usize, dim2: usize) -> PyResult<Self> {
        Ok(Self(self.0.transpose(dim1, dim2).map_err(wrap_err)?))
    }

    fn narrow(&self, dim: usize, start: usize, len: usize) -> PyResult<Self> {
        let dim = actual_dim(&self.0, dim as i64).map_err(wrap_err)?;
        let start = actual_index(&self.0, dim, start as i64).map_err(wrap_err)?;
        Ok(Self(self.0.narrow(dim, start, len).map_err(wrap_err)?))
    }

    fn argmax_keepdim(&self, dim: i64) -> PyResult<Self> {
        let dim = actual_dim(&self.0, dim).map_err(wrap_err)?;
        Ok(Self(self.0.argmax_keepdim(dim).map_err(wrap_err)?))
    }

    fn argmin_keepdim(&self, dim: i64) -> PyResult<Self> {
        let dim = actual_dim(&self.0, dim).map_err(wrap_err)?;
        Ok(Self(self.0.argmin_keepdim(dim).map_err(wrap_err)?))
    }

    fn max_keepdim(&self, dim: i64) -> PyResult<Self> {
        let dim = actual_dim(&self.0, dim).map_err(wrap_err)?;
        Ok(Self(self.0.max_keepdim(dim).map_err(wrap_err)?))
    }

    fn min_keepdim(&self, dim: i64) -> PyResult<Self> {
        let dim = actual_dim(&self.0, dim).map_err(wrap_err)?;
        Ok(Self(self.0.min_keepdim(dim).map_err(wrap_err)?))
    }

    fn sum_all(&self) -> PyResult<Self> {
        Ok(Self(self.0.sum_all().map_err(wrap_err)?))
    }

    fn mean_all(&self) -> PyTensor {
        let elements = self.0.elem_count();
        let sum = self.0.sum_all().unwrap();
        let mean = (sum / elements as f64).unwrap();
        PyTensor(mean)
    }

    fn flatten_from(&self, dim: i64) -> PyResult<Self> {
        let dim = actual_dim(&self.0, dim).map_err(wrap_err)?;
        Ok(Self(self.0.flatten_from(dim).map_err(wrap_err)?))
    }

    fn flatten_to(&self, dim: i64) -> PyResult<Self> {
        let dim = actual_dim(&self.0, dim).map_err(wrap_err)?;
        Ok(Self(self.0.flatten_to(dim).map_err(wrap_err)?))
    }

    fn flatten_all(&self) -> PyResult<Self> {
        Ok(Self(self.0.flatten_all().map_err(wrap_err)?))
    }

    fn t(&self) -> PyResult<Self> {
        Ok(Self(self.0.t().map_err(wrap_err)?))
    }

    fn contiguous(&self) -> PyResult<Self> {
        Ok(Self(self.0.contiguous().map_err(wrap_err)?))
    }

    fn is_contiguous(&self) -> bool {
        self.0.is_contiguous()
    }

    fn is_fortran_contiguous(&self) -> bool {
        self.0.is_fortran_contiguous()
    }

    fn detach(&self) -> PyResult<Self> {
        Ok(Self(self.0.detach().map_err(wrap_err)?))
    }

    fn copy(&self) -> PyResult<Self> {
        Ok(Self(self.0.copy().map_err(wrap_err)?))
    }

    fn to_dtype(&self, dtype: &PyDType) -> PyResult<Self> {
        Ok(Self(self.0.to_dtype(dtype.0).map_err(wrap_err)?))
    }
}

impl PyTensor {
    // fn cat(tensors: Vec<PyTensor>, dim: i64) -> PyResult<PyTensor> {
    //     if tensors.is_empty() {
    //         return Err(Error::new(
    //             magnus::exception::arg_error(),
    //             "empty input to cat",
    //         ));
    //     }
    //     let dim = actual_dim(&tensors[0].0, dim).map_err(wrap_err)?;
    //     let tensors = tensors.into_iter().map(|t| t.0).collect::<Vec<_>>();
    //     let tensor = Tensor::cat(&tensors, dim).map_err(wrap_err)?;
    //     Ok(PyTensor(tensor))
    // }

    // fn stack(tensors: Vec<PyTensor>, dim: usize) -> PyResult<Self> {
    //     let tensors = tensors.into_iter().map(|t| t.0).collect::<Vec<_>>();
    //     let tensor = Tensor::stack(&tensors, dim).map_err(wrap_err)?;
    //     Ok(Self(tensor))
    // }

    fn rand(shape: Vec<usize>) -> PyResult<Self> {
        let device = PyDevice::Cpu.as_device()?;
        Ok(Self(
            Tensor::rand(0f32, 1f32, shape, &device).map_err(wrap_err)?,
        ))
    }

    fn randn(shape: Vec<usize>) -> PyResult<Self> {
        let device = PyDevice::Cpu.as_device()?;
        Ok(Self(
            Tensor::randn(0f32, 1f32, shape, &device).map_err(wrap_err)?,
        ))
    }

    fn ones(shape: Vec<usize>) -> PyResult<Self> {
        let device = PyDevice::Cpu.as_device()?;
        Ok(Self(
            Tensor::ones(shape, DType::F32, &device).map_err(wrap_err)?,
        ))
    }

    fn zeros(shape: Vec<usize>) -> PyResult<Self> {
        let device = PyDevice::Cpu.as_device()?;
        Ok(Self(
            Tensor::zeros(shape, DType::F32, &device).map_err(wrap_err)?,
        ))
    }
}

#[derive(Debug)]
struct PyQTensor(Arc<QTensor>);

impl std::ops::Deref for PyQTensor {
    type Target = QTensor;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl PyQTensor {
    fn ggml_dtype(&self) -> String {
        format!("{:?}", self.0.dtype())
    }

    fn rank(&self) -> usize {
        self.0.rank()
    }

    // fn shape(&self, py: Python<'_>) -> PyObject {
    //     PyTuple::new(py, self.0.shape().dims()).to_object(py)
    // }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn dequantize(&self) -> PyResult<PyTensor> {
        let tensor = self.0.dequantize(&Device::Cpu).map_err(wrap_err)?;
        Ok(PyTensor(tensor))
    }

    // fn matmul_t(&self, lhs: &PyTensor) -> PyResult<PyTensor> {
    //     let qmatmul = ::candle_core::quantized::QMatMul::from_arc(self.0.clone());
    //     let res = qmatmul.forward(lhs).map_err(wrap_err)?;
    //     Ok(PyTensor(res))
    // }
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
fn init(ruby: &Ruby) -> PyResult<()> {
    let rb_candle = ruby.define_module("Candle")?;
    candle_utils(rb_candle)?;
    let rb_tensor = rb_candle.define_class("Tensor", Ruby::class_object(ruby))?;
    rb_tensor.define_singleton_method("new", function!(PyTensor::new, 1))?;
    // rb_tensor.define_singleton_method("cat", function!(PyTensor::cat, 2))?;
    // rb_tensor.define_singleton_method("stack", function!(PyTensor::stack, 2))?;
    rb_tensor.define_singleton_method("rand", function!(PyTensor::rand, 1))?;
    rb_tensor.define_singleton_method("randn", function!(PyTensor::randn, 1))?;
    rb_tensor.define_singleton_method("ones", function!(PyTensor::ones, 1))?;
    rb_tensor.define_singleton_method("zeros", function!(PyTensor::zeros, 1))?;
    rb_tensor.define_method("shape", method!(PyTensor::shape, 0))?;
    rb_tensor.define_method("stride", method!(PyTensor::stride, 0))?;
    rb_tensor.define_method("dtype", method!(PyTensor::dtype, 0))?;
    rb_tensor.define_method("rank", method!(PyTensor::rank, 0))?;
    rb_tensor.define_method("sin", method!(PyTensor::sin, 0))?;
    rb_tensor.define_method("cos", method!(PyTensor::cos, 0))?;
    rb_tensor.define_method("log", method!(PyTensor::log, 0))?;
    rb_tensor.define_method("sqr", method!(PyTensor::sqr, 0))?;
    rb_tensor.define_method("sqrt", method!(PyTensor::sqrt, 0))?;
    rb_tensor.define_method("recip", method!(PyTensor::recip, 0))?;
    rb_tensor.define_method("exp", method!(PyTensor::exp, 0))?;
    rb_tensor.define_method("powf", method!(PyTensor::powf, 1))?;
    rb_tensor.define_method("matmul", method!(PyTensor::matmul, 1))?;
    rb_tensor.define_method("where_cond", method!(PyTensor::where_cond, 2))?;
    rb_tensor.define_method("+", method!(PyTensor::__add__, 1))?;
    rb_tensor.define_method("*", method!(PyTensor::__mul__, 1))?;
    rb_tensor.define_method("-", method!(PyTensor::__sub__, 1))?;
    rb_tensor.define_method("reshape", method!(PyTensor::reshape, 1))?;
    rb_tensor.define_method("broadcast_as", method!(PyTensor::broadcast_as, 1))?;
    rb_tensor.define_method("broadcast_left", method!(PyTensor::broadcast_left, 1))?;
    rb_tensor.define_method("squeeze", method!(PyTensor::squeeze, 1))?;
    rb_tensor.define_method("unsqueeze", method!(PyTensor::unsqueeze, 1))?;
    rb_tensor.define_method("get", method!(PyTensor::get, 1))?;
    rb_tensor.define_method("transpose", method!(PyTensor::transpose, 2))?;
    rb_tensor.define_method("narrow", method!(PyTensor::narrow, 3))?;
    rb_tensor.define_method("argmax_keepdim", method!(PyTensor::argmax_keepdim, 1))?;
    rb_tensor.define_method("argmin_keepdim", method!(PyTensor::argmin_keepdim, 1))?;
    rb_tensor.define_method("max_keepdim", method!(PyTensor::max_keepdim, 1))?;
    rb_tensor.define_method("min_keepdim", method!(PyTensor::min_keepdim, 1))?;
    rb_tensor.define_method("sum_all", method!(PyTensor::sum_all, 0))?;
    rb_tensor.define_method("mean_all", method!(PyTensor::mean_all, 0))?;
    rb_tensor.define_method("flatten_from", method!(PyTensor::flatten_from, 1))?;
    rb_tensor.define_method("flatten_to", method!(PyTensor::flatten_to, 1))?;
    rb_tensor.define_method("flatten_all", method!(PyTensor::flatten_all, 0))?;
    rb_tensor.define_method("t", method!(PyTensor::t, 0))?;
    rb_tensor.define_method("contiguous", method!(PyTensor::contiguous, 0))?;
    rb_tensor.define_method("is_contiguous", method!(PyTensor::is_contiguous, 0))?;
    rb_tensor.define_method(
        "is_fortran_contiguous",
        method!(PyTensor::is_fortran_contiguous, 0),
    )?;
    rb_tensor.define_method("detach", method!(PyTensor::detach, 0))?;
    rb_tensor.define_method("copy", method!(PyTensor::copy, 0))?;
    rb_tensor.define_method("to_s", method!(PyTensor::__str__, 0))?;
    rb_tensor.define_method("inspect", method!(PyTensor::__repr__, 0))?;
    let rb_dtype = rb_candle.define_class("DType", Ruby::class_object(ruby))?;
    rb_dtype.define_method("to_s", method!(PyDType::__str__, 0))?;
    rb_dtype.define_method("inspect", method!(PyDType::__repr__, 0))?;
    Ok(())
}
