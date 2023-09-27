use magnus::{function, method, prelude::*, Error, Ruby};
use std::sync::Arc;

use half::{bf16, f16};

use ::candle_core::{quantized::QTensor, DType, Device, Tensor, WithDType};

type PyResult<T> = Result<T, Error>;

pub fn wrap_err(err: candle_core::Error) -> Error {
    Error::new(magnus::exception::runtime_error(), err.to_string())
}

// #[derive(Clone, Debug)]
// struct RbShape(Vec<usize>);

// impl magnus::TryConvert for RbShape {
//     fn try_convert(val: magnus::Value) -> PyResult<Self> {
//         let ary = magnus::RArray::try_convert(val)?;
//         let shape = ary
//             .each()
//             .map(|v| magnus::Integer::try_convert(v?).map(|v| v.to_usize().unwrap()))
//             .collect::<PyResult<Vec<_>>>()?;
//         Ok(Self(shape))
//     }
// }

// impl magnus::IntoValue for RbShape {
//     fn into_value_with(self, ruby: &Ruby) -> magnus::Value {
//         let ary = magnus::RArray::from_vec(self.0);
//         ary.into_value_with(ruby)
//     }
//}

#[derive(Clone, Debug)]
#[magnus::wrap(class = "Candle::Tensor", free_immediately, size)]
/// A `candle` tensor.
struct PyTensor(Tensor);

impl std::ops::Deref for PyTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[magnus::wrap(class = "Candle::DType", free_immediately, size)]
/// A `candle` dtype.
struct PyDType(DType);

impl PyDType {
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl PyDType {
    fn from_pyobject(dtype: magnus::Symbol) -> PyResult<Self> {
        let dtype = unsafe { dtype.to_s() }.unwrap().into_owned();
        use std::str::FromStr;
        let dtype = DType::from_str(&dtype).unwrap();
        Ok(Self(dtype))
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

    fn __repr__(&self) -> String {
        match self {
            Self::Cpu => "cpu".to_string(),
            Self::Cuda => "cuda".to_string(),
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl magnus::TryConvert for PyDevice {
    fn try_convert(val: magnus::Value) -> PyResult<Self> {
        let device = magnus::RString::try_convert(val)?;
        let device = unsafe { device.as_str() }.unwrap();
        let device = match device {
            "cpu" => PyDevice::Cpu,
            "cuda" => PyDevice::Cuda,
            _ => return Err(Error::new(magnus::exception::arg_error(), "invalid device")),
        };
        Ok(device)
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
impl PyTensor {
    fn new(array: magnus::RArray, dtype: Option<magnus::Symbol>) -> PyResult<Self> {
        let dtype = dtype
            .map(|dtype| PyDType::from_pyobject(dtype))
            .unwrap_or(Ok(PyDType(DType::F32)))?;
        // FIXME: Do not use `to_f64` here.
        let array = array
            .each()
            .map(|v| magnus::Float::try_convert(v?).map(|v| v.to_f64()))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self(
            Tensor::new(array.as_slice(), &Device::Cpu)
                .map_err(wrap_err)?
                .to_dtype(dtype.0)
                .map_err(wrap_err)?,
        ))
    }

    /// Gets the tensor's shape.
    /// &RETURNS&: Tuple[int]
    fn shape(&self) -> Vec<usize> {
        self.0.dims().to_vec()
    }

    /// Gets the tensor's strides.
    /// &RETURNS&: Tuple[int]
    fn stride(&self) -> Vec<usize> {
        self.0.stride().to_vec()
    }

    /// Gets the tensor's dtype.
    /// &RETURNS&: DType
    fn dtype(&self) -> PyDType {
        PyDType(self.0.dtype())
    }

    /// Gets the tensor's device.
    /// &RETURNS&: Device
    fn device(&self) -> PyDevice {
        PyDevice::from_device(self.0.device())
    }

    /// Gets the tensor's rank.
    /// &RETURNS&: int
    fn rank(&self) -> usize {
        self.0.rank()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Performs the `sin` operation on the tensor.
    /// &RETURNS&: Tensor
    fn sin(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.sin().map_err(wrap_err)?))
    }

    /// Performs the `cos` operation on the tensor.
    /// &RETURNS&: Tensor
    fn cos(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.cos().map_err(wrap_err)?))
    }

    /// Performs the `log` operation on the tensor.
    /// &RETURNS&: Tensor
    fn log(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.log().map_err(wrap_err)?))
    }

    /// Squares the tensor.
    /// &RETURNS&: Tensor
    fn sqr(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.sqr().map_err(wrap_err)?))
    }

    /// Calculates the square root of the tensor.
    /// &RETURNS&: Tensor
    fn sqrt(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.sqrt().map_err(wrap_err)?))
    }

    /// Get the `recip` of the tensor.
    /// &RETURNS&: Tensor
    fn recip(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.recip().map_err(wrap_err)?))
    }

    /// Performs the `exp` operation on the tensor.
    /// &RETURNS&: Tensor
    fn exp(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.exp().map_err(wrap_err)?))
    }

    /// Performs the `pow` operation on the tensor with the given exponent.
    /// &RETURNS&: Tensor
    fn powf(&self, p: f64) -> PyResult<Self> {
        Ok(PyTensor(self.0.powf(p).map_err(wrap_err)?))
    }

    /// Select values for the input tensor at the target indexes across the specified dimension.
    ///
    /// The `indexes` is argument is an int tensor with a single dimension.
    /// The output has the same number of dimension as the `self` input. The target dimension of
    /// the output has length the length of `indexes` and the values are taken from `self` using
    /// the index from `indexes`. Other dimensions have the same number of elements as the input
    /// tensor.
    /// &RETURNS&: Tensor
    fn index_select(&self, rhs: &Self, dim: i64) -> PyResult<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_err)?;
        Ok(PyTensor(self.0.index_select(rhs, dim).map_err(wrap_err)?))
    }

    /// Performs a matrix multiplication between the two tensors.
    /// &RETURNS&: Tensor
    fn matmul(&self, rhs: &Self) -> PyResult<Self> {
        Ok(PyTensor(self.0.matmul(rhs).map_err(wrap_err)?))
    }

    /// Adds the two tensors, while broadcasting the right-hand-side tensor to match the shape of the left-hand-side tensor.
    /// &RETURNS&: Tensor
    fn broadcast_add(&self, rhs: &Self) -> PyResult<Self> {
        Ok(PyTensor(self.0.broadcast_add(rhs).map_err(wrap_err)?))
    }

    /// Subtracts the two tensors, while broadcasting the right-hand-side tensor to match the shape of the left-hand-side tensor.
    /// &RETURNS&: Tensor
    fn broadcast_sub(&self, rhs: &Self) -> PyResult<Self> {
        Ok(PyTensor(self.0.broadcast_sub(rhs).map_err(wrap_err)?))
    }

    /// Multiplies the two tensors, while broadcasting the right-hand-side tensor to match the shape of the left-hand-side tensor.
    /// &RETURNS&: Tensor
    fn broadcast_mul(&self, rhs: &Self) -> PyResult<Self> {
        Ok(PyTensor(self.0.broadcast_mul(rhs).map_err(wrap_err)?))
    }

    /// Divides the two tensors, while broadcasting the right-hand-side tensor to match the shape of the left-hand-side tensor.
    /// &RETURNS&: Tensor
    fn broadcast_div(&self, rhs: &Self) -> PyResult<Self> {
        Ok(PyTensor(self.0.broadcast_div(rhs).map_err(wrap_err)?))
    }

    /// Returns a tensor with the same shape as the input tensor, the values are taken from
    /// `on_true` if the input tensor value is not zero, and `on_false` at the positions where the
    /// input tensor is equal to zero.
    /// &RETURNS&: Tensor
    fn where_cond(&self, on_true: &Self, on_false: &Self) -> PyResult<Self> {
        Ok(PyTensor(
            self.0.where_cond(on_true, on_false).map_err(wrap_err)?,
        ))
    }

    /// Add two tensors.
    /// &RETURNS&: Tensor
    fn __add__(&self, rhs: &PyTensor) -> PyResult<Self> {
        Ok(Self(self.0.add(&rhs.0).map_err(wrap_err)?))
    }

    /// Multiply two tensors.
    /// &RETURNS&: Tensor
    fn __mul__(&self, rhs: &PyTensor) -> PyResult<Self> {
        Ok(Self(self.0.mul(&rhs.0).map_err(wrap_err)?))
    }

    /// Subtract two tensors.
    /// &RETURNS&: Tensor
    fn __sub__(&self, rhs: &PyTensor) -> PyResult<Self> {
        Ok(Self(self.0.sub(&rhs.0).map_err(wrap_err)?))
    }

    /// Divide two tensors.
    /// &RETURNS&: Tensor
    fn __truediv__(&self, rhs: &PyTensor) -> PyResult<Self> {
        Ok(Self(self.0.div(&rhs.0).map_err(wrap_err)?))
    }

    /// Reshapes the tensor to the given shape.
    /// &RETURNS&: Tensor
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        Ok(PyTensor(self.0.reshape(shape).map_err(wrap_err)?))
    }

    /// Broadcasts the tensor to the given shape.
    /// &RETURNS&: Tensor
    fn broadcast_as(&self, shape: Vec<usize>) -> PyResult<Self> {
        Ok(PyTensor(self.0.broadcast_as(shape).map_err(wrap_err)?))
    }

    /// Broadcasts the tensor to the given shape, adding new dimensions on the left.
    /// &RETURNS&: Tensor
    fn broadcast_left(&self, shape: Vec<usize>) -> PyResult<Self> {
        Ok(PyTensor(self.0.broadcast_left(shape).map_err(wrap_err)?))
    }

    /// Creates a new tensor with the specified dimension removed if its size was one.
    /// &RETURNS&: Tensor
    fn squeeze(&self, dim: i64) -> PyResult<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_err)?;
        Ok(PyTensor(self.0.squeeze(dim).map_err(wrap_err)?))
    }

    /// Creates a new tensor with a dimension of size one inserted at the specified position.
    /// &RETURNS&: Tensor
    fn unsqueeze(&self, dim: usize) -> PyResult<Self> {
        Ok(PyTensor(self.0.unsqueeze(dim).map_err(wrap_err)?))
    }

    /// Gets the value at the specified index.
    /// &RETURNS&: Tensor
    fn get(&self, index: i64) -> PyResult<Self> {
        let index = actual_index(self, 0, index).map_err(wrap_err)?;
        Ok(PyTensor(self.0.get(index).map_err(wrap_err)?))
    }

    /// Returns a tensor that is a transposed version of the input, the given dimensions are swapped.
    /// &RETURNS&: Tensor
    fn transpose(&self, dim1: usize, dim2: usize) -> PyResult<Self> {
        Ok(PyTensor(self.0.transpose(dim1, dim2).map_err(wrap_err)?))
    }

    /// Returns a new tensor that is a narrowed version of the input, the dimension `dim`
    /// ranges from `start` to `start + len`.
    /// &RETURNS&: Tensor
    fn narrow(&self, dim: i64, start: i64, len: usize) -> PyResult<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_err)?;
        let start = actual_index(self, dim, start).map_err(wrap_err)?;
        Ok(PyTensor(self.0.narrow(dim, start, len).map_err(wrap_err)?))
    }

    /// Returns the indices of the maximum value(s) across the selected dimension.
    /// &RETURNS&: Tensor
    fn argmax_keepdim(&self, dim: i64) -> PyResult<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_err)?;
        Ok(PyTensor(self.0.argmax_keepdim(dim).map_err(wrap_err)?))
    }

    /// Returns the indices of the minimum value(s) across the selected dimension.
    /// &RETURNS&: Tensor
    fn argmin_keepdim(&self, dim: i64) -> PyResult<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_err)?;
        Ok(PyTensor(self.0.argmin_keepdim(dim).map_err(wrap_err)?))
    }

    /// Gathers the maximum value across the selected dimension.
    /// &RETURNS&: Tensor
    fn max_keepdim(&self, dim: i64) -> PyResult<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_err)?;
        Ok(PyTensor(self.0.max_keepdim(dim).map_err(wrap_err)?))
    }

    /// Gathers the minimum value across the selected dimension.
    /// &RETURNS&: Tensor
    fn min_keepdim(&self, dim: i64) -> PyResult<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_err)?;
        Ok(PyTensor(self.0.min_keepdim(dim).map_err(wrap_err)?))
    }

    // fn eq(&self, rhs: &Self) -> PyResult<Self> {
    //     Ok(PyTensor(self.0.eq(rhs).map_err(wrap_err)?))
    // }

    // fn ne(&self, rhs: &Self) -> PyResult<Self> {
    //     Ok(PyTensor(self.0.ne(rhs).map_err(wrap_err)?))
    // }

    // fn lt(&self, rhs: &Self) -> PyResult<Self> {
    //     Ok(PyTensor(self.0.lt(rhs).map_err(wrap_err)?))
    // }

    // fn gt(&self, rhs: &Self) -> PyResult<Self> {
    //     Ok(PyTensor(self.0.gt(rhs).map_err(wrap_err)?))
    // }

    // fn ge(&self, rhs: &Self) -> PyResult<Self> {
    //     Ok(PyTensor(self.0.ge(rhs).map_err(wrap_err)?))
    // }

    // fn le(&self, rhs: &Self) -> PyResult<Self> {
    //     Ok(PyTensor(self.0.le(rhs).map_err(wrap_err)?))
    // }

    /// Returns the sum of the tensor.
    /// &RETURNS&: Tensor
    fn sum_all(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.sum_all().map_err(wrap_err)?))
    }

    /// Returns the mean of the tensor.
    /// &RETURNS&: Tensor
    fn mean_all(&self) -> PyResult<Self> {
        let elements = self.0.elem_count();
        let sum = self.0.sum_all().map_err(wrap_err)?;
        let mean = (sum / elements as f64).map_err(wrap_err)?;
        Ok(PyTensor(mean))
    }

    /// Flattens the tensor on the dimension indexes from `dim` (inclusive) to the last dimension.
    /// &RETURNS&: Tensor
    fn flatten_from(&self, dim: i64) -> PyResult<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_err)?;
        Ok(PyTensor(self.0.flatten_from(dim).map_err(wrap_err)?))
    }

    ///Flattens the tensor on the dimension indexes from `0` to `dim` (inclusive).
    /// &RETURNS&: Tensor
    fn flatten_to(&self, dim: i64) -> PyResult<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_err)?;
        Ok(PyTensor(self.0.flatten_to(dim).map_err(wrap_err)?))
    }

    /// Flattens the tensor into a 1D tensor.
    /// &RETURNS&: Tensor
    fn flatten_all(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.flatten_all().map_err(wrap_err)?))
    }

    /// Transposes the tensor.
    /// &RETURNS&: Tensor
    fn t(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.t().map_err(wrap_err)?))
    }

    /// Makes the tensor contiguous in memory.
    /// &RETURNS&: Tensor
    fn contiguous(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.contiguous().map_err(wrap_err)?))
    }

    /// Returns true if the tensor is contiguous in C order.
    /// &RETURNS&: bool
    fn is_contiguous(&self) -> bool {
        self.0.is_contiguous()
    }

    /// Returns true if the tensor is contiguous in Fortran order.
    /// &RETURNS&: bool
    fn is_fortran_contiguous(&self) -> bool {
        self.0.is_fortran_contiguous()
    }

    /// Detach the tensor from the computation graph.
    /// &RETURNS&: Tensor
    fn detach(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.detach().map_err(wrap_err)?))
    }

    /// Returns a copy of the tensor.
    /// &RETURNS&: Tensor
    fn copy(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.copy().map_err(wrap_err)?))
    }

    /// Convert the tensor to a new dtype.
    /// &RETURNS&: Tensor
    fn to_dtype(&self, dtype: magnus::Symbol) -> PyResult<Self> {
        let dtype = PyDType::from_pyobject(dtype)?;
        Ok(PyTensor(self.0.to_dtype(dtype.0).map_err(wrap_err)?))
    }

    /// Move the tensor to a new device.
    /// &RETURNS&: Tensor
    fn to_device(&self, device: PyDevice) -> PyResult<Self> {
        let device = device.as_device()?;
        Ok(PyTensor(self.0.to_device(&device).map_err(wrap_err)?))
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

    /// Creates a new tensor with random values.
    /// &RETURNS&: Tensor
    fn rand(shape: Vec<usize>) -> PyResult<Self> {
        let device = PyDevice::Cpu.as_device()?;
        Ok(Self(
            Tensor::rand(0f32, 1f32, shape, &device).map_err(wrap_err)?,
        ))
    }

    /// Creates a new tensor with random values from a normal distribution.
    /// &RETURNS&: Tensor
    fn randn(shape: Vec<usize>) -> PyResult<Self> {
        let device = PyDevice::Cpu.as_device()?;
        Ok(Self(
            Tensor::randn(0f32, 1f32, shape, &device).map_err(wrap_err)?,
        ))
    }

    /// Creates a new tensor filled with ones.
    /// &RETURNS&: Tensor
    fn ones(shape: Vec<usize>) -> PyResult<Self> {
        let device = PyDevice::Cpu.as_device()?;
        Ok(Self(
            Tensor::ones(shape, DType::F32, &device).map_err(wrap_err)?,
        ))
    }
    /// Creates a new tensor filled with zeros.
    /// &RETURNS&: Tensor
    fn zeros(shape: Vec<usize>) -> PyResult<Self> {
        let device = PyDevice::Cpu.as_device()?;
        Ok(Self(
            Tensor::zeros(shape, DType::F32, &device).map_err(wrap_err)?,
        ))
    }
}

#[derive(Debug)]
#[magnus::wrap(class = "Candle::QTensor", free_immediately, size)]
/// A quantized tensor.
struct PyQTensor(Arc<QTensor>);

impl std::ops::Deref for PyQTensor {
    type Target = QTensor;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl PyQTensor {
    ///Gets the tensors quantized dtype.
    /// &RETURNS&: str
    fn ggml_dtype(&self) -> String {
        format!("{:?}", self.0.dtype())
    }

    ///Gets the rank of the tensor.
    /// &RETURNS&: int
    fn rank(&self) -> usize {
        self.0.rank()
    }

    ///Gets the shape of the tensor.
    /// &RETURNS&: Tuple[int]
    fn shape(&self) -> Vec<usize> {
        self.0.shape().dims().to_vec()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Dequantizes the tensor.
    /// &RETURNS&: Tensor  
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

/// Returns true if the 'cuda' backend is available.
/// &RETURNS&: bool
fn cuda_is_available() -> bool {
    candle_core::utils::cuda_is_available()
}

/// Returns true if candle was compiled with 'accelerate' support.
/// &RETURNS&: bool
fn has_accelerate() -> bool {
    candle_core::utils::has_accelerate()
}

/// Returns true if candle was compiled with MKL support.
/// &RETURNS&: bool
fn has_mkl() -> bool {
    candle_core::utils::has_mkl()
}

/// Returns the number of threads used by the candle.
/// &RETURNS&: int
fn get_num_threads() -> usize {
    candle_core::utils::get_num_threads()
}

fn candle_utils(rb_candle: magnus::RModule) -> Result<(), Error> {
    let rb_utils = rb_candle.define_module("Utils")?;
    rb_utils.define_singleton_method("cuda_is_available", function!(cuda_is_available, 0))?;
    rb_utils.define_singleton_method("get_num_threads", function!(get_num_threads, 0))?;
    rb_utils.define_singleton_method("has_accelerate", function!(has_accelerate, 0))?;
    rb_utils.define_singleton_method("has_mkl", function!(has_mkl, 0))?;
    Ok(())
}

/// Applies the Softmax function to a given tensor.#
/// &RETURNS&: Tensor
fn softmax(tensor: PyTensor, dim: i64) -> PyResult<PyTensor> {
    let dim = actual_dim(&tensor, dim).map_err(wrap_err)?;
    let sm = candle_nn::ops::softmax(&tensor.0, dim).map_err(wrap_err)?;
    Ok(PyTensor(sm))
}

/// Applies the Sigmoid Linear Unit (SiLU) function to a given tensor.
/// &RETURNS&: Tensor
fn silu(tensor: PyTensor) -> PyResult<PyTensor> {
    let s = candle_nn::ops::silu(&tensor.0).map_err(wrap_err)?;
    Ok(PyTensor(s))
}

#[magnus::init]
fn init(ruby: &Ruby) -> PyResult<()> {
    let rb_candle = ruby.define_module("Candle")?;
    candle_utils(rb_candle)?;
    let rb_tensor = rb_candle.define_class("Tensor", Ruby::class_object(ruby))?;
    rb_tensor.define_singleton_method("new", function!(PyTensor::new, 2))?;
    // rb_tensor.define_singleton_method("cat", function!(PyTensor::cat, 2))?;
    // rb_tensor.define_singleton_method("stack", function!(PyTensor::stack, 2))?;
    rb_tensor.define_singleton_method("rand", function!(PyTensor::rand, 1))?;
    rb_tensor.define_singleton_method("randn", function!(PyTensor::randn, 1))?;
    rb_tensor.define_singleton_method("ones", function!(PyTensor::ones, 1))?;
    rb_tensor.define_singleton_method("zeros", function!(PyTensor::zeros, 1))?;
    rb_tensor.define_method("shape", method!(PyTensor::shape, 0))?;
    rb_tensor.define_method("stride", method!(PyTensor::stride, 0))?;
    rb_tensor.define_method("dtype", method!(PyTensor::dtype, 0))?;
    rb_tensor.define_method("device", method!(PyTensor::device, 0))?;
    rb_tensor.define_method("rank", method!(PyTensor::rank, 0))?;
    rb_tensor.define_method("sin", method!(PyTensor::sin, 0))?;
    rb_tensor.define_method("cos", method!(PyTensor::cos, 0))?;
    rb_tensor.define_method("log", method!(PyTensor::log, 0))?;
    rb_tensor.define_method("sqr", method!(PyTensor::sqr, 0))?;
    rb_tensor.define_method("sqrt", method!(PyTensor::sqrt, 0))?;
    rb_tensor.define_method("recip", method!(PyTensor::recip, 0))?;
    rb_tensor.define_method("exp", method!(PyTensor::exp, 0))?;
    rb_tensor.define_method("powf", method!(PyTensor::powf, 1))?;
    rb_tensor.define_method("index_select", method!(PyTensor::index_select, 2))?;
    rb_tensor.define_method("matmul", method!(PyTensor::matmul, 1))?;
    rb_tensor.define_method("broadcast_add", method!(PyTensor::broadcast_add, 1))?;
    rb_tensor.define_method("broadcast_sub", method!(PyTensor::broadcast_sub, 1))?;
    rb_tensor.define_method("broadcast_mul", method!(PyTensor::broadcast_mul, 1))?;
    rb_tensor.define_method("broadcast_div", method!(PyTensor::broadcast_div, 1))?;
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
    // rb_tensor.define_method("eq", method!(PyTensor::eq, 1))?;
    // rb_tensor.define_method("ne", method!(PyTensor::ne, 1))?;
    // rb_tensor.define_method("lt", method!(PyTensor::lt, 1))?;
    // rb_tensor.define_method("gt", method!(PyTensor::gt, 1))?;
    // rb_tensor.define_method("ge", method!(PyTensor::ge, 1))?;
    // rb_tensor.define_method("le", method!(PyTensor::le, 1))?;
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
    rb_tensor.define_method("to_dtype", method!(PyTensor::to_dtype, 1))?;
    rb_tensor.define_method("to_device", method!(PyTensor::to_device, 1))?;
    rb_tensor.define_method("to_s", method!(PyTensor::__str__, 0))?;
    rb_tensor.define_method("inspect", method!(PyTensor::__repr__, 0))?;
    let rb_dtype = rb_candle.define_class("DType", Ruby::class_object(ruby))?;
    rb_dtype.define_method("to_s", method!(PyDType::__str__, 0))?;
    rb_dtype.define_method("inspect", method!(PyDType::__repr__, 0))?;
    let rb_device = rb_candle.define_class("Device", Ruby::class_object(ruby))?;
    rb_device.define_method("to_s", method!(PyDevice::__str__, 0))?;
    rb_device.define_method("inspect", method!(PyDevice::__repr__, 0))?;
    let rb_qtensor = rb_candle.define_class("QTensor", Ruby::class_object(ruby))?;
    rb_qtensor.define_method("ggml_dtype", method!(PyQTensor::ggml_dtype, 0))?;
    rb_qtensor.define_method("rank", method!(PyQTensor::rank, 0))?;
    rb_qtensor.define_method("shape", method!(PyQTensor::shape, 0))?;
    rb_qtensor.define_method("dequantize", method!(PyQTensor::dequantize, 0))?;
    Ok(())
}
