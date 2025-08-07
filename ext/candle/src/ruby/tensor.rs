use magnus::prelude::*;
use magnus::{function, method, class, RModule, Module, Object};

use crate::ruby::{
    errors::wrap_candle_err,
    utils::{actual_dim, actual_index},
};
use crate::ruby::{DType, Device, Result};
use ::candle_core::{DType as CoreDType, Tensor as CoreTensor, Device as CoreDevice};

#[derive(Clone, Debug)]
#[magnus::wrap(class = "Candle::Tensor", free_immediately, size)]
/// A `candle` tensor.
pub struct Tensor(pub CoreTensor);

impl std::ops::Deref for Tensor {
    type Target = CoreTensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Tensor {
    pub fn new(array: magnus::RArray, dtype: Option<magnus::Symbol>, device: Option<Device>) -> Result<Self> {
        let dtype = dtype
            .map(|dtype| DType::from_rbobject(dtype))
            .unwrap_or(Ok(DType(CoreDType::F32)))?;
        let device = device.unwrap_or(Device::best()).as_device()?;
        
        // Create tensor based on target dtype to avoid conversion issues on Metal
        let tensor = match dtype.0 {
            CoreDType::F32 => {
                // Convert to f32 directly to avoid F64->F32 conversion on Metal
                let array: Vec<f32> = array
                    .into_iter()
                    .map(|v| magnus::Float::try_convert(v).map(|v| v.to_f64() as f32))
                    .collect::<Result<Vec<_>>>()?;
                let len = array.len();
                CoreTensor::from_vec(array, len, &device).map_err(wrap_candle_err)?
            }
            CoreDType::F64 => {
                let array: Vec<f64> = array
                    .into_iter()
                    .map(|v| magnus::Float::try_convert(v).map(|v| v.to_f64()))
                    .collect::<Result<Vec<_>>>()?;
                let len = array.len();
                CoreTensor::from_vec(array, len, &device).map_err(wrap_candle_err)?
            }
            CoreDType::I64 => {
                // Convert to i64 directly to avoid conversion issues on Metal
                let array: Vec<i64> = array
                    .into_iter()
                    .map(|v| {
                        // Try integer first, then float
                        if let Ok(i) = <i64>::try_convert(v) {
                            Ok(i)
                        } else if let Ok(f) = magnus::Float::try_convert(v) {
                            Ok(f.to_f64() as i64)
                        } else {
                            Err(magnus::Error::new(
                                magnus::exception::type_error(),
                                "Cannot convert to i64"
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                let len = array.len();
                CoreTensor::from_vec(array, len, &device).map_err(wrap_candle_err)?
            }
            _ => {
                // For other dtypes, create on CPU first if on Metal, then convert
                let cpu_device = CoreDevice::Cpu;
                let target_device = if matches!(device.location(), candle_core::DeviceLocation::Metal { .. }) {
                    &cpu_device
                } else {
                    &device
                };
                
                let array: Vec<f64> = array
                    .into_iter()
                    .map(|v| magnus::Float::try_convert(v).map(|v| v.to_f64()))
                    .collect::<Result<Vec<_>>>()?;
                let tensor = CoreTensor::new(array.as_slice(), target_device)
                    .map_err(wrap_candle_err)?
                    .to_dtype(dtype.0)
                    .map_err(wrap_candle_err)?;
                
                // Move to target device if needed
                if !std::ptr::eq(target_device, &device) {
                    tensor.to_device(&device).map_err(wrap_candle_err)?
                } else {
                    tensor
                }
            }
        };
        
        Ok(Self(tensor))
    }

    pub fn values(&self) -> Result<Vec<f64>> {
        // Handle dtype conversion for Metal which doesn't support many conversions
        let tensor = match (self.0.device().location(), self.0.dtype()) {
            (candle_core::DeviceLocation::Metal { .. }, dtype) if dtype != CoreDType::F64 => {
                // Move to CPU first to avoid Metal conversion limitations
                self.0
                    .to_device(&CoreDevice::Cpu)
                    .map_err(wrap_candle_err)?
                    .to_dtype(CoreDType::F64)
                    .map_err(wrap_candle_err)?
            }
            _ => {
                // Direct conversion for CPU or when already F64
                self.0
                    .to_dtype(CoreDType::F64)
                    .map_err(wrap_candle_err)?
            }
        };
        
        let values = tensor
            .flatten_all()
            .map_err(wrap_candle_err)?
            .to_vec1()
            .map_err(wrap_candle_err)?;
        Ok(values)
    }

    /// Get values as f32 without dtype conversion
    pub fn values_f32(&self) -> Result<Vec<f32>> {
        match self.0.dtype() {
            CoreDType::F32 => {
                let values = self
                    .0
                    .flatten_all()
                    .map_err(wrap_candle_err)?
                    .to_vec1()
                    .map_err(wrap_candle_err)?;
                Ok(values)
            }
            _ => Err(magnus::Error::new(
                magnus::exception::runtime_error(),
                "Tensor must be F32 dtype for values_f32",
            )),
        }
    }

    /// Get a single scalar value from a rank-0 tensor
    pub fn item(&self) -> Result<f64> {
        if self.0.rank() != 0 {
            return Err(magnus::Error::new(
                magnus::exception::runtime_error(),
                format!("item() can only be called on scalar tensors (rank 0), but tensor has rank {}", self.0.rank()),
            ));
        }
        
        // Try to get the value based on dtype
        match self.0.dtype() {
            CoreDType::F32 => {
                let val: f32 = self.0.to_vec0().map_err(wrap_candle_err)?;
                Ok(val as f64)
            }
            CoreDType::F64 => {
                let val: f64 = self.0.to_vec0().map_err(wrap_candle_err)?;
                Ok(val)
            }
            _ => {
                // For other dtypes, convert to F64 first
                // Handle Metal conversion limitations
                let tensor = match self.0.device().location() {
                    candle_core::DeviceLocation::Metal { .. } => {
                        // Move to CPU first to avoid Metal conversion limitations
                        self.0
                            .to_device(&CoreDevice::Cpu)
                            .map_err(wrap_candle_err)?
                            .to_dtype(CoreDType::F64)
                            .map_err(wrap_candle_err)?
                    }
                    _ => {
                        self.0
                            .to_dtype(CoreDType::F64)
                            .map_err(wrap_candle_err)?
                    }
                };
                let val: f64 = tensor.to_vec0().map_err(wrap_candle_err)?;
                Ok(val)
            }
        }
    }

    /// Gets the tensor's shape.
    /// &RETURNS&: Tuple[int]
    pub fn shape(&self) -> Vec<usize> {
        self.0.dims().to_vec()
    }

    /// Gets the tensor's strides.
    /// &RETURNS&: Tuple[int]
    pub fn stride(&self) -> Vec<usize> {
        self.0.stride().to_vec()
    }

    /// Gets the tensor's dtype.
    /// &RETURNS&: DType
    pub fn dtype(&self) -> DType {
        DType(self.0.dtype())
    }

    /// Gets the tensor's device.
    /// &RETURNS&: Device
    pub fn device(&self) -> Device {
        Device::from_device(self.0.device())
    }

    /// Gets the tensor's rank.
    /// &RETURNS&: int
    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    /// The number of elements stored in this tensor.
    /// &RETURNS&: int
    pub fn elem_count(&self) -> usize {
        self.0.elem_count()
    }

    pub fn __repr__(&self) -> String {
        format!("{}", self.0)
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Performs the `sin` operation on the tensor.
    /// &RETURNS&: Tensor
    pub fn sin(&self) -> Result<Self> {
        Ok(Tensor(self.0.sin().map_err(wrap_candle_err)?))
    }

    /// Performs the `cos` operation on the tensor.
    /// &RETURNS&: Tensor
    pub fn cos(&self) -> Result<Self> {
        Ok(Tensor(self.0.cos().map_err(wrap_candle_err)?))
    }

    /// Performs the `log` operation on the tensor.
    /// &RETURNS&: Tensor
    pub fn log(&self) -> Result<Self> {
        Ok(Tensor(self.0.log().map_err(wrap_candle_err)?))
    }

    /// Squares the tensor.
    /// &RETURNS&: Tensor
    pub fn sqr(&self) -> Result<Self> {
        Ok(Tensor(self.0.sqr().map_err(wrap_candle_err)?))
    }

    /// Returns the mean along the specified axis.
    /// @param axis [Integer, optional] The axis to reduce over (default: 0)
    /// @return [Candle::Tensor]
    pub fn mean(&self, axis: Option<i64>) -> Result<Self> {
        let axis = axis.unwrap_or(0) as usize;
        Ok(Tensor(self.0.mean(axis).map_err(wrap_candle_err)?))
    }

    /// Returns the sum along the specified axis.
    /// @param axis [Integer, optional] The axis to reduce over (default: 0)
    /// @return [Candle::Tensor]
    pub fn sum(&self, axis: Option<i64>) -> Result<Self> {
        let axis = axis.unwrap_or(0) as usize;
        Ok(Tensor(self.0.sum(axis).map_err(wrap_candle_err)?))
    }

    /// Calculates the square root of the tensor.
    /// &RETURNS&: Tensor
    pub fn sqrt(&self) -> Result<Self> {
        Ok(Tensor(self.0.sqrt().map_err(wrap_candle_err)?))
    }

    /// Get the `recip` of the tensor.
    /// &RETURNS&: Tensor
    pub fn recip(&self) -> Result<Self> {
        Ok(Tensor(self.0.recip().map_err(wrap_candle_err)?))
    }

    /// Performs the `exp` operation on the tensor.
    /// &RETURNS&: Tensor
    pub fn exp(&self) -> Result<Self> {
        Ok(Tensor(self.0.exp().map_err(wrap_candle_err)?))
    }

    /// Performs the `pow` operation on the tensor with the given exponent.
    /// &RETURNS&: Tensor
    pub fn powf(&self, p: f64) -> Result<Self> {
        Ok(Tensor(self.0.powf(p).map_err(wrap_candle_err)?))
    }

    /// Select values for the input tensor at the target indexes across the specified dimension.
    ///
    /// The `indexes` is argument is an int tensor with a single dimension.
    /// The output has the same number of dimension as the `self` input. The target dimension of
    /// the output has length the length of `indexes` and the values are taken from `self` using
    /// the index from `indexes`. Other dimensions have the same number of elements as the input
    /// tensor.
    /// &RETURNS&: Tensor
    pub fn index_select(&self, rhs: &Self, dim: i64) -> Result<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_candle_err)?;
        Ok(Tensor(
            self.0.index_select(rhs, dim).map_err(wrap_candle_err)?,
        ))
    }

    /// Performs a matrix multiplication between the two tensors.
    /// &RETURNS&: Tensor
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        Ok(Tensor(self.0.matmul(rhs).map_err(wrap_candle_err)?))
    }

    /// Adds the two tensors, while broadcasting the right-hand-side tensor to match the shape of the left-hand-side tensor.
    /// &RETURNS&: Tensor
    pub fn broadcast_add(&self, rhs: &Self) -> Result<Self> {
        Ok(Tensor(
            self.0.broadcast_add(rhs).map_err(wrap_candle_err)?,
        ))
    }

    /// Subtracts the two tensors, while broadcasting the right-hand-side tensor to match the shape of the left-hand-side tensor.
    /// &RETURNS&: Tensor
    pub fn broadcast_sub(&self, rhs: &Self) -> Result<Self> {
        Ok(Tensor(
            self.0.broadcast_sub(rhs).map_err(wrap_candle_err)?,
        ))
    }

    /// Multiplies the two tensors, while broadcasting the right-hand-side tensor to match the shape of the left-hand-side tensor.
    /// &RETURNS&: Tensor
    pub fn broadcast_mul(&self, rhs: &Self) -> Result<Self> {
        Ok(Tensor(
            self.0.broadcast_mul(rhs).map_err(wrap_candle_err)?,
        ))
    }

    /// Divides the two tensors, while broadcasting the right-hand-side tensor to match the shape of the left-hand-side tensor.
    /// &RETURNS&: Tensor
    pub fn broadcast_div(&self, rhs: &Self) -> Result<Self> {
        Ok(Tensor(
            self.0.broadcast_div(rhs).map_err(wrap_candle_err)?,
        ))
    }

    /// Returns a tensor with the same shape as the input tensor, the values are taken from
    /// `on_true` if the input tensor value is not zero, and `on_false` at the positions where the
    /// input tensor is equal to zero.
    /// &RETURNS&: Tensor
    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> Result<Self> {
        Ok(Tensor(
            self.0
                .where_cond(on_true, on_false)
                .map_err(wrap_candle_err)?,
        ))
    }

    /// Add two tensors.
    /// &RETURNS&: Tensor
    pub fn __add__(&self, rhs: &Tensor) -> Result<Self> {
        Ok(Self(self.0.add(&rhs.0).map_err(wrap_candle_err)?))
    }

    /// Multiply two tensors.
    /// &RETURNS&: Tensor
    pub fn __mul__(&self, rhs: &Tensor) -> Result<Self> {
        Ok(Self(self.0.mul(&rhs.0).map_err(wrap_candle_err)?))
    }

    /// Subtract two tensors.
    /// &RETURNS&: Tensor
    pub fn __sub__(&self, rhs: &Tensor) -> Result<Self> {
        Ok(Self(self.0.sub(&rhs.0).map_err(wrap_candle_err)?))
    }

    /// Divide two tensors.
    /// &RETURNS&: Tensor
    /// Divides this tensor by another tensor or a scalar (Float/Integer).
    /// @param rhs [Candle::Tensor, Float, or Integer]
    /// @return [Candle::Tensor]
    pub fn __truediv__(&self, rhs: magnus::Value) -> Result<Self> {
        use magnus::TryConvert;
        if let Ok(tensor) = <&Tensor>::try_convert(rhs) {
            Ok(Self(self.0.broadcast_div(&tensor.0).map_err(wrap_candle_err)?))
        } else if let Ok(f) = <f64>::try_convert(rhs) {
            let scalar = CoreTensor::from_vec(vec![f as f32], (1,), &self.0.device()).map_err(wrap_candle_err)?;
            Ok(Self(self.0.broadcast_div(&scalar).map_err(wrap_candle_err)?))
        } else if let Ok(i) = <i64>::try_convert(rhs) {
            let scalar = CoreTensor::from_vec(vec![i as f32], (1,), &self.0.device()).map_err(wrap_candle_err)?;
            Ok(Self(self.0.broadcast_div(&scalar).map_err(wrap_candle_err)?))
        } else {
            Err(magnus::Error::new(magnus::exception::type_error(), "Right-hand side must be a Candle::Tensor, Float, or Integer"))
        }
    }

    /// Reshapes the tensor to the given shape.
    /// &RETURNS&: Tensor
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self> {
        Ok(Tensor(self.0.reshape(shape).map_err(wrap_candle_err)?))
    }

    /// Broadcasts the tensor to the given shape.
    /// &RETURNS&: Tensor
    pub fn broadcast_as(&self, shape: Vec<usize>) -> Result<Self> {
        Ok(Tensor(
            self.0.broadcast_as(shape).map_err(wrap_candle_err)?,
        ))
    }

    /// Broadcasts the tensor to the given shape, adding new dimensions on the left.
    /// &RETURNS&: Tensor
    pub fn broadcast_left(&self, shape: Vec<usize>) -> Result<Self> {
        Ok(Tensor(
            self.0.broadcast_left(shape).map_err(wrap_candle_err)?,
        ))
    }

    /// Creates a new tensor with the specified dimension removed if its size was one.
    /// &RETURNS&: Tensor
    pub fn squeeze(&self, dim: i64) -> Result<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_candle_err)?;
        Ok(Tensor(self.0.squeeze(dim).map_err(wrap_candle_err)?))
    }

    /// Creates a new tensor with a dimension of size one inserted at the specified position.
    /// &RETURNS&: Tensor
    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        Ok(Tensor(self.0.unsqueeze(dim).map_err(wrap_candle_err)?))
    }

    /// Gets the value at the specified index.
    /// &RETURNS&: Tensor
    pub fn get(&self, index: i64) -> Result<Self> {
        let index = actual_index(self, 0, index).map_err(wrap_candle_err)?;
        Ok(Tensor(self.0.get(index).map_err(wrap_candle_err)?))
    }

    /// Returns a tensor that is a transposed version of the input, the given dimensions are swapped.
    /// &RETURNS&: Tensor
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        Ok(Tensor(
            self.0.transpose(dim1, dim2).map_err(wrap_candle_err)?,
        ))
    }

    /// Returns a new tensor that is a narrowed version of the input, the dimension `dim`
    /// ranges from `start` to `start + len`.
    /// &RETURNS&: Tensor
    pub fn narrow(&self, dim: i64, start: i64, len: usize) -> Result<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_candle_err)?;
        let start = actual_index(self, dim, start).map_err(wrap_candle_err)?;
        Ok(Tensor(
            self.0.narrow(dim, start, len).map_err(wrap_candle_err)?,
        ))
    }

    /// Returns the indices of the maximum value(s) across the selected dimension.
    /// &RETURNS&: Tensor
    pub fn argmax_keepdim(&self, dim: i64) -> Result<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_candle_err)?;
        Ok(Tensor(
            self.0.argmax_keepdim(dim).map_err(wrap_candle_err)?,
        ))
    }

    /// Returns the indices of the minimum value(s) across the selected dimension.
    /// &RETURNS&: Tensor
    pub fn argmin_keepdim(&self, dim: i64) -> Result<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_candle_err)?;
        Ok(Tensor(
            self.0.argmin_keepdim(dim).map_err(wrap_candle_err)?,
        ))
    }

    /// Gathers the maximum value across the selected dimension.
    /// &RETURNS&: Tensor
    pub fn max_keepdim(&self, dim: i64) -> Result<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_candle_err)?;
        Ok(Tensor(self.0.max_keepdim(dim).map_err(wrap_candle_err)?))
    }

    /// Gathers the minimum value across the selected dimension.
    /// &RETURNS&: Tensor
    pub fn min_keepdim(&self, dim: i64) -> Result<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_candle_err)?;
        Ok(Tensor(self.0.min_keepdim(dim).map_err(wrap_candle_err)?))
    }

    // fn eq(&self, rhs: &Self) -> Result<Self> {
    //     Ok(Tensor(self.0.eq(rhs).map_err(wrap_candle_err)?))
    // }

    // fn ne(&self, rhs: &Self) -> Result<Self> {
    //     Ok(Tensor(self.0.ne(rhs).map_err(wrap_candle_err)?))
    // }

    // fn lt(&self, rhs: &Self) -> Result<Self> {
    //     Ok(Tensor(self.0.lt(rhs).map_err(wrap_candle_err)?))
    // }

    // fn gt(&self, rhs: &Self) -> Result<Self> {
    //     Ok(Tensor(self.0.gt(rhs).map_err(wrap_candle_err)?))
    // }

    // fn ge(&self, rhs: &Self) -> Result<Self> {
    //     Ok(Tensor(self.0.ge(rhs).map_err(wrap_candle_err)?))
    // }

    // fn le(&self, rhs: &Self) -> Result<Self> {
    //     Ok(Tensor(self.0.le(rhs).map_err(wrap_candle_err)?))
    // }

    /// Returns the sum of the tensor.
    /// &RETURNS&: Tensor
    pub fn sum_all(&self) -> Result<Self> {
        Ok(Tensor(self.0.sum_all().map_err(wrap_candle_err)?))
    }

    /// Returns the mean of the tensor.
    /// &RETURNS&: Tensor
    pub fn mean_all(&self) -> Result<Self> {
        let elements = self.0.elem_count();
        let sum = self.0.sum_all().map_err(wrap_candle_err)?;
        let mean = (sum / elements as f64).map_err(wrap_candle_err)?;
        Ok(Tensor(mean))
    }

    /// Flattens the tensor on the dimension indexes from `dim` (inclusive) to the last dimension.
    /// &RETURNS&: Tensor
    pub fn flatten_from(&self, dim: i64) -> Result<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_candle_err)?;
        Ok(Tensor(self.0.flatten_from(dim).map_err(wrap_candle_err)?))
    }

    ///Flattens the tensor on the dimension indexes from `0` to `dim` (inclusive).
    /// &RETURNS&: Tensor
    pub fn flatten_to(&self, dim: i64) -> Result<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_candle_err)?;
        Ok(Tensor(self.0.flatten_to(dim).map_err(wrap_candle_err)?))
    }

    /// Flattens the tensor into a 1D tensor.
    /// &RETURNS&: Tensor
    pub fn flatten_all(&self) -> Result<Self> {
        Ok(Tensor(self.0.flatten_all().map_err(wrap_candle_err)?))
    }

    /// Transposes the tensor.
    /// &RETURNS&: Tensor
    pub fn t(&self) -> Result<Self> {
        Ok(Tensor(self.0.t().map_err(wrap_candle_err)?))
    }

    /// Makes the tensor contiguous in memory.
    /// &RETURNS&: Tensor
    pub fn contiguous(&self) -> Result<Self> {
        Ok(Tensor(self.0.contiguous().map_err(wrap_candle_err)?))
    }

    /// Returns true if the tensor is contiguous in C order.
    /// &RETURNS&: bool
    pub fn is_contiguous(&self) -> bool {
        self.0.is_contiguous()
    }

    /// Returns true if the tensor is contiguous in Fortran order.
    /// &RETURNS&: bool
    pub fn is_fortran_contiguous(&self) -> bool {
        self.0.is_fortran_contiguous()
    }

    /// Detach the tensor from the computation graph.
    /// &RETURNS&: Tensor
    pub fn detach(&self) -> Result<Self> {
        Ok(Tensor(self.0.detach()))
    }

    /// Returns a copy of the tensor.
    /// &RETURNS&: Tensor
    pub fn copy(&self) -> Result<Self> {
        Ok(Tensor(self.0.copy().map_err(wrap_candle_err)?))
    }

    /// Convert the tensor to a new dtype.
    /// &RETURNS&: Tensor
    pub fn to_dtype(&self, dtype: magnus::Symbol) -> Result<Self> {
        let dtype = DType::from_rbobject(dtype)?;
        Ok(Tensor(self.0.to_dtype(dtype.0).map_err(wrap_candle_err)?))
    }

    /// Move the tensor to a new device.
    /// &RETURNS&: Tensor
    pub fn to_device(&self, device: Device) -> Result<Self> {
        let device = device.as_device()?;
        Ok(Tensor(
            self.0.to_device(&device).map_err(wrap_candle_err)?,
        ))
    }
}

impl Tensor {
    // fn cat(tensors: Vec<Tensor>, dim: i64) -> Result<Tensor> {
    //     if tensors.is_empty() {
    //         return Err(Error::new(
    //             magnus::exception::arg_error(),
    //             "empty input to cat",
    //         ));
    //     }
    //     let dim = actual_dim(&tensors[0].0, dim).map_err(wrap_candle_err)?;
    //     let tensors = tensors.into_iter().map(|t| t.0).collect::<Vec<_>>();
    //     let tensor = CoreTensor::cat(&tensors, dim).map_err(wrap_candle_err)?;
    //     Ok(Tensor(tensor))
    // }

    // fn stack(tensors: Vec<Tensor>, dim: usize) -> Result<Self> {
    //     let tensors = tensors.into_iter().map(|t| t.0).collect::<Vec<_>>();
    //     let tensor = CoreTensor::stack(&tensors, dim).map_err(wrap_candle_err)?;
    //     Ok(Self(tensor))
    // }

    /// Creates a new tensor with random values.
    /// &RETURNS&: Tensor
    pub fn rand(shape: Vec<usize>, device: Option<Device>) -> Result<Self> {
        let device = device.unwrap_or(Device::best()).as_device()?;
        Ok(Self(
            CoreTensor::rand(0f32, 1f32, shape, &device).map_err(wrap_candle_err)?,
        ))
    }

    /// Creates a new tensor with random values from a normal distribution.
    /// &RETURNS&: Tensor
    pub fn randn(shape: Vec<usize>, device: Option<Device>) -> Result<Self> {
        let device = device.unwrap_or(Device::best()).as_device()?;
        Ok(Self(
            CoreTensor::randn(0f32, 1f32, shape, &device).map_err(wrap_candle_err)?,
        ))
    }

    /// Creates a new tensor filled with ones.
    /// &RETURNS&: Tensor
    pub fn ones(shape: Vec<usize>, device: Option<Device>) -> Result<Self> {
        let device = device.unwrap_or(Device::best()).as_device()?;
        Ok(Self(
            CoreTensor::ones(shape, CoreDType::F32, &device).map_err(wrap_candle_err)?,
        ))
    }
    /// Creates a new tensor filled with zeros.
    /// &RETURNS&: Tensor
    pub fn zeros(shape: Vec<usize>, device: Option<Device>) -> Result<Self> {
        let device = device.unwrap_or(Device::best()).as_device()?;
        Ok(Self(
            CoreTensor::zeros(shape, CoreDType::F32, &device).map_err(wrap_candle_err)?,
        ))
    }
}

pub fn init(rb_candle: RModule) -> Result<()> {
    let rb_tensor = rb_candle.define_class("Tensor", class::object())?;
    rb_tensor.define_singleton_method("new", function!(Tensor::new, 3))?;
    // rb_tensor.define_singleton_method("cat", function!(Tensor::cat, 2))?;
    // rb_tensor.define_singleton_method("stack", function!(Tensor::stack, 2))?;
    rb_tensor.define_singleton_method("rand", function!(Tensor::rand, 2))?;
    rb_tensor.define_singleton_method("randn", function!(Tensor::randn, 2))?;
    rb_tensor.define_singleton_method("ones", function!(Tensor::ones, 2))?;
    rb_tensor.define_singleton_method("zeros", function!(Tensor::zeros, 2))?;
    rb_tensor.define_method("values", method!(Tensor::values, 0))?;
    rb_tensor.define_method("values_f32", method!(Tensor::values_f32, 0))?;
    rb_tensor.define_method("item", method!(Tensor::item, 0))?;
    rb_tensor.define_method("shape", method!(Tensor::shape, 0))?;
    rb_tensor.define_method("stride", method!(Tensor::stride, 0))?;
    rb_tensor.define_method("dtype", method!(Tensor::dtype, 0))?;
    rb_tensor.define_method("device", method!(Tensor::device, 0))?;
    rb_tensor.define_method("rank", method!(Tensor::rank, 0))?;
    rb_tensor.define_method("elem_count", method!(Tensor::elem_count, 0))?;
    rb_tensor.define_method("sin", method!(Tensor::sin, 0))?;
    rb_tensor.define_method("cos", method!(Tensor::cos, 0))?;
    rb_tensor.define_method("log", method!(Tensor::log, 0))?;
    rb_tensor.define_method("sqr", method!(Tensor::sqr, 0))?;
    rb_tensor.define_method("mean", method!(Tensor::mean, 1))?;
    rb_tensor.define_method("sum", method!(Tensor::sum, 1))?;
    rb_tensor.define_method("sqrt", method!(Tensor::sqrt, 0))?;
    rb_tensor.define_method("/", method!(Tensor::__truediv__, 1))?; // Accepts Tensor, Float, or Integer
    rb_tensor.define_method("recip", method!(Tensor::recip, 0))?;
    rb_tensor.define_method("exp", method!(Tensor::exp, 0))?;
    rb_tensor.define_method("powf", method!(Tensor::powf, 1))?;
    rb_tensor.define_method("index_select", method!(Tensor::index_select, 2))?;
    rb_tensor.define_method("matmul", method!(Tensor::matmul, 1))?;
    rb_tensor.define_method("broadcast_add", method!(Tensor::broadcast_add, 1))?;
    rb_tensor.define_method("broadcast_sub", method!(Tensor::broadcast_sub, 1))?;
    rb_tensor.define_method("broadcast_mul", method!(Tensor::broadcast_mul, 1))?;
    rb_tensor.define_method("broadcast_div", method!(Tensor::broadcast_div, 1))?;
    rb_tensor.define_method("where_cond", method!(Tensor::where_cond, 2))?;
    rb_tensor.define_method("+", method!(Tensor::__add__, 1))?;
    rb_tensor.define_method("*", method!(Tensor::__mul__, 1))?;
    rb_tensor.define_method("-", method!(Tensor::__sub__, 1))?;
    rb_tensor.define_method("reshape", method!(Tensor::reshape, 1))?;
    rb_tensor.define_method("broadcast_as", method!(Tensor::broadcast_as, 1))?;
    rb_tensor.define_method("broadcast_left", method!(Tensor::broadcast_left, 1))?;
    rb_tensor.define_method("squeeze", method!(Tensor::squeeze, 1))?;
    rb_tensor.define_method("unsqueeze", method!(Tensor::unsqueeze, 1))?;
    rb_tensor.define_method("get", method!(Tensor::get, 1))?;
    rb_tensor.define_method("[]", method!(Tensor::get, 1))?;
    rb_tensor.define_method("transpose", method!(Tensor::transpose, 2))?;
    rb_tensor.define_method("narrow", method!(Tensor::narrow, 3))?;
    rb_tensor.define_method("argmax_keepdim", method!(Tensor::argmax_keepdim, 1))?;
    rb_tensor.define_method("argmin_keepdim", method!(Tensor::argmin_keepdim, 1))?;
    rb_tensor.define_method("max_keepdim", method!(Tensor::max_keepdim, 1))?;
    rb_tensor.define_method("min_keepdim", method!(Tensor::min_keepdim, 1))?;
    // rb_tensor.define_method("eq", method!(Tensor::eq, 1))?;
    // rb_tensor.define_method("ne", method!(Tensor::ne, 1))?;
    // rb_tensor.define_method("lt", method!(Tensor::lt, 1))?;
    // rb_tensor.define_method("gt", method!(Tensor::gt, 1))?;
    // rb_tensor.define_method("ge", method!(Tensor::ge, 1))?;
    // rb_tensor.define_method("le", method!(Tensor::le, 1))?;
    rb_tensor.define_method("sum_all", method!(Tensor::sum_all, 0))?;
    rb_tensor.define_method("mean_all", method!(Tensor::mean_all, 0))?;
    rb_tensor.define_method("flatten_from", method!(Tensor::flatten_from, 1))?;
    rb_tensor.define_method("flatten_to", method!(Tensor::flatten_to, 1))?;
    rb_tensor.define_method("flatten_all", method!(Tensor::flatten_all, 0))?;
    rb_tensor.define_method("t", method!(Tensor::t, 0))?;
    rb_tensor.define_method("contiguous", method!(Tensor::contiguous, 0))?;
    rb_tensor.define_method("is_contiguous", method!(Tensor::is_contiguous, 0))?;
    rb_tensor.define_method(
        "is_fortran_contiguous",
        method!(Tensor::is_fortran_contiguous, 0),
    )?;
    rb_tensor.define_method("detach", method!(Tensor::detach, 0))?;
    rb_tensor.define_method("copy", method!(Tensor::copy, 0))?;
    rb_tensor.define_method("to_dtype", method!(Tensor::to_dtype, 1))?;
    rb_tensor.define_method("to_device", method!(Tensor::to_device, 1))?;
    rb_tensor.define_method("to_s", method!(Tensor::__str__, 0))?;
    rb_tensor.define_method("inspect", method!(Tensor::__repr__, 0))?;
    Ok(())
}

