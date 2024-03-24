use magnus::prelude::*;

use crate::model::{
    errors::wrap_candle_err,
    rb_d_type::RbDType,
    rb_device::RbDevice,
    rb_result::RbResult,
    utils::{actual_dim, actual_index},
};
use ::candle_core::{DType, Device, Tensor};

#[derive(Clone, Debug)]
#[magnus::wrap(class = "Candle::Tensor", free_immediately, size)]
/// A `candle` tensor.
pub struct RbTensor(pub Tensor);

impl std::ops::Deref for RbTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl RbTensor {
    pub fn new(array: magnus::RArray, dtype: Option<magnus::Symbol>) -> RbResult<Self> {
        let dtype = dtype
            .map(|dtype| RbDType::from_rbobject(dtype))
            .unwrap_or(Ok(RbDType(DType::F32)))?;
        // FIXME: Do not use `to_f64` here.
        let array = array
            .each()
            .map(|v| magnus::Float::try_convert(v?).map(|v| v.to_f64()))
            .collect::<RbResult<Vec<_>>>()?;
        Ok(Self(
            Tensor::new(array.as_slice(), &Device::Cpu)
                .map_err(wrap_candle_err)?
                .to_dtype(dtype.0)
                .map_err(wrap_candle_err)?,
        ))
    }

    pub fn values(&self) -> RbResult<Vec<f64>> {
        let values = self
            .0
            .to_dtype(DType::F64)
            .map_err(wrap_candle_err)?
            .flatten_all()
            .map_err(wrap_candle_err)?
            .to_vec1()
            .map_err(wrap_candle_err)?;
        Ok(values)
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
    pub fn dtype(&self) -> RbDType {
        RbDType(self.0.dtype())
    }

    /// Gets the tensor's device.
    /// &RETURNS&: Device
    pub fn device(&self) -> RbDevice {
        RbDevice::from_device(self.0.device())
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
    pub fn sin(&self) -> RbResult<Self> {
        Ok(RbTensor(self.0.sin().map_err(wrap_candle_err)?))
    }

    /// Performs the `cos` operation on the tensor.
    /// &RETURNS&: Tensor
    pub fn cos(&self) -> RbResult<Self> {
        Ok(RbTensor(self.0.cos().map_err(wrap_candle_err)?))
    }

    /// Performs the `log` operation on the tensor.
    /// &RETURNS&: Tensor
    pub fn log(&self) -> RbResult<Self> {
        Ok(RbTensor(self.0.log().map_err(wrap_candle_err)?))
    }

    /// Squares the tensor.
    /// &RETURNS&: Tensor
    pub fn sqr(&self) -> RbResult<Self> {
        Ok(RbTensor(self.0.sqr().map_err(wrap_candle_err)?))
    }

    /// Calculates the square root of the tensor.
    /// &RETURNS&: Tensor
    pub fn sqrt(&self) -> RbResult<Self> {
        Ok(RbTensor(self.0.sqrt().map_err(wrap_candle_err)?))
    }

    /// Get the `recip` of the tensor.
    /// &RETURNS&: Tensor
    pub fn recip(&self) -> RbResult<Self> {
        Ok(RbTensor(self.0.recip().map_err(wrap_candle_err)?))
    }

    /// Performs the `exp` operation on the tensor.
    /// &RETURNS&: Tensor
    pub fn exp(&self) -> RbResult<Self> {
        Ok(RbTensor(self.0.exp().map_err(wrap_candle_err)?))
    }

    /// Performs the `pow` operation on the tensor with the given exponent.
    /// &RETURNS&: Tensor
    pub fn powf(&self, p: f64) -> RbResult<Self> {
        Ok(RbTensor(self.0.powf(p).map_err(wrap_candle_err)?))
    }

    /// Select values for the input tensor at the target indexes across the specified dimension.
    ///
    /// The `indexes` is argument is an int tensor with a single dimension.
    /// The output has the same number of dimension as the `self` input. The target dimension of
    /// the output has length the length of `indexes` and the values are taken from `self` using
    /// the index from `indexes`. Other dimensions have the same number of elements as the input
    /// tensor.
    /// &RETURNS&: Tensor
    pub fn index_select(&self, rhs: &Self, dim: i64) -> RbResult<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_candle_err)?;
        Ok(RbTensor(
            self.0.index_select(rhs, dim).map_err(wrap_candle_err)?,
        ))
    }

    /// Performs a matrix multiplication between the two tensors.
    /// &RETURNS&: Tensor
    pub fn matmul(&self, rhs: &Self) -> RbResult<Self> {
        Ok(RbTensor(self.0.matmul(rhs).map_err(wrap_candle_err)?))
    }

    /// Adds the two tensors, while broadcasting the right-hand-side tensor to match the shape of the left-hand-side tensor.
    /// &RETURNS&: Tensor
    pub fn broadcast_add(&self, rhs: &Self) -> RbResult<Self> {
        Ok(RbTensor(
            self.0.broadcast_add(rhs).map_err(wrap_candle_err)?,
        ))
    }

    /// Subtracts the two tensors, while broadcasting the right-hand-side tensor to match the shape of the left-hand-side tensor.
    /// &RETURNS&: Tensor
    pub fn broadcast_sub(&self, rhs: &Self) -> RbResult<Self> {
        Ok(RbTensor(
            self.0.broadcast_sub(rhs).map_err(wrap_candle_err)?,
        ))
    }

    /// Multiplies the two tensors, while broadcasting the right-hand-side tensor to match the shape of the left-hand-side tensor.
    /// &RETURNS&: Tensor
    pub fn broadcast_mul(&self, rhs: &Self) -> RbResult<Self> {
        Ok(RbTensor(
            self.0.broadcast_mul(rhs).map_err(wrap_candle_err)?,
        ))
    }

    /// Divides the two tensors, while broadcasting the right-hand-side tensor to match the shape of the left-hand-side tensor.
    /// &RETURNS&: Tensor
    pub fn broadcast_div(&self, rhs: &Self) -> RbResult<Self> {
        Ok(RbTensor(
            self.0.broadcast_div(rhs).map_err(wrap_candle_err)?,
        ))
    }

    /// Returns a tensor with the same shape as the input tensor, the values are taken from
    /// `on_true` if the input tensor value is not zero, and `on_false` at the positions where the
    /// input tensor is equal to zero.
    /// &RETURNS&: Tensor
    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> RbResult<Self> {
        Ok(RbTensor(
            self.0
                .where_cond(on_true, on_false)
                .map_err(wrap_candle_err)?,
        ))
    }

    /// Add two tensors.
    /// &RETURNS&: Tensor
    pub fn __add__(&self, rhs: &RbTensor) -> RbResult<Self> {
        Ok(Self(self.0.add(&rhs.0).map_err(wrap_candle_err)?))
    }

    /// Multiply two tensors.
    /// &RETURNS&: Tensor
    pub fn __mul__(&self, rhs: &RbTensor) -> RbResult<Self> {
        Ok(Self(self.0.mul(&rhs.0).map_err(wrap_candle_err)?))
    }

    /// Subtract two tensors.
    /// &RETURNS&: Tensor
    pub fn __sub__(&self, rhs: &RbTensor) -> RbResult<Self> {
        Ok(Self(self.0.sub(&rhs.0).map_err(wrap_candle_err)?))
    }

    /// Divide two tensors.
    /// &RETURNS&: Tensor
    // fn __truediv__(&self, rhs: &RbTensor) -> RbResult<Self> {
    //     Ok(Self(self.0.div(&rhs.0).map_err(wrap_candle_err)?))
    // }

    /// Reshapes the tensor to the given shape.
    /// &RETURNS&: Tensor
    pub fn reshape(&self, shape: Vec<usize>) -> RbResult<Self> {
        Ok(RbTensor(self.0.reshape(shape).map_err(wrap_candle_err)?))
    }

    /// Broadcasts the tensor to the given shape.
    /// &RETURNS&: Tensor
    pub fn broadcast_as(&self, shape: Vec<usize>) -> RbResult<Self> {
        Ok(RbTensor(
            self.0.broadcast_as(shape).map_err(wrap_candle_err)?,
        ))
    }

    /// Broadcasts the tensor to the given shape, adding new dimensions on the left.
    /// &RETURNS&: Tensor
    pub fn broadcast_left(&self, shape: Vec<usize>) -> RbResult<Self> {
        Ok(RbTensor(
            self.0.broadcast_left(shape).map_err(wrap_candle_err)?,
        ))
    }

    /// Creates a new tensor with the specified dimension removed if its size was one.
    /// &RETURNS&: Tensor
    pub fn squeeze(&self, dim: i64) -> RbResult<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_candle_err)?;
        Ok(RbTensor(self.0.squeeze(dim).map_err(wrap_candle_err)?))
    }

    /// Creates a new tensor with a dimension of size one inserted at the specified position.
    /// &RETURNS&: Tensor
    pub fn unsqueeze(&self, dim: usize) -> RbResult<Self> {
        Ok(RbTensor(self.0.unsqueeze(dim).map_err(wrap_candle_err)?))
    }

    /// Gets the value at the specified index.
    /// &RETURNS&: Tensor
    pub fn get(&self, index: i64) -> RbResult<Self> {
        let index = actual_index(self, 0, index).map_err(wrap_candle_err)?;
        Ok(RbTensor(self.0.get(index).map_err(wrap_candle_err)?))
    }

    /// Returns a tensor that is a transposed version of the input, the given dimensions are swapped.
    /// &RETURNS&: Tensor
    pub fn transpose(&self, dim1: usize, dim2: usize) -> RbResult<Self> {
        Ok(RbTensor(
            self.0.transpose(dim1, dim2).map_err(wrap_candle_err)?,
        ))
    }

    /// Returns a new tensor that is a narrowed version of the input, the dimension `dim`
    /// ranges from `start` to `start + len`.
    /// &RETURNS&: Tensor
    pub fn narrow(&self, dim: i64, start: i64, len: usize) -> RbResult<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_candle_err)?;
        let start = actual_index(self, dim, start).map_err(wrap_candle_err)?;
        Ok(RbTensor(
            self.0.narrow(dim, start, len).map_err(wrap_candle_err)?,
        ))
    }

    /// Returns the indices of the maximum value(s) across the selected dimension.
    /// &RETURNS&: Tensor
    pub fn argmax_keepdim(&self, dim: i64) -> RbResult<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_candle_err)?;
        Ok(RbTensor(
            self.0.argmax_keepdim(dim).map_err(wrap_candle_err)?,
        ))
    }

    /// Returns the indices of the minimum value(s) across the selected dimension.
    /// &RETURNS&: Tensor
    pub fn argmin_keepdim(&self, dim: i64) -> RbResult<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_candle_err)?;
        Ok(RbTensor(
            self.0.argmin_keepdim(dim).map_err(wrap_candle_err)?,
        ))
    }

    /// Gathers the maximum value across the selected dimension.
    /// &RETURNS&: Tensor
    pub fn max_keepdim(&self, dim: i64) -> RbResult<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_candle_err)?;
        Ok(RbTensor(self.0.max_keepdim(dim).map_err(wrap_candle_err)?))
    }

    /// Gathers the minimum value across the selected dimension.
    /// &RETURNS&: Tensor
    pub fn min_keepdim(&self, dim: i64) -> RbResult<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_candle_err)?;
        Ok(RbTensor(self.0.min_keepdim(dim).map_err(wrap_candle_err)?))
    }

    // fn eq(&self, rhs: &Self) -> RbResult<Self> {
    //     Ok(RbTensor(self.0.eq(rhs).map_err(wrap_candle_err)?))
    // }

    // fn ne(&self, rhs: &Self) -> RbResult<Self> {
    //     Ok(RbTensor(self.0.ne(rhs).map_err(wrap_candle_err)?))
    // }

    // fn lt(&self, rhs: &Self) -> RbResult<Self> {
    //     Ok(RbTensor(self.0.lt(rhs).map_err(wrap_candle_err)?))
    // }

    // fn gt(&self, rhs: &Self) -> RbResult<Self> {
    //     Ok(RbTensor(self.0.gt(rhs).map_err(wrap_candle_err)?))
    // }

    // fn ge(&self, rhs: &Self) -> RbResult<Self> {
    //     Ok(RbTensor(self.0.ge(rhs).map_err(wrap_candle_err)?))
    // }

    // fn le(&self, rhs: &Self) -> RbResult<Self> {
    //     Ok(RbTensor(self.0.le(rhs).map_err(wrap_candle_err)?))
    // }

    /// Returns the sum of the tensor.
    /// &RETURNS&: Tensor
    pub fn sum_all(&self) -> RbResult<Self> {
        Ok(RbTensor(self.0.sum_all().map_err(wrap_candle_err)?))
    }

    /// Returns the mean of the tensor.
    /// &RETURNS&: Tensor
    pub fn mean_all(&self) -> RbResult<Self> {
        let elements = self.0.elem_count();
        let sum = self.0.sum_all().map_err(wrap_candle_err)?;
        let mean = (sum / elements as f64).map_err(wrap_candle_err)?;
        Ok(RbTensor(mean))
    }

    /// Flattens the tensor on the dimension indexes from `dim` (inclusive) to the last dimension.
    /// &RETURNS&: Tensor
    pub fn flatten_from(&self, dim: i64) -> RbResult<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_candle_err)?;
        Ok(RbTensor(self.0.flatten_from(dim).map_err(wrap_candle_err)?))
    }

    ///Flattens the tensor on the dimension indexes from `0` to `dim` (inclusive).
    /// &RETURNS&: Tensor
    pub fn flatten_to(&self, dim: i64) -> RbResult<Self> {
        let dim = actual_dim(self, dim).map_err(wrap_candle_err)?;
        Ok(RbTensor(self.0.flatten_to(dim).map_err(wrap_candle_err)?))
    }

    /// Flattens the tensor into a 1D tensor.
    /// &RETURNS&: Tensor
    pub fn flatten_all(&self) -> RbResult<Self> {
        Ok(RbTensor(self.0.flatten_all().map_err(wrap_candle_err)?))
    }

    /// Transposes the tensor.
    /// &RETURNS&: Tensor
    pub fn t(&self) -> RbResult<Self> {
        Ok(RbTensor(self.0.t().map_err(wrap_candle_err)?))
    }

    /// Makes the tensor contiguous in memory.
    /// &RETURNS&: Tensor
    pub fn contiguous(&self) -> RbResult<Self> {
        Ok(RbTensor(self.0.contiguous().map_err(wrap_candle_err)?))
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
    pub fn detach(&self) -> RbResult<Self> {
        Ok(RbTensor(self.0.detach()))
    }

    /// Returns a copy of the tensor.
    /// &RETURNS&: Tensor
    pub fn copy(&self) -> RbResult<Self> {
        Ok(RbTensor(self.0.copy().map_err(wrap_candle_err)?))
    }

    /// Convert the tensor to a new dtype.
    /// &RETURNS&: Tensor
    pub fn to_dtype(&self, dtype: magnus::Symbol) -> RbResult<Self> {
        let dtype = RbDType::from_rbobject(dtype)?;
        Ok(RbTensor(self.0.to_dtype(dtype.0).map_err(wrap_candle_err)?))
    }

    /// Move the tensor to a new device.
    /// &RETURNS&: Tensor
    pub fn to_device(&self, device: RbDevice) -> RbResult<Self> {
        let device = device.as_device()?;
        Ok(RbTensor(
            self.0.to_device(&device).map_err(wrap_candle_err)?,
        ))
    }
}

impl RbTensor {
    // fn cat(tensors: Vec<RbTensor>, dim: i64) -> RbResult<RbTensor> {
    //     if tensors.is_empty() {
    //         return Err(Error::new(
    //             magnus::exception::arg_error(),
    //             "empty input to cat",
    //         ));
    //     }
    //     let dim = actual_dim(&tensors[0].0, dim).map_err(wrap_candle_err)?;
    //     let tensors = tensors.into_iter().map(|t| t.0).collect::<Vec<_>>();
    //     let tensor = Tensor::cat(&tensors, dim).map_err(wrap_candle_err)?;
    //     Ok(RbTensor(tensor))
    // }

    // fn stack(tensors: Vec<RbTensor>, dim: usize) -> RbResult<Self> {
    //     let tensors = tensors.into_iter().map(|t| t.0).collect::<Vec<_>>();
    //     let tensor = Tensor::stack(&tensors, dim).map_err(wrap_candle_err)?;
    //     Ok(Self(tensor))
    // }

    /// Creates a new tensor with random values.
    /// &RETURNS&: Tensor
    pub fn rand(shape: Vec<usize>) -> RbResult<Self> {
        let device = RbDevice::Cpu.as_device()?;
        Ok(Self(
            Tensor::rand(0f32, 1f32, shape, &device).map_err(wrap_candle_err)?,
        ))
    }

    /// Creates a new tensor with random values from a normal distribution.
    /// &RETURNS&: Tensor
    pub fn randn(shape: Vec<usize>) -> RbResult<Self> {
        let device = RbDevice::Cpu.as_device()?;
        Ok(Self(
            Tensor::randn(0f32, 1f32, shape, &device).map_err(wrap_candle_err)?,
        ))
    }

    /// Creates a new tensor filled with ones.
    /// &RETURNS&: Tensor
    pub fn ones(shape: Vec<usize>) -> RbResult<Self> {
        let device = RbDevice::Cpu.as_device()?;
        Ok(Self(
            Tensor::ones(shape, DType::F32, &device).map_err(wrap_candle_err)?,
        ))
    }
    /// Creates a new tensor filled with zeros.
    /// &RETURNS&: Tensor
    pub fn zeros(shape: Vec<usize>) -> RbResult<Self> {
        let device = RbDevice::Cpu.as_device()?;
        Ok(Self(
            Tensor::zeros(shape, DType::F32, &device).map_err(wrap_candle_err)?,
        ))
    }
}
