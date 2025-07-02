use std::sync::Arc;

use crate::ruby::errors::wrap_candle_err;
use crate::ruby::{Tensor, Result as RbResult};
use ::candle_core::{quantized::QTensor as CoreQTensor, Device as CoreDevice};

#[derive(Debug)]
#[magnus::wrap(class = "Candle::QTensor", free_immediately, size)]
/// A quantized tensor.
pub struct QTensor(Arc<CoreQTensor>);

impl std::ops::Deref for QTensor {
    type Target = CoreQTensor;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl QTensor {
    ///Gets the tensors quantized dtype.
    /// &RETURNS&: str
    pub fn ggml_dtype(&self) -> String {
        format!("{:?}", self.0.dtype())
    }

    ///Gets the rank of the tensor.
    /// &RETURNS&: int
    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    ///Gets the shape of the tensor.
    /// &RETURNS&: Tuple[int]
    pub fn shape(&self) -> Vec<usize> {
        self.0.shape().dims().to_vec()
    }

    pub fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Dequantizes the tensor.
    /// &RETURNS&: Tensor
    pub fn dequantize(&self) -> RbResult<Tensor> {
        let tensor = self.0.dequantize(&CoreDevice::Cpu).map_err(wrap_candle_err)?;
        Ok(Tensor(tensor))
    }

    // fn matmul_t(&self, lhs: &Tensor) -> RbResult<Tensor> {
    //     let qmatmul = ::candle_core::quantized::QMatMul::from_arc(self.0.clone());
    //     let res = qmatmul.forward(lhs).map_err(wrap_candle_err)?;
    //     Ok(Tensor(res))
    // }
}
