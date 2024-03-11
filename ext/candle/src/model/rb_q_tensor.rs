use magnus::Error;
use std::sync::Arc;

use ::candle_core::{quantized::QTensor, Device};
use crate::model::errors::wrap_candle_err;
use crate::model::rb_tensor::RbTensor;

type RbResult<T> = Result<T, Error>;

#[derive(Debug)]
#[magnus::wrap(class = "Candle::QTensor", free_immediately, size)]
/// A quantized tensor.
pub struct RbQTensor(Arc<QTensor>);

impl std::ops::Deref for RbQTensor {
    type Target = QTensor;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl RbQTensor {
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
    pub fn dequantize(&self) -> RbResult<RbTensor> {
        let tensor = self.0.dequantize(&Device::Cpu).map_err(wrap_candle_err)?;
        Ok(RbTensor(tensor))
    }

    // fn matmul_t(&self, lhs: &RbTensor) -> RbResult<RbTensor> {
    //     let qmatmul = ::candle_core::quantized::QMatMul::from_arc(self.0.clone());
    //     let res = qmatmul.forward(lhs).map_err(wrap_candle_err)?;
    //     Ok(RbTensor(res))
    // }
}
