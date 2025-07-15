use magnus::{function, Error, Module, Object};

use ::candle_core::Tensor;

use crate::ruby::errors::wrap_candle_err;
use crate::ruby::{Result, Tensor as RbTensor};

pub fn actual_index(t: &Tensor, dim: usize, index: i64) -> candle_core::Result<usize> {
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

pub fn actual_dim(t: &Tensor, dim: i64) -> candle_core::Result<usize> {
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

pub fn candle_utils(rb_candle: magnus::RModule) -> Result<()> {
    let rb_utils = rb_candle.define_module("Utils")?;
    rb_utils.define_singleton_method("cuda_is_available", function!(cuda_is_available, 0))?;
    rb_utils.define_singleton_method("get_num_threads", function!(get_num_threads, 0))?;
    rb_utils.define_singleton_method("has_accelerate", function!(has_accelerate, 0))?;
    rb_utils.define_singleton_method("has_mkl", function!(has_mkl, 0))?;
    Ok(())
}

/// Applies the Softmax function to a given tensor.#
/// &RETURNS&: Tensor
#[allow(dead_code)]
fn softmax(tensor: RbTensor, dim: i64) -> Result<RbTensor> {
    let dim = actual_dim(&tensor, dim).map_err(wrap_candle_err)?;
    let sm = candle_nn::ops::softmax(&tensor.0, dim).map_err(wrap_candle_err)?;
    Ok(RbTensor(sm))
}

/// Applies the Sigmoid Linear Unit (SiLU) function to a given tensor.
/// &RETURNS&: Tensor
#[allow(dead_code)]
fn silu(tensor: RbTensor) -> Result<RbTensor> {
    let s = candle_nn::ops::silu(&tensor.0).map_err(wrap_candle_err)?;
    Ok(RbTensor(s))
}
