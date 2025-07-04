use magnus::{function, method, prelude::*, Ruby};

use crate::ruby::candle_utils;
use crate::ruby::{DType, Device, QTensor, Result as RbResult, Tensor};

pub mod llm;
pub mod reranker;
pub mod ruby;

#[magnus::init]
fn init(ruby: &Ruby) -> RbResult<()> {
    let rb_candle = ruby.define_module("Candle")?;
    ruby::init_embedding_model(rb_candle)?;
    ruby::init_llm(rb_candle)?;
    reranker::init(rb_candle)?;
    candle_utils(rb_candle)?;
    let rb_tensor = rb_candle.define_class("Tensor", Ruby::class_object(ruby))?;
    rb_tensor.define_singleton_method("new", function!(Tensor::new, 2))?;
    // rb_tensor.define_singleton_method("cat", function!(Tensor::cat, 2))?;
    // rb_tensor.define_singleton_method("stack", function!(Tensor::stack, 2))?;
    rb_tensor.define_singleton_method("rand", function!(Tensor::rand, 1))?;
    rb_tensor.define_singleton_method("randn", function!(Tensor::randn, 1))?;
    rb_tensor.define_singleton_method("ones", function!(Tensor::ones, 1))?;
    rb_tensor.define_singleton_method("zeros", function!(Tensor::zeros, 1))?;
    rb_tensor.define_method("values", method!(Tensor::values, 0))?;
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

    let rb_dtype = rb_candle.define_class("DType", Ruby::class_object(ruby))?;
    rb_dtype.define_method("to_s", method!(DType::__str__, 0))?;
    rb_dtype.define_method("inspect", method!(DType::__repr__, 0))?;

    let rb_device = rb_candle.define_class("Device", Ruby::class_object(ruby))?;
    rb_device.define_singleton_method("cpu", function!(Device::cpu, 0))?;
    rb_device.define_singleton_method("cuda", function!(Device::cuda, 0))?;
    rb_device.define_singleton_method("metal", function!(Device::metal, 0))?;
    rb_device.define_method("to_s", method!(Device::__str__, 0))?;
    rb_device.define_method("inspect", method!(Device::__repr__, 0))?;

    let rb_qtensor = rb_candle.define_class("QTensor", Ruby::class_object(ruby))?;
    rb_qtensor.define_method("ggml_dtype", method!(QTensor::ggml_dtype, 0))?;
    rb_qtensor.define_method("rank", method!(QTensor::rank, 0))?;
    rb_qtensor.define_method("shape", method!(QTensor::shape, 0))?;
    rb_qtensor.define_method("dequantize", method!(QTensor::dequantize, 0))?;

    Ok(())
}
