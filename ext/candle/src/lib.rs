use magnus::{function, method, prelude::*, Ruby};

use crate::model::{candle_utils, RbModel, RbDType, RbDevice, RbQTensor, RbResult, RbTensor};

pub mod model;

#[magnus::init]
fn init(ruby: &Ruby) -> RbResult<()> {
    let rb_candle = ruby.define_module("Candle")?;
    candle_utils(rb_candle)?;
    let rb_tensor = rb_candle.define_class("Tensor", Ruby::class_object(ruby))?;
    rb_tensor.define_singleton_method("new", function!(RbTensor::new, 2))?;
    // rb_tensor.define_singleton_method("cat", function!(RbTensor::cat, 2))?;
    // rb_tensor.define_singleton_method("stack", function!(RbTensor::stack, 2))?;
    rb_tensor.define_singleton_method("rand", function!(RbTensor::rand, 1))?;
    rb_tensor.define_singleton_method("randn", function!(RbTensor::randn, 1))?;
    rb_tensor.define_singleton_method("ones", function!(RbTensor::ones, 1))?;
    rb_tensor.define_singleton_method("zeros", function!(RbTensor::zeros, 1))?;
    rb_tensor.define_method("values", method!(RbTensor::values, 0))?;
    rb_tensor.define_method("shape", method!(RbTensor::shape, 0))?;
    rb_tensor.define_method("stride", method!(RbTensor::stride, 0))?;
    rb_tensor.define_method("dtype", method!(RbTensor::dtype, 0))?;
    rb_tensor.define_method("device", method!(RbTensor::device, 0))?;
    rb_tensor.define_method("rank", method!(RbTensor::rank, 0))?;
    rb_tensor.define_method("elem_count", method!(RbTensor::elem_count, 0))?;
    rb_tensor.define_method("sin", method!(RbTensor::sin, 0))?;
    rb_tensor.define_method("cos", method!(RbTensor::cos, 0))?;
    rb_tensor.define_method("log", method!(RbTensor::log, 0))?;
    rb_tensor.define_method("sqr", method!(RbTensor::sqr, 0))?;
    rb_tensor.define_method("sqrt", method!(RbTensor::sqrt, 0))?;
    rb_tensor.define_method("recip", method!(RbTensor::recip, 0))?;
    rb_tensor.define_method("exp", method!(RbTensor::exp, 0))?;
    rb_tensor.define_method("powf", method!(RbTensor::powf, 1))?;
    rb_tensor.define_method("index_select", method!(RbTensor::index_select, 2))?;
    rb_tensor.define_method("matmul", method!(RbTensor::matmul, 1))?;
    rb_tensor.define_method("broadcast_add", method!(RbTensor::broadcast_add, 1))?;
    rb_tensor.define_method("broadcast_sub", method!(RbTensor::broadcast_sub, 1))?;
    rb_tensor.define_method("broadcast_mul", method!(RbTensor::broadcast_mul, 1))?;
    rb_tensor.define_method("broadcast_div", method!(RbTensor::broadcast_div, 1))?;
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
    rb_tensor.define_method("[]", method!(RbTensor::get, 1))?;
    rb_tensor.define_method("transpose", method!(RbTensor::transpose, 2))?;
    rb_tensor.define_method("narrow", method!(RbTensor::narrow, 3))?;
    rb_tensor.define_method("argmax_keepdim", method!(RbTensor::argmax_keepdim, 1))?;
    rb_tensor.define_method("argmin_keepdim", method!(RbTensor::argmin_keepdim, 1))?;
    rb_tensor.define_method("max_keepdim", method!(RbTensor::max_keepdim, 1))?;
    rb_tensor.define_method("min_keepdim", method!(RbTensor::min_keepdim, 1))?;
    // rb_tensor.define_method("eq", method!(RbTensor::eq, 1))?;
    // rb_tensor.define_method("ne", method!(RbTensor::ne, 1))?;
    // rb_tensor.define_method("lt", method!(RbTensor::lt, 1))?;
    // rb_tensor.define_method("gt", method!(RbTensor::gt, 1))?;
    // rb_tensor.define_method("ge", method!(RbTensor::ge, 1))?;
    // rb_tensor.define_method("le", method!(RbTensor::le, 1))?;
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
    rb_tensor.define_method("to_dtype", method!(RbTensor::to_dtype, 1))?;
    rb_tensor.define_method("to_device", method!(RbTensor::to_device, 1))?;
    rb_tensor.define_method("to_s", method!(RbTensor::__str__, 0))?;
    rb_tensor.define_method("inspect", method!(RbTensor::__repr__, 0))?;

    let rb_dtype = rb_candle.define_class("DType", Ruby::class_object(ruby))?;
    rb_dtype.define_method("to_s", method!(RbDType::__str__, 0))?;
    rb_dtype.define_method("inspect", method!(RbDType::__repr__, 0))?;

    let rb_device = rb_candle.define_class("Device", Ruby::class_object(ruby))?;
    rb_device.define_method("to_s", method!(RbDevice::__str__, 0))?;
    rb_device.define_method("inspect", method!(RbDevice::__repr__, 0))?;

    let rb_qtensor = rb_candle.define_class("QTensor", Ruby::class_object(ruby))?;
    rb_qtensor.define_method("ggml_dtype", method!(RbQTensor::ggml_dtype, 0))?;
    rb_qtensor.define_method("rank", method!(RbQTensor::rank, 0))?;
    rb_qtensor.define_method("shape", method!(RbQTensor::shape, 0))?;
    rb_qtensor.define_method("dequantize", method!(RbQTensor::dequantize, 0))?;

    let rb_model = rb_candle.define_class("Model", Ruby::class_object(ruby))?;
    rb_model.define_singleton_method("new", function!(RbModel::new, 0))?;
    rb_model.define_singleton_method("new1", function!(RbModel::new1, 1))?;
    rb_model.define_singleton_method("new2", function!(RbModel::new2, 2))?;
    rb_model.define_method("embedding", method!(RbModel::embedding, 1))?;
    rb_model.define_method("to_s", method!(RbModel::__str__, 0))?;
    rb_model.define_method("inspect", method!(RbModel::__repr__, 0))?;

    Ok(())
}
