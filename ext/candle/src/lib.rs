use magnus::{function, method, prelude::*, Error, Ruby};

//use candle_core::{DType, Device, Tensor, WithDType};

#[magnus::wrap(class = "Candle::Tensor", free_immediately, size)]
struct Tensor(candle_core::Tensor);

#[magnus::wrap(class = "Candle::DType", free_immediately, size)]
struct DType(candle_core::DType);

impl Tensor {
    fn new(array: Vec<f32>) -> Self {
        use candle_core::Device::Cpu;
        Self(candle_core::Tensor::new(array.as_slice(), &Cpu).unwrap())
    }

    fn shape(&self) -> Vec<usize> {
        self.0.dims().to_vec()
    }

    fn stride(&self) -> Vec<usize> {
        self.0.stride().to_vec()
    }

    fn dtype(&self) -> DType {
        DType(self.0.dtype())
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

    fn sin(&self) -> Tensor {
        Tensor(self.0.sin().unwrap())
    }

    fn cos(&self) -> Tensor {
        Tensor(self.0.cos().unwrap())
    }

    fn log(&self) -> Tensor {
        Tensor(self.0.log().unwrap())
    }

    fn sqr(&self) -> Tensor {
        Tensor(self.0.sqr().unwrap())
    }

    fn sqrt(&self) -> Tensor {
        Tensor(self.0.sqrt().unwrap())
    }

    fn recip(&self) -> Tensor {
        Tensor(self.0.recip().unwrap())
    }

    fn exp(&self) -> Tensor {
        Tensor(self.0.exp().unwrap())
    }

    fn powf(&self, n: f64) -> Tensor {
        Tensor(self.0.powf(n).unwrap())
    }

    fn matmul(&self, other: &Tensor) -> Tensor {
        Tensor(self.0.matmul(&other.0).unwrap())
    }

    fn where_cond(&self, on_true: &Tensor, on_false: &Tensor) -> Tensor {
        Tensor(self.0.where_cond(&on_true.0, &on_false.0).unwrap())
    }

    fn __add__(&self, rhs: &Tensor) -> Tensor {
        Tensor((&self.0 + &rhs.0).unwrap())
    }

    fn __mul__(&self, rhs: &Tensor) -> Tensor {
        Tensor((&self.0 * &rhs.0).unwrap())
    }

    fn __sub__(&self, rhs: &Tensor) -> Tensor {
        Tensor((&self.0 - &rhs.0).unwrap())
    }

    fn reshape(&self, shape: Vec<usize>) -> Tensor {
        Tensor(self.0.reshape(shape).unwrap())
    }

    fn broadcast_as(&self, shape: Vec<usize>) -> Tensor {
        Tensor(self.0.broadcast_as(shape).unwrap())
    }

    fn broadcast_left(&self, shape: Vec<usize>) -> Tensor {
        Tensor(self.0.broadcast_left(shape).unwrap())
    }
}

impl DType {
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

fn candle_utils(rb_candle: magnus::RModule) -> Result<(), Error> {
    let rb_utils = rb_candle.define_module("Utils")?;
    rb_utils.define_singleton_method("cuda_is_available", function!(candle_core::utils::cuda_is_available, 0))?;
    rb_utils.define_singleton_method("get_num_threads", function!(candle_core::utils::get_num_threads, 0))?;
    rb_utils.define_singleton_method("has_accelerate", function!(candle_core::utils::has_accelerate, 0))?;
    rb_utils.define_singleton_method("has_mkl", function!(candle_core::utils::has_mkl, 0))?;
    Ok(())
}

#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    let rb_candle = ruby.define_module("Candle")?;
    candle_utils(rb_candle)?;
    let rb_tensor = rb_candle.define_class("Tensor", Ruby::class_object(ruby))?;
    let rb_dtype = rb_candle.define_class("DType", Ruby::class_object(ruby))?;
    rb_tensor.define_singleton_method("new", function!(Tensor::new, 1))?;
    rb_tensor.define_method("shape", method!(Tensor::shape, 0))?;
    rb_tensor.define_method("stride", method!(Tensor::stride, 0))?;
    rb_tensor.define_method("dtype", method!(Tensor::dtype, 0))?;
    rb_tensor.define_method("rank", method!(Tensor::rank, 0))?;
    rb_tensor.define_method("sin", method!(Tensor::sin, 0))?;
    rb_tensor.define_method("cos", method!(Tensor::cos, 0))?;
    rb_tensor.define_method("log", method!(Tensor::log, 0))?;
    rb_tensor.define_method("sqr", method!(Tensor::sqr, 0))?;
    rb_tensor.define_method("sqrt", method!(Tensor::sqrt, 0))?;
    rb_tensor.define_method("recip", method!(Tensor::recip, 0))?;
    rb_tensor.define_method("exp", method!(Tensor::exp, 0))?;
    rb_tensor.define_method("powf", method!(Tensor::powf, 1))?;
    rb_tensor.define_method("matmul", method!(Tensor::matmul, 1))?;
    rb_tensor.define_method("where_cond", method!(Tensor::where_cond, 2))?;
    rb_tensor.define_method("__add__", method!(Tensor::__add__, 1))?;
    rb_tensor.define_method("__mul__", method!(Tensor::__mul__, 1))?;
    rb_tensor.define_method("__sub__", method!(Tensor::__sub__, 1))?;
    rb_tensor.define_method("to_s", method!(Tensor::__str__, 0))?;
    Ok(())
}
