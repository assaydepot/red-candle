use magnus::{function, method, prelude::*, Error, Ruby};

//use candle_core::{DType, Device, Tensor, WithDType};

#[magnus::wrap(class = "Candle::Tensor", free_immediately, size)]
struct Tensor(candle_core::Tensor);

#[magnus::wrap(class = "Candle::DType", free_immediately, size)]
struct DType(candle_core::DType);

#[magnus::wrap(class = "Candle::Device")]
enum Device {
    Cpu,
    Cuda,
}

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

    fn squeeze(&self, dim: usize) -> Tensor {
        Tensor(self.0.squeeze(dim).unwrap())
    }

    fn unsqueeze(&self, dim: usize) -> Tensor {
        Tensor(self.0.unsqueeze(dim).unwrap())
    }

    fn get(&self, index: usize) -> Tensor {
        Tensor(self.0.get(index).unwrap())
    }

    fn transpose(&self, dim1: usize, dim2: usize) -> Tensor {
        Tensor(self.0.transpose(dim1, dim2).unwrap())
    }

    fn narrow(&self, dim: usize, start: usize, len: usize) -> Tensor {
        Tensor(self.0.narrow(dim, start, len).unwrap())
    }

    fn sum_all(&self) -> Tensor {
        Tensor(self.0.sum_all().unwrap())
    }

    fn mean_all(&self) -> Tensor {
        let elements = self.0.elem_count();
        let sum = self.0.sum_all().unwrap();
        let mean = (sum / elements as f64).unwrap();
        Tensor(mean)
    }

    fn flatten_all(&self) -> Tensor {
        Tensor(self.0.flatten_all().unwrap())
    }

    fn t(&self) -> Tensor {
        Tensor(self.0.t().unwrap())
    }

    fn contiguous(&self) -> Tensor {
        Tensor(self.0.contiguous().unwrap())
    }

    fn is_contiguous(&self) -> bool {
        self.0.is_contiguous()
    }

    fn is_fortran_contiguous(&self) -> bool {
        self.0.is_fortran_contiguous()
    }

    fn detach(&self) -> Tensor {
        Tensor(self.0.detach().unwrap())
    }

    fn copy(&self) -> Tensor {
        Tensor(self.0.copy().unwrap())
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
fn init(ruby: &Ruby) -> Result<(), Error> {
    let rb_candle = ruby.define_module("Candle")?;
    rb_candle.const_set("VERSION", "0.1.0")?;
    candle_utils(rb_candle)?;
    let rb_tensor = rb_candle.define_class("Tensor", Ruby::class_object(ruby))?;
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
    rb_tensor.define_method("+", method!(Tensor::__add__, 1))?;
    rb_tensor.define_method("*", method!(Tensor::__mul__, 1))?;
    rb_tensor.define_method("-", method!(Tensor::__sub__, 1))?;
    rb_tensor.define_method("reshape", method!(Tensor::reshape, 1))?;
    rb_tensor.define_method("broadcast_as", method!(Tensor::broadcast_as, 1))?;
    rb_tensor.define_method("broadcast_left", method!(Tensor::broadcast_left, 1))?;
    rb_tensor.define_method("squeeze", method!(Tensor::squeeze, 1))?;
    rb_tensor.define_method("unsqueeze", method!(Tensor::unsqueeze, 1))?;
    rb_tensor.define_method("get", method!(Tensor::get, 1))?;
    rb_tensor.define_method("transpose", method!(Tensor::transpose, 2))?;
    rb_tensor.define_method("narrow", method!(Tensor::narrow, 3))?;
    rb_tensor.define_method("sum_all", method!(Tensor::sum_all, 0))?;
    rb_tensor.define_method("mean_all", method!(Tensor::mean_all, 0))?;
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
    rb_tensor.define_method("to_s", method!(Tensor::__str__, 0))?;
    let rb_dtype = rb_candle.define_class("DType", Ruby::class_object(ruby))?;
    rb_dtype.define_method("to_s", method!(DType::__str__, 0))?;
    Ok(())
}
