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
}

impl DType {
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    let module = ruby.define_module("Candle")?;
    let class1 = module.define_class("Tensor", Ruby::class_object(ruby))?;
    let class2 = module.define_class("DType", Ruby::class_object(ruby))?;
    class1.define_singleton_method("new", function!(Tensor::new, 1))?;
    class1.define_method("shape", method!(Tensor::shape, 0))?;
    class1.define_method("stride", method!(Tensor::stride, 0))?;
    class1.define_method("dtype", method!(Tensor::dtype, 0))?;
    class1.define_method("rank", method!(Tensor::rank, 0))?;
    class1.define_method("to_s", method!(Tensor::__str__, 0))?;
    Ok(())
}
