use std::cell::RefCell;
use magnus::{function, method, prelude::*, Error, Ruby};

//use candle_core::{DType, Device, Tensor, WithDType};

#[magnus::wrap(class = "Candle::Tensor", free_immediately, size)]
struct Tensor(RefCell<candle_core::Tensor>);

impl Tensor {
    fn new(array: Vec<f32>) -> Self {
        use candle_core::Device::Cpu;
        Self(RefCell::new(candle_core::Tensor::new(array.as_slice(), &Cpu).unwrap()))
    }
    fn shape(&self) -> Vec<usize> {
        self.0.borrow().dims().to_vec()
    }
    fn rank(&self) -> usize {
        self.0.borrow().rank()
    }
}

#[magnus::wrap(class = "Candle::DType", free_immediately, size)]
struct DType(candle_core::DType);

#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    let module = ruby.define_module("Candle")?;
    let class1 = module.define_class("Tensor", Ruby::class_object(ruby))?;
    let class2 = module.define_class("DType", Ruby::class_object(ruby))?;
    class1.define_singleton_method("new", function!(Tensor::new, 1))?;
    class1.define_method("shape", method!(Tensor::shape, 0))?;
    class1.define_method("rank", method!(Tensor::rank, 0))?;
    Ok(())
}
