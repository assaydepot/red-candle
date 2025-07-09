use magnus::value::ReprValue;
use magnus::{method, class, RModule, Error, Module};

use ::candle_core::DType as CoreDType;
use crate::ruby::Result as RbResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[magnus::wrap(class = "Candle::DType", free_immediately, size)]

/// A `candle` dtype.
pub struct DType(pub CoreDType);

impl DType {
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl DType {
    pub fn from_rbobject(dtype: magnus::Symbol) -> RbResult<Self> {
        let dtype = unsafe { dtype.to_s() }.unwrap().into_owned();
        use std::str::FromStr;
        let dtype = CoreDType::from_str(&dtype).unwrap();
        Ok(Self(dtype))
    }
}

pub fn init(rb_candle: RModule) -> Result<(), Error> {
    let rb_dtype = rb_candle.define_class("DType", class::object())?;
    rb_dtype.define_method("to_s", method!(DType::__str__, 0))?;
    rb_dtype.define_method("inspect", method!(DType::__repr__, 0))?;
    Ok(())
}
