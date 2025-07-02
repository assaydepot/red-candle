use magnus::value::ReprValue;

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
