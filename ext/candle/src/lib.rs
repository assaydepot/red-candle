use magnus::{function, prelude::*, Ruby};

use crate::ruby::candle_utils;
use crate::ruby::Result;

pub mod llm;
pub mod ner;
pub mod reranker;
pub mod ruby;
pub mod tokenizer;

// Configuration detection from build.rs
#[cfg(all(has_metal, not(force_cpu)))]
const DEFAULT_DEVICE: &str = "metal";

#[cfg(all(has_cuda, not(has_metal), not(force_cpu)))]
const DEFAULT_DEVICE: &str = "cuda";

#[cfg(any(force_cpu, not(any(has_metal, has_cuda))))]
const DEFAULT_DEVICE: &str = "cpu";

// Export build configuration for runtime checks
pub fn get_build_info() -> magnus::RHash {
    let ruby = magnus::Ruby::get().unwrap();
    let hash = ruby.hash_new();
    
    let _ = hash.aset("default_device", DEFAULT_DEVICE);
    let _ = hash.aset("cuda_available", cfg!(feature = "cuda"));
    let _ = hash.aset("metal_available", cfg!(feature = "metal"));
    let _ = hash.aset("mkl_available", cfg!(feature = "mkl"));
    let _ = hash.aset("accelerate_available", cfg!(feature = "accelerate"));
    let _ = hash.aset("cudnn_available", cfg!(feature = "cudnn"));
    
    hash
}

#[magnus::init]
fn init(ruby: &Ruby) -> Result<()> {
    let rb_candle = ruby.define_module("Candle")?;
    
    // Export build info
    rb_candle.define_singleton_method("build_info", function!(get_build_info, 0))?;
    
    ruby::init_embedding_model(rb_candle)?;
    ruby::init_llm(rb_candle)?;
    ner::init(rb_candle)?;
    reranker::init(rb_candle)?;
    ruby::dtype::init(rb_candle)?;
    ruby::device::init(rb_candle)?;
    ruby::tensor::init(rb_candle)?;
    ruby::tokenizer::init(rb_candle)?;
    candle_utils(rb_candle)?;

    Ok(())
}
