use candle_core::{Result as CandleResult, Tensor};
use std::sync::{Arc, Mutex};

use super::{SD3Pipeline, SD3Config};

/// Thread-safe wrapper for SD3Pipeline
pub struct ThreadSafeSD3Pipeline {
    inner: Arc<Mutex<SD3Pipeline>>,
}

impl ThreadSafeSD3Pipeline {
    pub fn new(pipeline: SD3Pipeline) -> Self {
        Self {
            inner: Arc::new(Mutex::new(pipeline)),
        }
    }
    
    pub fn generate(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        config: &SD3Config,
        seed: Option<u64>,
        progress_callback: Option<&mut dyn FnMut(usize, usize, Option<&Tensor>)>,
    ) -> CandleResult<Tensor> {
        let mut pipeline = self.inner.lock().unwrap();
        pipeline.generate(prompt, negative_prompt, config, seed, progress_callback)
    }
}

unsafe impl Send for ThreadSafeSD3Pipeline {}
unsafe impl Sync for ThreadSafeSD3Pipeline {}