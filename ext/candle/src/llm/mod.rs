use candle_core::{Device, Result as CandleResult};

pub mod mistral;
pub mod llama;
pub mod gemma;
pub mod qwen;
pub mod generation_config;
pub mod text_generation;
pub mod quantized_gguf;

pub use generation_config::GenerationConfig;
pub use text_generation::TextGeneration;
pub use quantized_gguf::QuantizedGGUF;
pub use crate::tokenizer::TokenizerWrapper;

#[cfg(test)]
mod constrained_generation_test;

/// Trait for text generation models
pub trait TextGenerator: Send + Sync {
    /// Generate text from a prompt
    fn generate(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> CandleResult<String>;

    /// Generate text with streaming callback
    fn generate_stream(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
        callback: impl FnMut(&str),
    ) -> CandleResult<String>;

    /// Get the model's name
    fn model_name(&self) -> &str;

    /// Get the device the model is running on
    fn device(&self) -> &Device;
    
    /// Clear any cached state (like KV cache)
    fn clear_cache(&mut self);
}