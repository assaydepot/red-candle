use candle_core::{Device, Result as CandleResult};
use tokenizers::Tokenizer;

pub mod mistral;
pub mod generation_config;
pub mod text_generation;

pub use generation_config::GenerationConfig;
pub use text_generation::TextGeneration;

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

/// Common structure for managing tokenizer
#[derive(Debug)]
pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
}

impl TokenizerWrapper {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> CandleResult<Vec<u32>> {
        let encoding = self.tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenizer error: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> CandleResult<String> {
        self.tokenizer
            .decode(tokens, skip_special_tokens)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenizer decode error: {}", e)))
    }

    pub fn token_to_piece(&self, token: u32) -> CandleResult<String> {
        self.tokenizer
            .id_to_token(token)
            .map(|s| s.to_string())
            .ok_or_else(|| candle_core::Error::Msg(format!("Unknown token id: {}", token)))
    }
}