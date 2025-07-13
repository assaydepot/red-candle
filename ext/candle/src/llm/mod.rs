use candle_core::{Device, Result as CandleResult};
use tokenizers::Tokenizer;

pub mod mistral;
pub mod llama;
pub mod gemma;
pub mod generation_config;
pub mod text_generation;
pub mod quantized_gguf;

pub use generation_config::GenerationConfig;
pub use text_generation::TextGeneration;
pub use quantized_gguf::QuantizedGGUF;

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
    
    /// Decode a single token for streaming output
    pub fn decode_token(&self, token: u32) -> CandleResult<String> {
        // Decode the single token properly
        self.decode(&[token], true)
    }
    
    /// Decode tokens incrementally for streaming
    /// This is more efficient than decoding single tokens
    pub fn decode_incremental(&self, all_tokens: &[u32], new_tokens_start: usize) -> CandleResult<String> {
        if new_tokens_start >= all_tokens.len() {
            return Ok(String::new());
        }
        
        // Decode all tokens up to this point
        let full_text = self.decode(all_tokens, true)?;
        
        // If we're at the start, return everything
        if new_tokens_start == 0 {
            return Ok(full_text);
        }
        
        // Otherwise, decode up to the previous token and return the difference
        let previous_text = self.decode(&all_tokens[..new_tokens_start], true)?;
        
        // Find the common prefix between the two strings to handle cases where
        // the tokenizer might produce slightly different text when decoding
        // different token sequences
        let common_prefix_len = full_text
            .char_indices()
            .zip(previous_text.chars())
            .take_while(|((_, c1), c2)| c1 == c2)
            .count();
        
        // Find the byte position of the character boundary
        let byte_pos = full_text
            .char_indices()
            .nth(common_prefix_len)
            .map(|(pos, _)| pos)
            .unwrap_or(full_text.len());
        
        // Return only the new portion
        Ok(full_text[byte_pos..].to_string())
    }
    
    /// Format tokens with debug information
    pub fn format_tokens_with_debug(&self, tokens: &[u32]) -> CandleResult<String> {
        let mut result = String::new();
        
        for &token in tokens {
            let token_piece = self.token_to_piece(token)?;
            result.push_str(&format!("[{}:{}]", token, token_piece));
        }
        
        Ok(result)
    }
}