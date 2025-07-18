use candle_core::Result as CandleResult;
use tokenizers::Tokenizer;

pub mod loader;

/// Common structure for managing tokenizer
#[derive(Debug, Clone)]
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
        let common_len = full_text
            .chars()
            .zip(previous_text.chars())
            .take_while(|(a, b)| a == b)
            .count();
        
        Ok(full_text.chars().skip(common_len).collect())
    }
    
    /// Format tokens with debug information
    pub fn format_tokens_with_debug(&self, tokens: &[u32]) -> CandleResult<String> {
        let mut result = String::new();
        for &token in tokens {
            let piece = self.token_to_piece(token)?;
            result.push_str(&format!("[{}:{}]", token, piece));
        }
        Ok(result)
    }

    /// Encode a batch of texts (needed for reranker)
    pub fn encode_batch(&self, texts: Vec<String>, add_special_tokens: bool) -> CandleResult<Vec<Vec<u32>>> {
        let encodings = self.tokenizer
            .encode_batch(texts, add_special_tokens)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenizer batch error: {}", e)))?;
        
        Ok(encodings.into_iter()
            .map(|encoding| encoding.get_ids().to_vec())
            .collect())
    }

    /// Get the underlying tokenizer (for advanced use cases)
    pub fn inner(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get a mutable reference to the underlying tokenizer (for configuration)
    pub fn inner_mut(&mut self) -> &mut Tokenizer {
        &mut self.tokenizer
    }
}