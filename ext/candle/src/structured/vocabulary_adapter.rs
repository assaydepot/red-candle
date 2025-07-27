use crate::tokenizer::TokenizerWrapper;
use candle_core::Result as CandleResult;
use outlines_core::vocabulary::Vocabulary;
use std::collections::HashMap;

/// Adapter to convert between red-candle's TokenizerWrapper and Outlines' Vocabulary
pub struct VocabularyAdapter;

impl VocabularyAdapter {
    /// Convert a TokenizerWrapper's vocabulary to an Outlines Vocabulary
    /// 
    /// # Arguments
    /// * `tokenizer` - The tokenizer to extract vocabulary from
    /// 
    /// # Returns
    /// An Outlines Vocabulary suitable for use with Index construction
    pub fn from_tokenizer(tokenizer: &TokenizerWrapper) -> CandleResult<Vocabulary> {
        // Get the vocabulary mapping from the tokenizer
        let vocab_map: HashMap<String, u32> = tokenizer.inner().get_vocab(true);
        
        // Try to find EOS token in vocabulary
        let eos_token_id = vocab_map.get("</s>")
            .or_else(|| vocab_map.get("<|endoftext|>"))
            .or_else(|| vocab_map.get("<eos>"))
            .or_else(|| vocab_map.get("[SEP]"))
            .copied();
        
        // Create a sorted list of (token_id, token_string) pairs
        let mut token_pairs: Vec<(u32, String)> = vocab_map
            .into_iter()
            .map(|(token, id)| (id, token))
            .collect();
        
        // Sort by token ID to ensure correct indexing
        token_pairs.sort_by_key(|(id, _)| *id);
        
        // Find the maximum token ID to determine vocabulary size
        let max_token_id = token_pairs
            .last()
            .map(|(id, _)| *id)
            .unwrap_or(0);
        
        // Create vocabulary items in the format expected by Outlines
        // We need to handle potential gaps in token IDs
        let mut vocab_items: Vec<(String, Vec<u8>)> = Vec::new();
        let mut current_id = 0;
        
        for (token_id, token_string) in token_pairs {
            // Fill gaps with placeholder tokens
            while current_id < token_id {
                vocab_items.push((
                    format!("<unused_{}>", current_id),
                    format!("<unused_{}>", current_id).into_bytes(),
                ));
                current_id += 1;
            }
            
            // Add the actual token
            // Convert token string to bytes for Outlines
            vocab_items.push((
                token_string.clone(),
                token_string.into_bytes(),
            ));
            current_id += 1;
        }
        
        // Fill any remaining gaps up to a reasonable vocabulary size
        // This ensures we don't have issues with token IDs beyond our vocabulary
        while current_id <= max_token_id {
            vocab_items.push((
                format!("<unused_{}>", current_id),
                format!("<unused_{}>", current_id).into_bytes(),
            ));
            current_id += 1;
        }
        
        // Create the Outlines vocabulary
        // The Vocabulary API expects us to build it token by token
        let mut vocabulary = Vocabulary::new(
            eos_token_id.unwrap_or(0) // Use EOS token ID or 0 as default
        );
        
        // Insert all tokens into the vocabulary
        for (idx, (token, bytes)) in vocab_items.into_iter().enumerate() {
            vocabulary.try_insert(bytes, idx as u32)
                .map_err(|e| candle_core::Error::Msg(
                    format!("Failed to insert token '{}': {:?}", token, e)
                ))?;
        }
        
        Ok(vocabulary)
    }
    
    /// Get vocabulary size from a tokenizer
    pub fn vocab_size(tokenizer: &TokenizerWrapper) -> usize {
        tokenizer.inner().get_vocab_size(true)
    }
    
    /// Extract and validate special tokens
    pub fn get_special_tokens(tokenizer: &TokenizerWrapper) -> HashMap<String, u32> {
        let tokenizer_inner = tokenizer.inner();
        let mut special_tokens = HashMap::new();
        
        // Get common special tokens if they exist
        if let Some(_token) = tokenizer_inner.id_to_token(0) {
            special_tokens.insert("pad_token".to_string(), 0);
        }
        
        // Try to find EOS token
        let vocab = tokenizer_inner.get_vocab(true);
        if let Some(&eos_id) = vocab.get("</s>")
            .or_else(|| vocab.get("<|endoftext|>"))
            .or_else(|| vocab.get("<eos>"))
            .or_else(|| vocab.get("[SEP]")) {
            special_tokens.insert("eos_token".to_string(), eos_id);
        }
        
        // Try to get BOS token if it exists
        if let Some(bos_token) = tokenizer_inner.token_to_id("<s>") {
            special_tokens.insert("bos_token".to_string(), bos_token);
        } else if let Some(bos_token) = tokenizer_inner.token_to_id("<|startoftext|>") {
            special_tokens.insert("bos_token".to_string(), bos_token);
        }
        
        special_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vocabulary_adapter_creation() {
        // This test will be implemented once we have a way to create test tokenizers
        // For now, it serves as a placeholder for the test structure
    }
    
    #[test]
    fn test_special_tokens_extraction() {
        // Test special token extraction logic
    }
    
    #[test]
    fn test_vocab_size() {
        // Test vocabulary size calculation
    }
}