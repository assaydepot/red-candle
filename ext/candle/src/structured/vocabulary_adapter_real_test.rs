#[cfg(test)]
mod real_tests {
    use super::super::*;
    use crate::tokenizer::{TokenizerWrapper, loader::TokenizerLoader};
    
    #[tokio::test]
    async fn test_vocabulary_conversion_with_real_outlines() {
        // This test requires network access to download a tokenizer
        // It verifies that our adapter works with the real outlines-core crate
        
        // Load a simple tokenizer
        let tokenizer_result = TokenizerLoader::from_hf_hub("bert-base-uncased", None).await;
        
        if let Ok(tokenizer) = tokenizer_result {
            let wrapper = TokenizerWrapper::new(tokenizer);
            
            // Convert to Outlines vocabulary
            let vocab_result = VocabularyAdapter::from_tokenizer(&wrapper);
            assert!(vocab_result.is_ok(), "Vocabulary conversion should succeed");
            
            let vocabulary = vocab_result.unwrap();
            
            // Verify the vocabulary was created
            // The real Vocabulary doesn't expose a size method directly,
            // but we can verify it exists and has the correct EOS token
            assert_eq!(vocabulary.eos_token_id(), 102); // BERT's [SEP] token
            
            println!("✓ Successfully created Outlines Vocabulary from BERT tokenizer");
        } else {
            println!("⚠️  Skipping test - couldn't download tokenizer (likely offline)");
        }
    }
    
    #[test]
    fn test_vocabulary_adapter_with_mock_data() {
        // This test doesn't require network access
        // It uses a mock tokenizer to verify the conversion logic
        
        use tokenizers::models::wordpiece::WordPiece;
        use tokenizers::Tokenizer;
        use std::collections::HashMap;
        
        // Create a minimal vocabulary
        let mut vocab = HashMap::new();
        vocab.insert("[PAD]".to_string(), 0);
        vocab.insert("[UNK]".to_string(), 1);
        vocab.insert("[SEP]".to_string(), 2);
        vocab.insert("hello".to_string(), 3);
        vocab.insert("world".to_string(), 4);
        
        let model = WordPiece::from_vocab(vocab);
        let tokenizer = Tokenizer::new(model);
        let wrapper = TokenizerWrapper::new(tokenizer);
        
        // Convert to Outlines vocabulary
        let vocab_result = VocabularyAdapter::from_tokenizer(&wrapper);
        assert!(vocab_result.is_ok(), "Vocabulary conversion should succeed");
        
        let vocabulary = vocab_result.unwrap();
        
        // Verify EOS token was found
        assert_eq!(vocabulary.eos_token_id(), 2); // [SEP] token
        
        println!("✓ Mock vocabulary conversion successful");
    }
}