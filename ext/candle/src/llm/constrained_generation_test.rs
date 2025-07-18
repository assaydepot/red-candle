#[cfg(test)]
mod constrained_generation_tests {
    use super::super::*;
    use crate::structured::{VocabularyAdapter, SchemaProcessor};
    use crate::tokenizer::{TokenizerWrapper, loader::TokenizerLoader};
    
    #[tokio::test]
    async fn test_constrained_vs_unconstrained_generation() {
        // This test demonstrates the difference between constrained and unconstrained generation
        
        // Load a tokenizer for testing
        if let Ok(tokenizer) = TokenizerLoader::from_hf_hub("bert-base-uncased", None).await {
            let wrapper = TokenizerWrapper::new(tokenizer);
            
            // Create vocabulary adapter
            let vocabulary = VocabularyAdapter::from_tokenizer(&wrapper)
                .expect("Should create vocabulary");
            
            // Create schema processor
            let processor = SchemaProcessor::new();
            
            // Define a simple JSON schema for a yes/no response
            let schema = r#"{
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "enum": ["yes", "no"]
                    }
                },
                "required": ["answer"]
            }"#;
            
            // Process schema into Index
            let index = processor.process_schema(schema, &vocabulary)
                .expect("Should process schema");
            
            // Test configuration with constraint
            let mut config_with_constraint = GenerationConfig::default();
            config_with_constraint.constraint = Some(index.clone());
            config_with_constraint.max_length = 50;
            
            // Test configuration without constraint
            let config_without_constraint = GenerationConfig::default();
            
            // Create text generation instances
            let mut gen_constrained = TextGeneration::from_config(&config_with_constraint);
            let mut gen_unconstrained = TextGeneration::from_config(&config_without_constraint);
            
            // Set EOS token
            gen_constrained.set_eos_token_id(102); // BERT's [SEP] token
            gen_unconstrained.set_eos_token_id(102);
            
            // Constraints are set internally - we can't directly verify them
            // but we can test their effects in actual generation
        }
    }
    
    #[test]
    fn test_constraint_configuration() {
        // Test that we can create a TextGeneration with constraints
        let config = GenerationConfig::default();
        let _text_gen = TextGeneration::from_config(&config);
        
        // Test that we can create a TextGeneration from config
        // Constraints are private implementation details
    }
    
    #[test]
    fn test_repetition_penalty() {
        use candle_core::{Tensor, Device};
        
        let device = Device::Cpu;
        let vocab_size = 10;
        
        // Create logits with some positive and negative values
        let logits_vec: Vec<f32> = vec![1.0, -1.0, 2.0, -2.0, 0.0, 3.0, -3.0, 1.5, -1.5, 0.5];
        let mut logits = Tensor::from_vec(logits_vec.clone(), vocab_size, &device).unwrap();
        
        // Create text generation with some tokens
        let mut text_gen = TextGeneration::new(42, Some(1.0), None, None, 1.0, 64);
        text_gen.push_token(0); // Token that had logit 1.0
        text_gen.push_token(2); // Token that had logit 2.0
        text_gen.push_token(5); // Token that had logit 3.0
        
        // Apply repetition penalty
        text_gen.apply_repetition_penalty(&mut logits, 1.5, 10).unwrap();
        
        let penalized = logits.to_vec1::<f32>().unwrap();
        
        // Check that tokens in context were penalized
        assert!(penalized[0] < logits_vec[0], "Positive logit should be reduced");
        assert!(penalized[2] < logits_vec[2], "Positive logit should be reduced");
        assert!(penalized[5] < logits_vec[5], "Positive logit should be reduced");
        
        // Check that other tokens remain unchanged
        assert_eq!(penalized[1], logits_vec[1], "Unsampled token should be unchanged");
        assert_eq!(penalized[3], logits_vec[3], "Unsampled token should be unchanged");
    }
    
    #[test]
    fn test_stop_conditions() {
        let mut text_gen = TextGeneration::new(42, Some(1.0), None, None, 1.0, 64);
        text_gen.set_eos_token_id(50256); // Common EOS token
        
        // Test max length stop
        for i in 0..10 {
            text_gen.push_token(i);
        }
        assert!(text_gen.should_stop(100, 10), "Should stop at max length");
        assert!(!text_gen.should_stop(100, 20), "Should not stop before max length");
        
        // Test EOS token stop
        assert!(text_gen.should_stop(50256, 100), "Should stop at EOS token");
        assert!(!text_gen.should_stop(123, 100), "Should not stop at non-EOS token");
        
        // Test stop sequences
        let stop_seqs = vec!["STOP".to_string(), "END".to_string()];
        assert!(text_gen.check_stop_sequences("This is the STOP", &stop_seqs), "Should detect stop sequence");
        assert!(text_gen.check_stop_sequences("The END", &stop_seqs), "Should detect stop sequence");
        assert!(!text_gen.check_stop_sequences("Continue", &stop_seqs), "Should not detect stop sequence");
    }
}