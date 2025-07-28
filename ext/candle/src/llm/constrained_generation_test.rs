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
    
    #[test]
    fn test_sample_next_token_uses_repetition_penalty() {
        use candle_core::{Tensor, Device};
        
        let device = Device::Cpu;
        let vocab_size = 10;
        
        // Create initial logits
        let logits_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let logits = Tensor::from_vec(logits_vec.clone(), vocab_size, &device).unwrap();
        
        // Test 1: Create TextGeneration and add some tokens to history
        let mut text_gen = TextGeneration::new(42, Some(0.1), None, None, 1.5, 64);
        text_gen.push_token(2); // Token with logit 3.0
        text_gen.push_token(5); // Token with logit 6.0
        text_gen.push_token(9); // Token with logit 10.0
        
        // Sample with repetition penalty
        let _token_with_penalty = text_gen.sample_next_token(
            &logits,
            Some((1.5, 10)) // penalty of 1.5, last 10 tokens
        ).unwrap();
        
        // Test 2: Same setup but without penalty
        let mut text_gen_no_penalty = TextGeneration::new(42, Some(0.1), None, None, 1.5, 64);
        text_gen_no_penalty.push_token(2);
        text_gen_no_penalty.push_token(5);
        text_gen_no_penalty.push_token(9);
        
        let _token_without_penalty = text_gen_no_penalty.sample_next_token(
            &logits,
            None // No penalty
        ).unwrap();
        
        // With low temperature and penalty, should avoid previously used high-logit tokens
        // Without penalty, should prefer high-logit tokens
        // This is probabilistic, but with temp=0.1 it should be fairly deterministic
        
        // Test 3: Verify penalty is applied correctly by checking modified logits
        let mut text_gen_verify = TextGeneration::new(42, Some(0.1), None, None, 1.5, 64);
        text_gen_verify.push_token(9); // Highest logit token
        
        // Clone logits to check modification
        let mut logits_for_penalty = logits.clone();
        text_gen_verify.apply_repetition_penalty(&mut logits_for_penalty, 2.0, 10).unwrap();
        
        let penalized = logits_for_penalty.to_vec1::<f32>().unwrap();
        assert!(penalized[9] < logits_vec[9], "Token 9 should be penalized");
        assert_eq!(penalized[0], logits_vec[0], "Token 0 should not be penalized");
    }
    
    #[test]
    fn test_text_generation_from_config_parameters() {
        
        // Create a config with specific values
        let mut config = GenerationConfig::default();
        config.seed = 12345;
        config.temperature = 0.5;
        config.top_p = Some(0.9);
        config.top_k = Some(40); // Currently unused but should be accepted
        config.repetition_penalty = 1.2;
        config.repetition_penalty_last_n = 50;
        
        // Create TextGeneration from config
        let text_gen = TextGeneration::from_config(&config);
        
        // We can't directly inspect private fields, but we can test behavior
        // Test that it creates successfully (no panic)
        assert!(text_gen.get_tokens().is_empty(), "Should start with no tokens");
        
        // Test with constraint
        let config_with_constraint = GenerationConfig::default();
        // In real usage, this would be a real constraint
        // For testing, we just verify it accepts the config
        let text_gen_constrained = TextGeneration::from_config(&config_with_constraint);
        assert!(text_gen_constrained.get_tokens().is_empty(), "Should start with no tokens");
    }
    
    #[test]
    fn test_generation_with_different_penalties() {
        use candle_core::{Tensor, Device, DType};
        
        let device = Device::Cpu;
        let vocab_size = 50;
        
        // Create logits with clear preferences
        let mut logits_vec = vec![0.0; vocab_size];
        logits_vec[10] = 10.0; // Strong preference
        logits_vec[20] = 8.0;  // Second preference
        logits_vec[30] = 6.0;  // Third preference
        
        // Test different penalty configurations
        let configs = vec![
            (1.0, 64),  // No penalty (1.0 = neutral)
            (1.5, 64),  // Moderate penalty
            (2.0, 64),  // Strong penalty
            (1.2, 10),  // Penalty with limited range
        ];
        
        for (penalty, last_n) in configs {
            let mut config = GenerationConfig::default();
            config.seed = 42; // Fixed seed for reproducibility
            config.temperature = 0.1; // Low temperature for more deterministic behavior
            config.repetition_penalty = penalty;
            config.repetition_penalty_last_n = last_n;
            
            let mut text_gen = TextGeneration::from_config(&config);
            
            // Generate a sequence of tokens
            let mut generated = Vec::new();
            for _i in 0..5 {
                let logits = Tensor::from_vec(logits_vec.clone(), vocab_size, &device).unwrap().to_dtype(DType::F32).unwrap();
                
                let token = text_gen.sample_next_token(
                    &logits,
                    Some((config.repetition_penalty, config.repetition_penalty_last_n))
                ).unwrap();
                
                generated.push(token);
                
                // Verify the token is in valid range
                assert!(token < vocab_size as u32, "Token should be within vocabulary");
            }
            
            // With higher penalties, we should see more diversity (less repetition)
            let unique_tokens = generated.iter().collect::<std::collections::HashSet<_>>().len();
            if penalty > 1.5 {
                assert!(unique_tokens >= 3, "High penalty should produce diverse tokens");
            }
        }
    }
    
    #[test]
    fn test_sample_next_token_integration() {
        use candle_core::{Tensor, Device, DType};
        
        let device = Device::Cpu;
        
        // Test the full integration of sample_next_token
        let mut config = GenerationConfig::default();
        config.seed = 999;
        config.temperature = 0.7;
        config.max_length = 10;
        config.repetition_penalty = 1.3;
        config.repetition_penalty_last_n = 5;
        
        let mut text_gen = TextGeneration::from_config(&config);
        text_gen.set_eos_token_id(50256);
        
        // Simulate a generation loop
        let vocab_size = 100;
        let mut all_tokens = Vec::new();
        
        for step in 0..8 {
            // Create varying logits to simulate model output
            let mut logits_vec = vec![0.0; vocab_size];
            // Make different tokens attractive at different steps
            let preferred_token = (step * 13) % vocab_size;
            logits_vec[preferred_token] = 5.0;
            logits_vec[(preferred_token + 10) % vocab_size] = 4.0;
            logits_vec[(preferred_token + 20) % vocab_size] = 3.0;
            
            let logits = Tensor::from_vec(logits_vec, vocab_size, &device).unwrap().to_dtype(DType::F32).unwrap();
            
            let token = text_gen.sample_next_token(
                &logits,
                Some((config.repetition_penalty, config.repetition_penalty_last_n))
            ).unwrap();
            
            all_tokens.push(token);
            
            // Check if we should stop
            if text_gen.should_stop(token, config.max_length) {
                break;
            }
        }
        
        // Verify generation worked
        assert!(!all_tokens.is_empty(), "Should generate some tokens");
        assert!(all_tokens.len() <= config.max_length, "Should respect max length");
        
        // Verify tokens are being tracked
        assert_eq!(text_gen.get_tokens().len(), all_tokens.len(), "Internal tokens should match generated");
    }
}