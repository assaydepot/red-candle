# frozen_string_literal: true

require_relative "test_helper"

class StructuredGenerationTest < Minitest::Test
  def setup
    @device = Candle::Device.cpu
  end

  def test_vocabulary_adapter_loads
    # This test verifies that our Rust vocabulary adapter compiles and loads
    # We'll use it indirectly through a tokenizer
    tokenizer = Candle::Tokenizer.from_pretrained("bert-base-uncased")
    assert tokenizer, "Should load tokenizer"
    
    # Get vocabulary info
    vocab = tokenizer.get_vocab
    assert vocab.is_a?(Hash), "Vocabulary should be a hash"
    assert vocab.size > 0, "Vocabulary should not be empty"
    
    # Check for special tokens
    assert vocab.key?("[SEP]"), "Should have SEP token"
    assert vocab.key?("[PAD]"), "Should have PAD token"
    assert vocab.key?("[UNK]"), "Should have UNK token"
  end
  
  def test_vocabulary_has_expected_structure
    tokenizer = Candle::Tokenizer.from_pretrained("gpt2")
    vocab = tokenizer.get_vocab
    
    # GPT-2 specific checks
    assert vocab.key?("<|endoftext|>"), "GPT-2 should have endoftext token"
    
    # Check token IDs are contiguous (important for our adapter)
    ids = vocab.values.sort
    expected_ids = (0...ids.size).to_a
    
    # Allow for some gaps, but not too many
    missing_ids = expected_ids - ids
    assert missing_ids.size < 100, "Should not have too many gaps in token IDs"
  end
  
  def test_different_tokenizer_types
    # Test with different tokenizer architectures to ensure compatibility
    tokenizers_to_test = [
      "bert-base-uncased",    # BERT style
      "gpt2",                 # GPT style
      "t5-small"              # T5 style
    ]
    
    tokenizers_to_test.each do |model_id|
      begin
        tokenizer = Candle::Tokenizer.from_pretrained(model_id)
        vocab = tokenizer.get_vocab
        
        assert vocab.size > 1000, "#{model_id} should have reasonable vocabulary size"
        
        # Each tokenizer should have some form of special tokens
        special_tokens = vocab.select { |k, _| k.include?("<") || k.include?("[") }
        assert special_tokens.size > 0, "#{model_id} should have special tokens"
        
        # Only output if verbose
        if ENV["VERBOSE"] || ARGV.include?("--verbose")
          puts "✓ #{model_id}: vocab_size=#{vocab.size}, special_tokens=#{special_tokens.size}"
        end
      rescue => e
        skip "Could not load #{model_id}: #{e.message}"
      end
    end
  end
end