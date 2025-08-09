# frozen_string_literal: true

require_relative "test_helper"

class TokenizerTokensTest < Minitest::Test
  def setup
    @tokenizer = Candle::Tokenizer.from_pretrained("bert-base-uncased")
  end

  def test_encode_to_tokens
    text = "Hello, world!"
    tokens = @tokenizer.encode_to_tokens(text)
    
    assert_instance_of Array, tokens
    assert tokens.all? { |t| t.is_a?(String) }
    
    # BERT should produce something like ["[CLS]", "hello", ",", "world", "!", "[SEP]"]
    assert_includes tokens, "[CLS]"
    assert_includes tokens, "[SEP]"
    assert tokens.any? { |t| t.include?("hello") }
    assert tokens.any? { |t| t.include?("world") }
  end

  def test_encode_to_tokens_without_special
    text = "Hello, world!"
    tokens = @tokenizer.encode_to_tokens(text, add_special_tokens: false)
    
    # Without special tokens
    refute_includes tokens, "[CLS]"
    refute_includes tokens, "[SEP]"
    assert tokens.any? { |t| t.include?("hello") }
  end

  def test_encode_with_tokens
    text = "Hello, world!"
    result = @tokenizer.encode_with_tokens(text)
    
    assert_instance_of Hash, result
    assert result.key?(:ids)
    assert result.key?(:tokens)
    
    ids = result[:ids]
    tokens = result[:tokens]
    
    assert_equal ids.length, tokens.length
    assert ids.all? { |id| id.is_a?(Integer) }
    assert tokens.all? { |t| t.is_a?(String) }
    
    # Verify alignment
    tokens.each_with_index do |token, i|
      id = ids[i]
      # Each token should decode to itself
      assert_equal token, @tokenizer.id_to_token(id)
    end
  end

  def test_encode_batch_to_tokens
    texts = ["Hello world", "How are you?", "Testing batch"]
    token_batches = @tokenizer.encode_batch_to_tokens(texts)
    
    assert_instance_of Array, token_batches
    assert_equal 3, token_batches.length
    
    token_batches.each do |tokens|
      assert_instance_of Array, tokens
      assert tokens.all? { |t| t.is_a?(String) }
      assert_includes tokens, "[CLS]"
      assert_includes tokens, "[SEP]"
    end
  end

  def test_subword_tokenization
    # Test a word that likely gets split into subwords
    text = "unbelievable"
    tokens = @tokenizer.encode_to_tokens(text, add_special_tokens: false)
    
    # BERT often splits longer words
    # Could be something like ["un", "##believ", "##able"]
    assert tokens.length >= 1
    
    # Join tokens and verify they reconstruct (roughly) to original
    joined = tokens.join("").gsub("##", "")
    assert joined.downcase.include?("unbeliev")
  end

  def test_tokenization_visualization
    # This demonstrates why token strings are useful
    text = "The quick brown fox jumps over the lazy dog."
    
    ids = @tokenizer.encode(text)
    tokens = @tokenizer.encode_to_tokens(text)
    
    # Create a visual mapping
    mapping = ids.zip(tokens)
    
    assert_equal ids.length, tokens.length
    
    # Each ID corresponds to a token
    mapping.each do |id, token|
      assert_equal token, @tokenizer.id_to_token(id)
    end
  end

  def test_special_characters_tokenization
    text = "Email: user@example.com, Price: $99.99"
    tokens = @tokenizer.encode_to_tokens(text, add_special_tokens: false)
    
    # Should tokenize special characters
    assert tokens.any? { |t| t.include?("@") || t == "@" }
    assert tokens.any? { |t| t.include?("$") || t == "$" }
    assert tokens.any? { |t| t.include?(".") || t == "." }
  end

  def test_unicode_tokenization
    text = "Hello 世界 🌍"
    tokens = @tokenizer.encode_to_tokens(text, add_special_tokens: false)
    
    assert tokens.length > 0
    # Unicode might be split into multiple tokens or preserved
    # depending on the tokenizer's vocabulary
  end
end