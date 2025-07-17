# frozen_string_literal: true

require_relative "test_helper"

class TokenizerTest < Minitest::Test
  def test_tokenizer_from_pretrained
    tokenizer = Candle::Tokenizer.from_pretrained("bert-base-uncased")
    assert_instance_of Candle::Tokenizer, tokenizer
  end

  def test_encode_decode
    tokenizer = Candle::Tokenizer.from_pretrained("bert-base-uncased")
    
    text = "Hello, world!"
    token_ids = tokenizer.encode(text)
    assert_instance_of Array, token_ids
    assert_equal token_ids.class, Array
    assert token_ids.all? { |id| id.is_a?(Integer) }
    
    # Decode back
    decoded = tokenizer.decode(token_ids)
    assert_instance_of String, decoded
    # BERT tokenizer lowercases and may add special tokens
    assert decoded.downcase.include?("hello")
    assert decoded.downcase.include?("world")
  end

  def test_encode_without_special_tokens
    tokenizer = Candle::Tokenizer.from_pretrained("bert-base-uncased")
    
    text = "test"
    tokens_with_special = tokenizer.encode(text)
    tokens_without_special = tokenizer.encode(text, add_special_tokens: false)
    
    # Without special tokens should be shorter
    assert tokens_without_special.length < tokens_with_special.length
  end

  def test_encode_batch
    tokenizer = Candle::Tokenizer.from_pretrained("bert-base-uncased")
    
    texts = ["Hello", "World", "Test"]
    batch_tokens = tokenizer.encode_batch(texts)
    
    assert_instance_of Array, batch_tokens
    assert_equal 3, batch_tokens.length
    batch_tokens.each do |tokens|
      assert_instance_of Array, tokens
      assert tokens.all? { |id| id.is_a?(Integer) }
    end
  end

  def test_vocab_operations
    tokenizer = Candle::Tokenizer.from_pretrained("bert-base-uncased")
    
    # Test vocab size
    vocab_size = tokenizer.vocab_size
    assert vocab_size > 0
    
    # Test get_vocab
    vocab = tokenizer.get_vocab
    assert_instance_of Hash, vocab
    assert vocab.size > 0
    
    # Check some known tokens
    assert vocab.key?("[CLS]")
    assert vocab.key?("[SEP]")
    assert vocab.key?("[PAD]")
  end

  def test_id_to_token
    tokenizer = Candle::Tokenizer.from_pretrained("bert-base-uncased")
    
    # Get a token ID from vocab
    vocab = tokenizer.get_vocab
    token_id = vocab["[CLS]"]
    
    # Convert back to token
    token = tokenizer.id_to_token(token_id)
    assert_equal "[CLS]", token
  end

  def test_special_tokens
    tokenizer = Candle::Tokenizer.from_pretrained("bert-base-uncased")
    
    special_tokens = tokenizer.get_special_tokens
    assert_instance_of Hash, special_tokens
    
    # BERT should have these special tokens
    assert special_tokens.key?("cls_token")
    assert special_tokens.key?("sep_token")
    assert special_tokens.key?("pad_token")
  end

  def test_with_padding
    tokenizer = Candle::Tokenizer.from_pretrained("bert-base-uncased")
    
    # Create a tokenizer with padding enabled
    padded_tokenizer = tokenizer.with_padding(length: 10)
    assert_instance_of Candle::Tokenizer, padded_tokenizer
    
    # The padded tokenizer should be a new instance
    refute_equal tokenizer.object_id, padded_tokenizer.object_id
  end

  def test_with_truncation
    tokenizer = Candle::Tokenizer.from_pretrained("bert-base-uncased")
    
    # Create a tokenizer with truncation enabled
    truncated_tokenizer = tokenizer.with_truncation(512)
    assert_instance_of Candle::Tokenizer, truncated_tokenizer
    
    # The truncated tokenizer should be a new instance
    refute_equal tokenizer.object_id, truncated_tokenizer.object_id
  end

  def test_tokenizer_from_llm
    llm = Candle::LLM.new(
      model_id: "hf-internal-testing/tiny-random-LlamaForCausalLM",
      device: Candle::Device::Cpu
    )
    
    tokenizer = llm.tokenizer
    assert_instance_of Candle::Tokenizer, tokenizer
    
    # Test basic functionality
    text = "Hello"
    tokens = tokenizer.encode(text)
    assert_instance_of Array, tokens
    assert tokens.length > 0
  end

  def test_tokenizer_from_embedding_model
    model = Candle::EmbeddingModel.new(
      model_path: "sentence-transformers/all-MiniLM-L6-v2",
      tokenizer_path: "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    tokenizer = model.tokenizer
    assert_instance_of Candle::Tokenizer, tokenizer
    
    # Test basic functionality
    text = "Test embedding"
    tokens = tokenizer.encode(text)
    assert_instance_of Array, tokens
    assert tokens.length > 0
  end

  def test_tokenizer_from_reranker
    reranker = Candle::Reranker.new(
      model_id: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    
    tokenizer = reranker.tokenizer
    assert_instance_of Candle::Tokenizer, tokenizer
    
    # Test basic functionality
    text = "Query text"
    tokens = tokenizer.encode(text)
    assert_instance_of Array, tokens
    assert tokens.length > 0
  end

  def test_inspect_method
    tokenizer = Candle::Tokenizer.from_pretrained("bert-base-uncased")
    
    inspect_str = tokenizer.inspect
    assert_instance_of String, inspect_str
    assert inspect_str.include?("Candle::Tokenizer")
    assert inspect_str.include?("vocab_size")
  end

  def test_to_s_method
    tokenizer = Candle::Tokenizer.from_pretrained("bert-base-uncased")
    
    str = tokenizer.to_s
    assert_instance_of String, str
    assert str.include?("Candle::Tokenizer")
  end
end