require_relative "test_helper"

class CandleEmbeddingVerificationTest < Minitest::Test
  include DeviceTestHelper

  def test_embedding_normalization
    model = Candle::EmbeddingModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", tokenizer_path: "sentence-transformers/all-MiniLM-L6-v2", model_type: Candle::EmbeddingModelType::MINILM)
    
    texts = [
      "The quick brown fox jumps over the lazy dog",
      "Machine learning is a subset of artificial intelligence",
      "Ruby is a dynamic programming language"
    ]
    
    texts.each do |text|
      # Get raw embeddings
      raw_tensor = model.embeddings(text)
      
      # Test pooled embedding
      pooled = model.pool_embedding(raw_tensor)
      pooled_values = pooled.values
      
      # Test pooled and normalized embedding
      pooled_normalized = model.pool_and_normalize_embedding(raw_tensor)
      normalized_values = pooled_normalized.values
      
      # Verify dimensions match
      assert_equal pooled_values.length, normalized_values.length,
                   "Pooled and normalized embeddings should have same dimensions"
      
      # Verify normalization (L2 norm should be ~1.0)
      l2_norm = Math.sqrt(normalized_values.map { |x| x**2 }.sum)
      assert_in_delta 1.0, l2_norm, 0.0001,
                      "L2 norm of normalized embedding should be 1.0, got #{l2_norm}"
      
      # Verify embeddings are non-zero
      assert normalized_values.any? { |v| v.abs > 0.01 },
             "Embedding values should not all be near zero"
      
      puts "Text: '#{text[0..50]}...'" if ENV['CANDLE_TEST_VERBOSE']
      puts "  Embedding dimensions: #{normalized_values.length}" if ENV['CANDLE_TEST_VERBOSE']
      puts "  L2 norm: #{l2_norm}" if ENV['CANDLE_TEST_VERBOSE']
      puts "  First 5 values: #{normalized_values[0...5]}" if ENV['CANDLE_TEST_VERBOSE']
    end
  end
  
  def test_embedding_similarity
    model = Candle::EmbeddingModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", tokenizer_path: "sentence-transformers/all-MiniLM-L6-v2", model_type: Candle::EmbeddingModelType::MINILM)
    
    # Similar sentences should have high cosine similarity
    similar_texts = [
      "The cat sat on the mat",
      "A cat was sitting on a mat"
    ]
    
    # Different sentences should have lower cosine similarity
    different_texts = [
      "The weather is sunny today",
      "I love programming in Ruby"
    ]
    
    # Get embeddings for similar texts
    emb1 = model.pool_and_normalize_embedding(model.embeddings(similar_texts[0])).values
    emb2 = model.pool_and_normalize_embedding(model.embeddings(similar_texts[1])).values
    similar_cosine = cosine_similarity(emb1, emb2)
    
    # Get embeddings for different texts
    emb3 = model.pool_and_normalize_embedding(model.embeddings(different_texts[0])).values
    emb4 = model.pool_and_normalize_embedding(model.embeddings(different_texts[1])).values
    different_cosine = cosine_similarity(emb3, emb4)
    
    puts "Similar texts cosine similarity: #{similar_cosine}" if ENV['CANDLE_TEST_VERBOSE']
    puts "Different texts cosine similarity: #{different_cosine}" if ENV['CANDLE_TEST_VERBOSE']
    
    # Similar texts should have higher similarity than different texts
    assert similar_cosine > different_cosine,
           "Similar texts should have higher cosine similarity than different texts"
    
    # Similar texts should have reasonably high similarity
    assert similar_cosine > 0.7,
           "Similar texts should have cosine similarity > 0.7, got #{similar_cosine}"
  end
  
  private
  
  def cosine_similarity(vec1, vec2)
    dot_product = vec1.zip(vec2).map { |a, b| a * b }.sum
    dot_product  # Already normalized vectors, so magnitude is 1
  end
end