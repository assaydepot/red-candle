require "spec_helper"

RSpec.describe "CandleEmbeddingVerification" do
  include DeviceHelpers
  
  let(:model) do
    @model ||= Candle::EmbeddingModel.from_pretrained(
      "sentence-transformers/all-MiniLM-L6-v2", 
      tokenizer: "sentence-transformers/all-MiniLM-L6-v2", 
      model_type: Candle::EmbeddingModelType::MINILM
    )
  end
  
  after(:all) do
    @model = nil
    GC.start
  end

  describe "embedding normalization" do
    it "produces properly normalized embeddings" do
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
        expect(pooled_values.length).to eq(normalized_values.length)
        
        # Verify normalization (L2 norm should be ~1.0)
        l2_norm = Math.sqrt(normalized_values.map { |x| x**2 }.sum)
        expect(l2_norm).to be_within(0.0001).of(1.0)
        
        # Verify embeddings are non-zero
        expect(normalized_values.any? { |v| v.abs > 0.01 }).to be true
        
        if ENV['CANDLE_TEST_VERBOSE']
          puts "Text: '#{text[0..50]}...'"
          puts "  Embedding dimensions: #{normalized_values.length}"
          puts "  L2 norm: #{l2_norm}"
          puts "  First 5 values: #{normalized_values[0...5]}"
        end
      end
    end
  end
  
  describe "embedding similarity" do
    it "produces higher similarity for similar texts" do
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
      
      if ENV['CANDLE_TEST_VERBOSE']
        puts "Similar texts cosine similarity: #{similar_cosine}"
        puts "Different texts cosine similarity: #{different_cosine}"
      end
      
      # Similar texts should have higher similarity than different texts
      expect(similar_cosine).to be > different_cosine
      
      # Similar texts should have reasonably high similarity
      expect(similar_cosine).to be > 0.7
    end
  end
  
  private
  
  def cosine_similarity(vec1, vec2)
    dot_product = vec1.zip(vec2).map { |a, b| a * b }.sum
    dot_product  # Already normalized vectors, so magnitude is 1
  end
end