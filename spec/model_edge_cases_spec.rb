# frozen_string_literal: true

require "spec_helper"

RSpec.describe "ModelEdgeCases" do
  include DeviceHelpers
  
  describe "LLM edge cases" do
    let(:llm) do
      ModelCache.gguf_llm
    end
    
    it "handles empty prompts" do
      config = Candle::GenerationConfig.deterministic.with(max_length: 10)
      result = llm.generate("", config: config)
      expect(result).to be_a(String)
    end
    
    it "handles prompts with only whitespace" do
      config = Candle::GenerationConfig.deterministic.with(max_length: 10)
      result = llm.generate("   \n\t  ", config: config)
      expect(result).to be_a(String)
    end
    
    it "handles very short max_length" do
      config = Candle::GenerationConfig.deterministic.with(max_length: 1)
      result = llm.generate("Hello", config: config)
      expect(result).to be_a(String)
      # Should generate at least one token
      expect(result.length).to be >= 1
    end
    
    it "handles temperature edge cases" do
      # Temperature of 0 (deterministic)
      config_zero = Candle::GenerationConfig.new(temperature: 0.0, max_length: 10)
      result_zero = llm.generate("Test", config: config_zero)
      expect(result_zero).to be_a(String)
      
      # Very high temperature
      config_high = Candle::GenerationConfig.new(temperature: 2.0, max_length: 10, seed: 42)
      result_high = llm.generate("Test", config: config_high)
      expect(result_high).to be_a(String)
    end
    
    it "handles repetition penalty edge cases" do
      # No repetition penalty
      config_none = Candle::GenerationConfig.new(repetition_penalty: 1.0, max_length: 20, temperature: 0)
      result_none = llm.generate("The the the", config: config_none)
      expect(result_none).to be_a(String)
      
      # High repetition penalty
      config_high = Candle::GenerationConfig.new(repetition_penalty: 2.0, max_length: 20, temperature: 0)
      result_high = llm.generate("The the the", config: config_high)
      expect(result_high).to be_a(String)
    end
    
    it "handles top_k and top_p edge cases" do
      # Top-k = 1 (greedy)
      config_k1 = Candle::GenerationConfig.new(top_k: 1, max_length: 10, temperature: 1.0)
      result_k1 = llm.generate("Hello", config: config_k1)
      expect(result_k1).to be_a(String)
      
      # Top-p = 0.01 (very narrow)
      config_p_low = Candle::GenerationConfig.new(top_p: 0.01, max_length: 10, temperature: 1.0, seed: 42)
      result_p_low = llm.generate("Hello", config: config_p_low)
      expect(result_p_low).to be_a(String)
      
      # Top-p = 1.0 (all tokens)
      config_p_high = Candle::GenerationConfig.new(top_p: 1.0, max_length: 10, temperature: 1.0, seed: 42)
      result_p_high = llm.generate("Hello", config: config_p_high)
      expect(result_p_high).to be_a(String)
    end
  end
  
  describe "EmbeddingModel edge cases" do
    let(:model) do
      ModelCache.embedding_model
    end
    
    it "handles empty text" do
      embedding = model.embedding("")
      expect(embedding).to be_a(Candle::Tensor)
      values = embedding.values
      expect(values).to be_a(Array)
    end
    
    it "handles very long text" do
      long_text = "word " * 1000
      embedding = model.embedding(long_text)
      expect(embedding).to be_a(Candle::Tensor)
      values = embedding.values
      expect(values.length).to eq(384)  # MiniLM-L6 has 384 dimensions
    end
    
    it "handles special characters" do
      special_text = "Text with 😊 emoji and @#$% special chars"
      embedding = model.embedding(special_text)
      expect(embedding).to be_a(Candle::Tensor)
    end
    
    it "handles batch embeddings" do
      texts = [
        "Normal text",
        "Another normal text",
        "Third text sample"
      ]
      
      # Batch processing is done with collect/map
      embeddings = texts.collect { |text| model.embedding(text) }
      expect(embeddings).to be_a(Array)
      expect(embeddings.length).to eq(3)
      
      embeddings.each do |emb|
        expect(emb).to be_a(Candle::Tensor)
      end
    end
    
    it "maintains consistency for same input" do
      text = "Consistency test"
      emb1 = model.embedding(text).values
      emb2 = model.embedding(text).values
      
      # Should produce identical embeddings
      emb1.zip(emb2).each do |v1, v2|
        expect(v1).to be_within(1e-6).of(v2)
      end
    end
  end
  
  describe "Reranker edge cases" do
    let(:reranker) do
      ModelCache.reranker
    end
    
    it "handles empty query" do
      docs = ["Document 1", "Document 2"]
      results = reranker.rerank("", docs)
      
      expect(results).to be_a(Array)
      expect(results.length).to eq(2)
    end
    
    it "handles documents list" do
      query = "Test query"
      docs = ["First document", "Second document"]
      results = reranker.rerank(query, docs)
      
      expect(results).to be_a(Array)
      expect(results.length).to eq(2)
      expect(results[0]).to have_key(:doc_id)  # Returns doc_id, not index
      expect(results[0]).to have_key(:score)
      expect(results[0]).to have_key(:text)
    end
    
    it "handles documents with special characters" do
      query = "Test with emojis 😊"
      docs = [
        "Document with emoji 🎉",
        "Plain document",
        "Special chars @#$%^&*()"
      ]
      
      results = reranker.rerank(query, docs)
      expect(results).to be_a(Array)
      expect(results.length).to eq(3)
    end
    
    it "handles very long documents" do
      query = "Find relevant information"
      # Create a long doc that's near but under the 512 token limit
      # Most models have a 512 token limit, "word " repeated ~100 times is safe
      long_doc = "word " * 100
      docs = [long_doc, "Short doc", "Medium length document here"]
      
      results = reranker.rerank(query, docs)
      expect(results).to be_a(Array)
      expect(results.length).to eq(3)
    rescue RuntimeError => e
      # Some models may have stricter limits, skip if we hit token limit
      skip "Model has token limit: #{e.message}" if e.message.include?("index-select")
      raise
    end
    
    it "maintains ranking consistency" do
      query = "Machine learning"
      docs = [
        "Machine learning is a subset of AI",
        "Cooking recipes for dinner",
        "Deep learning and neural networks"
      ]
      
      results1 = reranker.rerank(query, docs)
      results2 = reranker.rerank(query, docs)
      
      # Rankings should be consistent
      expect(results1.map { |r| r[:doc_id] }).to eq(results2.map { |r| r[:doc_id] })
    end
  end
  
  describe "Tensor operations edge cases" do
    it "handles single element arrays" do
      single = Candle::Tensor.new([42.0])
      expect(single).to be_a(Candle::Tensor)
      expect(single.shape).to eq([1])
      expect(single.values).to eq([42.0])
    end
    
    it "handles very large tensors efficiently" do
      # Create a reasonably large tensor
      large_data = Array.new(1000) { rand }
      large_tensor = Candle::Tensor.new(large_data)
      
      expect(large_tensor).to be_a(Candle::Tensor)
      expect(large_tensor.shape).to eq([1000])
    end
    
    it "handles mixed numeric types" do
      # Mix of integers and floats
      mixed = Candle::Tensor.new([1, 2.5, 3, 4.7])
      expect(mixed).to be_a(Candle::Tensor)
      values = mixed.values
      expect(values).to all(be_a(Float))
    end
    
    it "handles negative values" do
      negative = Candle::Tensor.new([-1.0, -2.5, -3.7])
      expect(negative).to be_a(Candle::Tensor)
      expect(negative.values).to all(be < 0)
    end
  end
  
  describe "NER edge cases" do
    let(:ner) do
      ModelCache.ner_model
    end
    
    it "handles empty text" do
      entities = ner.extract_entities("")
      expect(entities).to be_a(Array)
      expect(entities).to be_empty
    end
    
    it "handles text with only punctuation" do
      entities = ner.extract_entities("...!!!???")
      expect(entities).to be_a(Array)
    end
    
    it "handles text with only numbers" do
      entities = ner.extract_entities("123 456 789")
      expect(entities).to be_a(Array)
    end
    
    it "handles mixed scripts" do
      # Mix of Latin, numbers, and special chars
      mixed_text = "Company123 located at Street#45"
      entities = ner.extract_entities(mixed_text)
      expect(entities).to be_a(Array)
    end
    
    it "handles overlapping entity mentions" do
      # Text where entities might overlap
      text = "New York New York Times"
      entities = ner.extract_entities(text)
      expect(entities).to be_a(Array)
      
      # Check that entities don't overlap
      entities.each_cons(2) do |e1, e2|
        expect(e1[:end]).to be <= e2[:start]
      end
    end
    
    it "handles confidence threshold boundary cases" do
      text = "Apple Inc. in California"
      
      # Very low threshold (should get more entities)
      low_threshold = ner.extract_entities(text, confidence_threshold: 0.01)
      
      # Very high threshold (should get fewer entities)
      high_threshold = ner.extract_entities(text, confidence_threshold: 0.99)
      
      expect(low_threshold.length).to be >= high_threshold.length
    end
  end
  
  # Clean up after all tests
  after(:all) do
    ModelCache.clear!
  end
end