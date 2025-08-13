require "spec_helper"

RSpec.describe "Reranker truncation" do
  before(:all) do
    @reranker = Candle::Reranker.from_pretrained(
      "cross-encoder/ms-marco-MiniLM-L-12-v2",
      device: "cpu"
    )
  end
  
  describe "handling long documents" do
    let(:query) { "What is machine learning?" }
    
    it "processes short documents normally" do
      short_doc = "Machine learning is a type of artificial intelligence."
      results = @reranker.rerank(query, [short_doc])
      
      expect(results).to be_an(Array)
      expect(results.length).to eq(1)
      expect(results[0][:score]).to be_a(Float)
      expect(results[0][:text]).to eq(short_doc)
    end
    
    it "handles documents with 500+ words without error" do
      # Create a document with ~500 words
      long_doc = "Machine learning is a field of artificial intelligence. " * 50
      
      results = @reranker.rerank(query, [long_doc])
      
      expect(results).to be_an(Array)
      expect(results.length).to eq(1)
      expect(results[0][:score]).to be_a(Float)
    end
    
    it "handles extremely long documents without error" do
      # Create a document with ~5000 words
      very_long_doc = "Machine learning enables computers to learn from data without being explicitly programmed. " * 500
      
      results = @reranker.rerank(query, [very_long_doc])
      
      expect(results).to be_an(Array)
      expect(results.length).to eq(1)
      expect(results[0][:score]).to be_a(Float)
    end
    
    it "processes multiple long documents in batch" do
      docs = [
        "Short document about ML.",
        "Machine learning is important. " * 100,  # ~300 words
        "AI and ML are related fields. " * 200,    # ~800 words
        "Deep learning is a subset of machine learning. " * 300  # ~1800 words
      ]
      
      results = @reranker.rerank(query, docs)
      
      expect(results).to be_an(Array)
      expect(results.length).to eq(4)
      results.each do |result|
        expect(result[:score]).to be_a(Float)
        expect(result[:doc_id]).to be_a(Integer)
      end
    end
    
    it "completes in reasonable time even with very long documents" do
      require 'benchmark'
      
      very_long_doc = "Machine learning is important. " * 1000  # ~3000 words
      
      time = Benchmark.realtime do
        @reranker.rerank(query, [very_long_doc])
      end
      
      # Should complete in under 2 seconds even with very long document
      expect(time).to be < 2.0
    end
  end
  
  describe "truncation behavior" do
    it "gives similar scores for a document and its extended version" do
      query = "What is the capital of France?"
      base_doc = "Paris is the capital city of France. It is known for the Eiffel Tower. " * 10
      extended_doc = base_doc + ("Additional information about French history and culture. " * 100)
      
      base_results = @reranker.rerank(query, [base_doc])
      extended_results = @reranker.rerank(query, [extended_doc])
      
      base_score = base_results[0][:score]
      extended_score = extended_results[0][:score]
      
      # Scores should be very similar since extra content is truncated
      expect((base_score - extended_score).abs).to be < 0.1
    end
  end
end