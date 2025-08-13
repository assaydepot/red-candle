require "spec_helper"

RSpec.describe "Reranker max_length configuration" do
  let(:query) { "What is artificial intelligence?" }
  let(:long_document) { "Artificial intelligence and machine learning. " * 100 }
  
  describe "with different max_length values" do
    it "accepts default max_length (512)" do
      reranker = Candle::Reranker.from_pretrained(
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: "cpu"
      )
      
      results = reranker.rerank(query, [long_document])
      expect(results).to be_an(Array)
      expect(results[0][:score]).to be_a(Float)
    end
    
    it "accepts custom max_length of 256" do
      reranker = Candle::Reranker.from_pretrained(
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: "cpu",
        max_length: 256
      )
      
      results = reranker.rerank(query, [long_document])
      expect(results).to be_an(Array)
      expect(results[0][:score]).to be_a(Float)
    end
    
    it "accepts custom max_length of 128" do
      reranker = Candle::Reranker.from_pretrained(
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: "cpu",
        max_length: 128
      )
      
      results = reranker.rerank(query, [long_document])
      expect(results).to be_an(Array)
      expect(results[0][:score]).to be_a(Float)
    end
    
    it "processes faster with shorter max_length" do
      require 'benchmark'
      
      # Create rerankers with different max_lengths
      reranker_512 = Candle::Reranker.from_pretrained(
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: "cpu",
        max_length: 512
      )
      
      reranker_128 = Candle::Reranker.from_pretrained(
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: "cpu",
        max_length: 128
      )
      
      # Warm up
      reranker_512.rerank(query, ["warmup"])
      reranker_128.rerank(query, ["warmup"])
      
      # Benchmark
      time_512 = Benchmark.realtime do
        reranker_512.rerank(query, [long_document])
      end
      
      time_128 = Benchmark.realtime do
        reranker_128.rerank(query, [long_document])
      end
      
      # 128 should be noticeably faster than 512
      expect(time_128).to be < time_512
      
      # Should be at least 30% faster
      speedup = (time_512 - time_128) / time_512
      expect(speedup).to be > 0.3
    end
  end
  
  describe "backward compatibility" do
    it "works with deprecated .new method" do
      expect { 
        reranker = Candle::Reranker.new(
          model_path: "cross-encoder/ms-marco-MiniLM-L-12-v2",
          device: "cpu",
          max_length: 256
        )
        results = reranker.rerank(query, ["test document"])
        expect(results).to be_an(Array)
      }.to output(/DEPRECATION/).to_stderr
    end
  end
end