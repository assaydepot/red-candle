require "spec_helper"

RSpec.describe "Reranker" do
  let(:reranker) do
    @reranker ||= Candle::Reranker.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
  end
  
  # Clear cached model after spec completes
  after(:all) do
    @reranker = nil
    GC.start
  end
  
  describe "#rerank" do
    it "reranks documents based on query relevance" do
      query = "What is the capital of France?"
      documents = [
        "The capital of France is Paris.",
        "Berlin is the capital of Germany.",
        "London is the capital of the United Kingdom."
      ]
      
      ranked_documents = reranker.rerank(query, documents)
      
      expect(ranked_documents.length).to eq(3)
      # Check structure: [document, score, doc_id]
      expect(ranked_documents[0][:text]).to eq("The capital of France is Paris.")
      expect(ranked_documents[0][:doc_id]).to eq(0)  # doc_id should be 0 (first in input)
      # Ensure the French capital document has the highest score (raw logits)
      expect(ranked_documents[0][:score]).to be > ranked_documents[1][:score]
      expect(ranked_documents[0][:score]).to be > ranked_documents[2][:score]
    end
  end
  
  describe "pooling methods" do
    let(:query) { "What is the capital of France?" }
    let(:documents) { ["The capital of France is Paris.", "Berlin is the capital of Germany."] }
    
    it "works with pooler method (default)" do
      # Test pooler method (default, most accurate for cross-encoders)
      ranked_documents = reranker.rerank(query, documents, pooling_method: "pooler")
      
      expect(ranked_documents.length).to eq(2)
      expect(ranked_documents[0][:text]).to eq("The capital of France is Paris.")
      expect(ranked_documents[0][:doc_id]).to eq(0)  # doc_id
    end
    
    it "works with cls method" do
      # Test cls method (should also work well)
      ranked_documents = reranker.rerank(query, documents, pooling_method: "cls")
      
      expect(ranked_documents.length).to eq(2)
      expect(ranked_documents[0][:text]).to eq("The capital of France is Paris.")
      expect(ranked_documents[0][:doc_id]).to eq(0)  # doc_id
    end
    
    it "works with mean method" do
      # Test mean method (may give different results as it's not the intended pooling for this model)
      ranked_documents = reranker.rerank(query, documents, pooling_method: "mean")
      
      expect(ranked_documents.length).to eq(2)
      # Just verify we get results, not their order, as mean pooling isn't optimal for cross-encoders
    end
  end
end