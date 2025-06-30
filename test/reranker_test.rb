require "test_helper"

class RerankerTest < Minitest::Test
  def test_rerank
    reranker = Candle::Reranker.new("cross-encoder/ms-marco-MiniLM-L-12-v2")
    query = "What is the capital of France?"
    documents = [
      "The capital of France is Paris.",
      "Berlin is the capital of Germany.",
      "London is the capital of the United Kingdom."
    ]
    ranked_documents = reranker.rerank(query, documents)
    assert_equal(3, ranked_documents.length)
    # Check structure: [document, score, doc_id]
    assert_equal("The capital of France is Paris.", ranked_documents[0][0])
    assert_equal(0, ranked_documents[0][2])  # doc_id should be 0 (first in input)
    # Ensure the French capital document has the highest score (raw logits)
    assert(ranked_documents[0][1] > ranked_documents[1][1])
    assert(ranked_documents[0][1] > ranked_documents[2][1])
  end
  
  def test_new_cuda
    # This should work regardless of whether CUDA is available
    # If CUDA is not available, it should fall back to CPU
    reranker = Candle::Reranker.new_cuda("cross-encoder/ms-marco-MiniLM-L-12-v2")
    query = "Test query"
    documents = ["Test document"]
    ranked_documents = reranker.rerank(query, documents)
    assert_equal(1, ranked_documents.length)
  end
  
  def test_pooling_methods
    reranker = Candle::Reranker.new("cross-encoder/ms-marco-MiniLM-L-12-v2")
    query = "What is the capital of France?"
    documents = ["The capital of France is Paris.", "Berlin is the capital of Germany."]
    
    # Test pooler method (default, most accurate for cross-encoders)
    ranked_documents = reranker.rerank_with_pooling(query, documents, "pooler")
    assert_equal(2, ranked_documents.length)
    assert_equal("The capital of France is Paris.", ranked_documents[0][0])
    assert_equal(0, ranked_documents[0][2])  # doc_id
    
    # Test cls method (should also work well)
    ranked_documents = reranker.rerank_with_pooling(query, documents, "cls")
    assert_equal(2, ranked_documents.length)
    assert_equal("The capital of France is Paris.", ranked_documents[0][0])
    
    # Test mean method (may give different results as it's not the intended pooling for this model)
    ranked_documents = reranker.rerank_with_pooling(query, documents, "mean")
    assert_equal(2, ranked_documents.length)
    # Just verify we get results, not their order, as mean pooling isn't optimal for cross-encoders
  end
end
