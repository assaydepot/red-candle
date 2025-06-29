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
    assert_equal("The capital of France is Paris.", ranked_documents[0][0])
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
end
