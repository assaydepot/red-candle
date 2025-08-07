require "test_helper"

class RerankerTest < Minitest::Test
  def test_rerank
    reranker = Candle::Reranker.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
    query = "What is the capital of France?"
    documents = [
      "The capital of France is Paris.",
      "Berlin is the capital of Germany.",
      "London is the capital of the United Kingdom."
    ]
    ranked_documents = reranker.rerank(query, documents)
    assert_equal(3, ranked_documents.length)
    # Check structure: [document, score, doc_id]
    assert_equal("The capital of France is Paris.", ranked_documents[0][:text])
    assert_equal(0, ranked_documents[0][:doc_id])  # doc_id should be 0 (first in input)
    # Ensure the French capital document has the highest score (raw logits)
    assert(ranked_documents[0][:score] > ranked_documents[1][:score])
    assert(ranked_documents[0][:score] > ranked_documents[2][:score])
  end
  
  def test_pooling_methods
    reranker = Candle::Reranker.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
    query = "What is the capital of France?"
    documents = ["The capital of France is Paris.", "Berlin is the capital of Germany."]
    
    # Test pooler method (default, most accurate for cross-encoders)
    ranked_documents = reranker.rerank(query, documents, pooling_method: "pooler")
    assert_equal(2, ranked_documents.length)
    assert_equal("The capital of France is Paris.", ranked_documents[0][:text])
    assert_equal(0, ranked_documents[0][:doc_id])  # doc_id
    
    # Test cls method (should also work well)
    ranked_documents = reranker.rerank(query, documents, pooling_method: "cls")
    assert_equal(2, ranked_documents.length)
    assert_equal("The capital of France is Paris.", ranked_documents[0][:text])
    assert_equal(0, ranked_documents[0][:doc_id])  # doc_id
    
    # Test mean method (may give different results as it's not the intended pooling for this model)
    ranked_documents = reranker.rerank(query, documents, pooling_method: "mean")
    assert_equal(2, ranked_documents.length)
    # Just verify we get results, not their order, as mean pooling isn't optimal for cross-encoders
  end
end
