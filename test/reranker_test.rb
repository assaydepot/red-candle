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
    assert_in_delta(0.9, ranked_documents[0][1], 0.1)
  end
end
