require_relative "test_helper"
require "informers"

class CandleInformersComparisonTest < Minitest::Test
  include DeviceTestHelper

  # Tolerance for floating point comparison
  FLOAT_TOLERANCE = 1e-4
  
  def setup
    skip("Skipping comparison tests - set CANDLE_RUN_COMPARISON_TESTS=true to run") unless ENV['CANDLE_RUN_COMPARISON_TESTS'] == 'true'
  end

  def test_reranker_comparison
    model_id = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    query = "How many people live in London?"
    docs = [
      "London is known for its financial district",
      "Around 9 Million people live in London"
    ]

    # Test with Informers
    informers_model = Informers.pipeline("reranking", model_id)
    informers_result = informers_model.(query, docs, return_documents: false)
    # Extract just the scores from informers result
    informers_scores = informers_result.map { |r| r[:score] }
    
    # Test with Candle (default device)
    candle_reranker = Candle::Reranker.new(model_path: model_id)
    
    candle_result = candle_reranker.rerank(
      query, 
      docs
    )
    # Extract just the scores from candle result
    candle_scores = candle_result.map { |r| r[:score] }

    # Compare scores elementwise
    assert_equal informers_scores.length, candle_scores.length, "Reranker score arrays have different lengths"
    informers_scores.zip(candle_scores).each_with_index do |(a, b), i|
      assert (a - b).abs <= FLOAT_TOLERANCE, "Reranker score at index #{i} differs: informers=#{a} candle=#{b}"
    end
  end

  def test_embedding_model_comparison
    sentences = ["How is the weather today?", "What is the current weather like today?"]

    # Test with Informers
    informers_model = Informers.pipeline("embedding", "jinaai/jina-embeddings-v2-base-en", model_file_name: "../model")
    informers_embeddings = informers_model.(sentences)

    # Test with Candle
    candle_model = Candle::EmbeddingModel.new(model_path: "jinaai/jina-embeddings-v2-base-en")
    candle_embeddings = sentences.collect { |sentence| candle_model.embedding(sentence).values }

    assert_equal informers_embeddings.length, candle_embeddings.length, "Embedding arrays have different lengths"
    informers_embeddings.zip(candle_embeddings).each_with_index do |(informer_embedding, candle_embedding), i|
      informer_embedding.zip(candle_embedding).each_with_index do |(a, b), j|
        assert (a.to_f - b.to_f).abs <= FLOAT_TOLERANCE, "Embedding value at index #{i} #{j} differs: informers=#{a} candle=#{b}"
      end
    end
  end

  private

  def cosine_similarity(vec1, vec2)
    raise "Vectors have different lengths" if vec1.length != vec2.length
    
    dot_product = 0.0
    norm1 = 0.0
    norm2 = 0.0
    
    vec1.zip(vec2).each do |a, b|
      dot_product += a * b
      norm1 += a ** 2
      norm2 += b ** 2
    end
    
    dot_product / (Math.sqrt(norm1) * Math.sqrt(norm2))
  end
end