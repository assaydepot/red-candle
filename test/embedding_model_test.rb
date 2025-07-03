require_relative "test_helper"

class EmbeddingModelTest < Minitest::Test
  def test_pooled_embeddings_shape
    model = Candle::EmbeddingModel.new
    string = "Hi there"
    embeddings = model.embeddings(string)
    pooled = model.embedding(string, pooling_method: "pooled_normalized")
    assert_equal pooled.shape, [1, 768]
  end

  def test_pooled_embeddings
    model = Candle::EmbeddingModel.new
    string = "Hi there"
    embeddings = model.embeddings(string)
    pooled = model.embedding(string, pooling_method: "pooled")
    assert_equal pooled.first.to_a, embedding.first.to_a
  end
end
