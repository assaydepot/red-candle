require_relative "test_helper"

class EmbeddingModelTest < Minitest::Test
  def test_pooled_embeddings_shape
    model = Candle::EmbeddingModel.new
    string = "Hi there"
    embeddings = model.embeddings(string)
    pooled = model.pool_and_normalize_embedding(embeddings)
    assert_equal pooled.shape, [1, 768]
  end

  def test_pooled_embeddings
    model = Candle::EmbeddingModel.new
    string = "Hi there"
    embeddings = model.embeddings(string)
    pooled = model.pool_embedding(embeddings)
    embedding = model.embedding(string)
    assert_equal pooled.first.to_a, embedding.first.to_a
  end
end
