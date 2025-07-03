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
    embeddings = model.embeddings(string) # shape: [1, n_tokens, hidden_size]
    pooled = model.embedding(string, pooling_method: "pooled") # shape: [1, hidden_size]

    # Recreate pooling logic from Rust: mean over tokens axis (axis 1)
    # Use Candle::Tensor shape and to_a for backend-agnostic pooling
    shape = embeddings.shape # [1, n_tokens, hidden_size]
    n_tokens = shape[1]
    hidden_size = shape[2]
    arr = embeddings.to_a # [ [ [token1], [token2], ... ] ]

    # arr[0] is [n_tokens, hidden_size]
    sum = Array.new(hidden_size, 0.0)
    arr[0].each do |token_vec|
      token_vec.each_with_index { |v, i| sum[i] += v }
    end
    mean = sum.map { |v| v / n_tokens.to_f }
    custom_pooled = [mean]

    # Assert each element is close within a tolerance
    pooled_arr = pooled.first.to_a
    custom_arr = custom_pooled.first.to_a
    tolerance = 1e-6
    custom_arr.zip(pooled_arr).each_with_index do |(a, b), i|
      assert_in_delta a, b, tolerance, "Mismatch at index #{i}: #{a} vs #{b}"
    end
  end
end
