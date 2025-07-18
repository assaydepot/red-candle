require_relative "test_helper"

class EmbeddingModelPoolingTest < Minitest::Test
  def setup
    puts "SETUP"
    @model = Candle::EmbeddingModel.new
    @string = "The quick brown fox jumps over the lazy dog."
    @embeddings = @model.embeddings(@string)
  end

  def test_pool_embedding_matches_manual
    pooled_ruby = @model.pool_embedding(@embeddings)
    # Manually pool by averaging over sequence dimension (1)
    pooled_manual = @embeddings.mean(1)
    pooled_ruby.first.to_a.each_index do |i|
      assert_in_delta pooled_ruby.first.to_a[i], pooled_manual.first.to_a[i], 1e-5
    end
  end

  def test_pool_and_normalize_embedding_matches_manual
    pooled_norm_ruby = @model.embedding(@string, pooling_method: "pooled_normalized")
    # Manually pool and then normalize
    pooled_manual = @embeddings.mean(1)
    norm = pooled_manual / Math.sqrt(pooled_manual.sqr.sum(1).values.first)
    pooled_norm_ruby.first.to_a.each_index do |i|
      assert_in_delta pooled_norm_ruby.first.to_a[i], norm.first.to_a[i], 1e-5
    end
  end

  def test_pool_cls_embedding
    cls_ruby = @model.embedding(@string, pooling_method: "cls")
    # Manually extract CLS token (index 0)
    # For shape [batch, seq, hidden], get first token of first batch
    cls_manual = @embeddings.get(0).get(0)
    cls_ruby.first.to_a.each_index do |i|
      assert_in_delta cls_ruby.first.to_a[i], cls_manual.to_a[i], 1e-5
    end
  end

  def test_embedding_calls_correct_pooling
    pooled = @model.embedding(@string, pooling_method: "pooled")
    pooled_norm = @model.embedding(@string, pooling_method: "pooled_normalized")
    cls = @model.embedding(@string, pooling_method: "cls")
    refute_equal pooled.first.to_a, pooled_norm.first.to_a
    refute_equal pooled.first.to_a, cls.first.to_a
    refute_equal pooled_norm.first.to_a, cls.first.to_a
  end

  def test_embedding_default_is_pooled
    pooled = @model.embedding(@string, pooling_method: "pooled_normalized")
    default = @model.embedding(@string)
    assert_equal pooled.first.to_a, default.first.to_a
  end
end
