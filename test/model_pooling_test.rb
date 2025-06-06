require_relative "test_helper"

class ModelPoolingTest < Minitest::Test
  def setup
    @model = Candle::Model.new
    @string = "The quick brown fox jumps over the lazy dog."
    @embeddings = @model.embeddings(@string)
  end

  def test_pool_embedding_matches_manual
    pooled_ruby = @model.pool_embedding(@embeddings)
    # Manually pool by averaging over sequence dimension (1)
    pooled_manual = @embeddings.mean(1)
    assert_in_delta pooled_ruby.first.to_a[0], pooled_manual.first.to_a[0], 1e-5
  end

  def test_pool_and_normalize_embedding_matches_manual
    pooled_norm_ruby = @model.pool_and_normalize_embedding(@embeddings)
    # Manually pool and then normalize
    pooled_manual = @embeddings.mean(1)
    norm = pooled_manual / Math.sqrt(pooled_manual.sqr.sum(1).values.first)
    assert_in_delta pooled_norm_ruby.first.to_a[0], norm.first.to_a[0], 1e-5
  end

  def test_pool_cls_embedding
    cls_ruby = @model.pool_cls_embedding(@embeddings)
    # Manually extract CLS token (index 0)
    # For shape [batch, seq, hidden], get first token of first batch
    cls_manual = @embeddings.get(0).get(0)
    assert_in_delta cls_ruby.first.to_a[0], cls_manual.to_a[0], 1e-5
  end

  def test_embedding_calls_correct_pooling
    pooled = @model.embedding(@string, "pooled")
    pooled_norm = @model.embedding(@string, "pooled_normalized")
    cls = @model.embedding(@string, "cls")
    refute_equal pooled.first.to_a, pooled_norm.first.to_a
    refute_equal pooled.first.to_a, cls.first.to_a
    refute_equal pooled_norm.first.to_a, cls.first.to_a
  end

  def test_embedding_default_is_pooled
    pooled = @model.embedding(@string, "pooled")
    default = @model.embedding(@string)
    assert_equal pooled.first.to_a, default.first.to_a
  end
end
