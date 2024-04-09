require_relative "test_helper"

class CandleTest < Minitest::Test
  def test_that_it_has_a_version_number
    refute_nil ::Candle::VERSION
  end

  def test_something_useful
    t = Candle::Tensor.new([3.0, 1, 4, 1, 5, 9, 2, 6], :f32)
    assert_instance_of(Candle::Tensor, t)
    assert_equal [8], t.shape
    t = t.reshape([2, 4])
    assert_equal [2, 4], t.shape
    t = t.t
    assert_equal [4, 2], t.shape

    t = Candle::Tensor.randn([5, 3])
    assert_equal [5, 3], t.shape
    assert_instance_of Candle::DType, t.dtype
  end
end
