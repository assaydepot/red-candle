require_relative 'test_helper'

class CandleTest < Minitest::Test
  def test_that_it_has_a_version_number
    refute_nil ::Candle::VERSION
  end
end
