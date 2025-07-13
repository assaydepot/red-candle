require "test_helper"

class TestLLM < Minitest::Test
  def test_generation_config_creation
    # Test default config
    config = Candle::GenerationConfig.default
    assert_equal 512, config.max_length
    assert_equal 0.7, config.temperature
    
    # Test deterministic config
    config = Candle::GenerationConfig.deterministic
    assert_equal 0.0, config.temperature
    
    # Test creative config
    config = Candle::GenerationConfig.creative
    assert_equal 1.0, config.temperature
    
    # Test custom config
    config = Candle::GenerationConfig.new(
      max_length: 100,
      temperature: 0.5,
      top_p: 0.9,
      seed: 123
    )
    assert_equal 100, config.max_length
    assert_equal 0.5, config.temperature
    assert_equal 0.9, config.top_p
    assert_equal 123, config.seed
  end
  
  def test_generation_config_with_method
    config = Candle::GenerationConfig.default
    new_config = config.with(temperature: 0.9, max_length: 200)
    
    # Original should be unchanged
    assert_equal 0.7, config.temperature
    assert_equal 512, config.max_length
    
    # New config should have updated values
    assert_equal 0.9, new_config.temperature
    assert_equal 200, new_config.max_length
  end

  # Skip actual model loading tests as they require downloading large files
  def test_llm_initialization_error
    skip "Model loading requires large downloads"
    
    # This would fail with unsupported model error
    assert_raises do
      Candle::LLM.from_pretrained("unknown-model")
    end
  end
end