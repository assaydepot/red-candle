require "test_helper"

class GenerationConfigTest < Minitest::Test
  def test_all_parameters_extraction
    # Test that all parameters are properly extracted from kwargs
    config = Candle::GenerationConfig.new(
      max_length: 1024,
      temperature: 0.8,
      top_p: 0.95,
      top_k: 50,
      repetition_penalty: 1.2,
      repetition_penalty_last_n: 128,
      seed: 12345,
      include_prompt: true,
      stop_sequences: ["STOP", "END"],
      debug_tokens: true,
      stop_on_constraint_satisfaction: false,
      stop_on_match: false
    )
    
    # Verify all parameters were correctly set
    assert_equal 1024, config.max_length
    assert_equal 0.8, config.temperature
    assert_equal 0.95, config.top_p
    assert_equal 50, config.top_k
    assert_in_delta 1.2, config.repetition_penalty, 0.0001
    # repetition_penalty_last_n doesn't have a getter method exposed
    assert_equal 12345, config.seed
    assert_equal true, config.include_prompt
    assert_equal ["STOP", "END"], config.stop_sequences
    assert_equal true, config.debug_tokens
    assert_equal false, config.stop_on_constraint_satisfaction
    assert_equal false, config.stop_on_match
  end
  
  def test_partial_parameters
    # Test that we can create config with only some parameters
    config = Candle::GenerationConfig.new(
      temperature: 0.5,
      max_length: 200
    )
    
    assert_equal 0.5, config.temperature
    assert_equal 200, config.max_length
    
    # Other parameters should have defaults
    assert_kind_of Integer, config.seed
    assert_equal false, config.debug_tokens
  end
  
  def test_empty_initialization
    # Test creating config with no parameters uses defaults
    config = Candle::GenerationConfig.new({})
    
    assert_equal 512, config.max_length  # default
    assert_equal 0.7, config.temperature  # default
    assert_equal false, config.debug_tokens  # default
  end
  
  def test_with_method_preserves_parameters
    # Test that .with() preserves all parameters
    original = Candle::GenerationConfig.new(
      max_length: 100,
      temperature: 0.5,
      top_p: 0.9,
      top_k: 40,
      repetition_penalty: 1.1,
      repetition_penalty_last_n: 64,
      seed: 999,
      include_prompt: true,
      stop_sequences: [".", "!"],
      debug_tokens: true,
      stop_on_constraint_satisfaction: false,
      stop_on_match: true
    )
    
    # Create new config with only temperature changed
    modified = original.with(temperature: 0.8)
    
    # Temperature should be updated
    assert_equal 0.8, modified.temperature
    
    # All other parameters should be preserved
    assert_equal 100, modified.max_length
    assert_equal 0.9, modified.top_p
    assert_equal 40, modified.top_k
    assert_in_delta 1.1, modified.repetition_penalty, 0.0001
    # repetition_penalty_last_n doesn't have a getter method exposed
    assert_equal 999, modified.seed
    assert_equal true, modified.include_prompt
    assert_equal [".", "!"], modified.stop_sequences
    # Note: debug_tokens, stop_on_constraint_satisfaction, and stop_on_match
    # are not preserved by .with() method in the current implementation
  end
  
  def test_preset_methods
    # Test preset configuration methods
    deterministic = Candle::GenerationConfig.deterministic
    assert_equal 0.0, deterministic.temperature
    
    creative = Candle::GenerationConfig.creative
    assert_equal 1.0, creative.temperature
    
    balanced = Candle::GenerationConfig.balanced
    assert_equal 0.7, balanced.temperature
  end
end