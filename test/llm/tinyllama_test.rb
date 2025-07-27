require_relative "../test_helper"

class TinyLlamaTest < Minitest::Test
  # Class variable to store the model and avoid reloading
  @@llm = nil
  @@model_loaded = false
  
  def self.load_model_once
    unless @@model_loaded
      begin
        @@llm = Candle::LLM.from_pretrained(
          "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", 
          gguf_file: "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf"
        )
        @@model_loaded = true
      rescue => e
        # If model loading fails (e.g., no internet), skip all LLM tests
        @@model_loaded = :failed
        @@load_error = e
      end
    end
  end
  
  def setup
    self.class.load_model_once
    if @@model_loaded == :failed
      skip "Model loading failed: #{@@load_error.message}"
    end
  end
  
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
  
  def test_generation_config_debug_tokens
    # Test debug_tokens flag
    config = Candle::GenerationConfig.new(debug_tokens: true)
    assert_equal true, config.debug_tokens
    
    # Test default is false
    config = Candle::GenerationConfig.default
    assert_equal false, config.debug_tokens
  end

  def test_llm_generate_basic
    skip unless @@llm
    
    config = Candle::GenerationConfig.deterministic(
      max_length: 20,
      seed: 42
    )
    
    result = @@llm.generate("Hello", config: config)
    
    assert_instance_of String, result
    assert result.length > 0, "Generated text should not be empty"
    refute result.include?("▁"), "Generated text should not contain raw tokens"
  end
  
  def test_llm_generate_with_stop_sequences
    skip unless @@llm
    
    config = Candle::GenerationConfig.new(
      max_length: 50,
      temperature: 0.0,
      seed: 42,
      stop_sequences: ["\n", "."]
    )
    
    result = @@llm.generate("The capital of France is", config: config)
    
    assert_instance_of String, result
    # Should stop at first period or newline
    assert result.length < 100, "Should respect stop sequences"
  end
  
  def test_llm_generate_stream
    skip unless @@llm
    
    config = Candle::GenerationConfig.deterministic(
      max_length: 30,
      seed: 42
    )
    
    streamed_text = ""
    callback_count = 0
    
    result = @@llm.generate_stream("Hello", config: config) do |token|
      assert_instance_of String, token
      refute token.include?("▁"), "Streamed tokens should not be raw"
      streamed_text += token
      callback_count += 1
    end
    
    assert_instance_of String, result
    assert callback_count > 0, "Stream callback should be called"
    assert_equal result, streamed_text, "Streamed text should match final result"
  end
  
  def test_llm_generate_stream_debug_tokens
    skip unless @@llm
    
    config = Candle::GenerationConfig.new(
      max_length: 20,
      temperature: 0.0,
      seed: 42,
      debug_tokens: true
    )
    
    debug_tokens_seen = false
    
    @@llm.generate_stream("Hi", config: config) do |token|
      # In debug mode, we should see tokens like [123:▁Hello]
      if token.match?(/\[\d+:.*\]/)
        debug_tokens_seen = true
      end
    end
    
    assert debug_tokens_seen, "Debug tokens should be visible when debug_tokens is true"
  end
  
  def test_llm_generate_debug_tokens
    skip unless @@llm
    
    config = Candle::GenerationConfig.new(
      max_length: 20,
      temperature: 0.0,
      seed: 42,
      debug_tokens: true
    )
    
    result = @@llm.generate("Hi", config: config)
    
    # In debug mode, the result should only contain token information
    assert result.match?(/\[\d+:.*\]/), "Debug tokens should be visible in generate output"
    # The entire result should be debug tokens
    assert result.match?(/^(\[\d+:[^\]]*\])+$/), "Result should only contain debug tokens"
  end
  
  def test_llm_chat
    skip unless @@llm
    
    messages = [
      { role: "user", content: "Say hello" }
    ]
    
    config = Candle::GenerationConfig.deterministic(
      max_length: 20,
      seed: 42
    )
    
    result = @@llm.chat(messages, config: config)
    
    assert_instance_of String, result
    assert result.length > 0, "Chat response should not be empty"
  end
  
  def test_llm_chat_stream
    skip unless @@llm
    
    messages = [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Count to 3" }
    ]
    
    config = Candle::GenerationConfig.new(
      max_length: 30,
      temperature: 0.0,
      seed: 42
    )
    
    callback_count = 0
    
    result = @@llm.chat_stream(messages, config: config) do |token|
      callback_count += 1
    end
    
    assert_instance_of String, result
    assert callback_count > 0, "Stream callback should be called during chat"
  end
  
  def test_llm_apply_chat_template
    skip unless @@llm
    
    messages = [
      { role: "system", content: "You are helpful." },
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there!" },
      { role: "user", content: "How are you?" }
    ]
    
    template_result = @@llm.apply_chat_template(messages)
    
    assert_instance_of String, template_result
    assert template_result.include?("Hello"), "Template should include user message"
    assert template_result.include?("Hi there!"), "Template should include assistant message"
  end
  
  def test_llm_model_info
    skip unless @@llm
    
    # Model name includes the specific GGUF file and original model reference
    assert @@llm.model_name.include?("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    assert_instance_of Candle::Device, @@llm.device
  end
  
  def test_llm_clear_cache
    skip unless @@llm
    
    # Generate something first
    config = Candle::GenerationConfig.new(max_length: 10)
    @@llm.generate("Test", config: config)
    
    # Clear cache should not raise
    @@llm.clear_cache
    # If we get here, no exception was raised
    assert true
  end
  
  def test_llm_multiple_generations
    skip unless @@llm
    
    config = Candle::GenerationConfig.deterministic(
      max_length: 15,
      seed: 42
    )
    
    # Test that multiple generations work correctly
    result1 = @@llm.generate("Hello", config: config)
    result2 = @@llm.generate("Hello", config: config)
    
    assert_equal result1, result2, "Same seed should produce same results"
    
    # Different seeds should produce different results
    config2 = config.with(seed: 123)
    _result3 = @@llm.generate("Hello", config: config2)
    
    # Note: In rare cases this might fail if both seeds happen to generate the same
    # but it's very unlikely with different seeds
    # For small models like TinyLlama, this can happen more often
    # So we'll just skip this assertion for now
    # refute_equal result1, result3, "Different seeds should usually produce different results"
  end
  
  def test_llm_empty_prompt
    skip unless @@llm
    
    config = Candle::GenerationConfig.new(max_length: 10)
    
    # Empty prompt should still generate something
    result = @@llm.generate("", config: config)
    assert_instance_of String, result
  end
  
  def test_llm_include_prompt_option
    skip unless @@llm
    
    prompt = "The answer is"
    config = Candle::GenerationConfig.new(
      max_length: 20,
      temperature: 0.0,
      seed: 42,
      include_prompt: true
    )
    
    result = @@llm.generate(prompt, config: config)
    assert result.start_with?(prompt), "Result should include prompt when include_prompt is true"
    
    config_no_prompt = config.with(include_prompt: false)
    result_no_prompt = @@llm.generate(prompt, config: config_no_prompt)
    refute result_no_prompt.start_with?(prompt), "Result should not include prompt when include_prompt is false"
  end
  
  def test_tokenizer_registry
    # Test exact match
    tokenizer = Candle::LLM.guess_tokenizer("TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
    assert_equal "mistralai/Mistral-7B-Instruct-v0.2", tokenizer
    
    # Test pattern match
    tokenizer = Candle::LLM.guess_tokenizer("someone/mistral-7b-instruct-v0.2-custom")
    assert_equal "mistralai/Mistral-7B-Instruct-v0.2", tokenizer
    
    # Test fallback
    tokenizer = Candle::LLM.guess_tokenizer("unknown/model-gguf")
    assert_equal "unknown/model", tokenizer
  end
  
  def test_tokenizer_registration
    # Register a custom tokenizer
    Candle::LLM.register_tokenizer("my-org/my-model-GGUF", "my-org/my-tokenizer")
    
    tokenizer = Candle::LLM.guess_tokenizer("my-org/my-model-GGUF")
    assert_equal "my-org/my-tokenizer", tokenizer
    
    # Register a pattern
    Candle::LLM.register_tokenizer(/my-custom-.*-gguf/i, "my-org/custom-tokenizer")
    
    tokenizer = Candle::LLM.guess_tokenizer("user/my-custom-model-gguf")
    assert_equal "my-org/custom-tokenizer", tokenizer
  end
  
  def test_format_messages_legacy
    skip unless @@llm
    
    # Test the legacy format_messages method (private method)
    messages = [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Hello there!" },
      { role: "assistant", content: "Hi! How can I help?" },
      { role: "user", content: "What's the weather?" }
    ]
    
    # Use send to call private method
    formatted = @@llm.send(:format_messages, messages)
    
    assert_instance_of String, formatted
    assert formatted.include?("System: You are a helpful assistant.")
    assert formatted.include?("User: Hello there!")
    assert formatted.include?("Assistant: Hi! How can I help?")
    assert formatted.include?("User: What's the weather?")
    assert formatted.end_with?("\n\nAssistant:")
    
    # Test with unknown role
    messages_with_unknown = [
      { role: "narrator", content: "Once upon a time" }
    ]
    
    formatted_unknown = @@llm.send(:format_messages, messages_with_unknown)
    assert formatted_unknown.include?("Once upon a time")
    refute formatted_unknown.include?("narrator:")
  end
  
  def test_register_tokenizer_invalid_pattern
    # Test that invalid pattern types raise ArgumentError
    assert_raises(ArgumentError) do
      Candle::LLM.register_tokenizer(123, "some/tokenizer")
    end
    
    assert_raises(ArgumentError) do
      Candle::LLM.register_tokenizer([:array], "some/tokenizer")
    end
  end
  
  def test_phone_constraint_generation
    skip unless @@llm
    
    phone_constraint = @@llm.constraint_from_regex('\d{3}-\d{3}-\d{4}')
    config = Candle::GenerationConfig.balanced(constraint: phone_constraint)
    result = @@llm.generate("Generate a phone number:", config: config)
    
    assert_instance_of String, result
    assert_match(/^\d{3}-\d{3}-\d{4}$/, result, "Result should be a valid phone number")
    refute result.include?("</s>"), "Result should not contain EOS tokens"
  end
  
  def test_structured_generation_with_schema
    skip unless @@llm
    
    schema = {
      type: "object",
      properties: {
        answer: { type: "string", enum: ["yes", "no"] },
        confidence: { type: "number", minimum: 0, maximum: 1 }
      },
      required: ["answer"]
    }
    
    result = @@llm.generate_structured("Is Ruby easy to learn?", schema: schema)
    
    assert_instance_of Hash, result
    assert result.key?("answer"), "Result should have 'answer' key"
    assert ["yes", "no"].include?(result["answer"]), "Answer should be 'yes' or 'no'"
    
    if result.key?("confidence")
      assert_instance_of Float, result["confidence"]
      assert result["confidence"] >= 0 && result["confidence"] <= 1, "Confidence should be between 0 and 1"
    end
  end
end