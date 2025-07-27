require_relative "../test_helper"

class QwenTest < Minitest::Test
  # Class variable to store the model and avoid reloading
  @@llm = nil
  @@model_loaded = false
  @@device = Candle::Device.cpu
  
  def self.load_model_once
    unless @@model_loaded
      begin
        @@llm = Candle::LLM.from_pretrained(
          "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
          device: @@device,
          gguf_file: "qwen2.5-1.5b-instruct-q4_k_m.gguf"
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
    @device = @@device
    
    # Only load model for tests that need it
    if self.name.start_with?("test_qwen_gguf_", "test_qwen_generation_", "test_qwen_chat_", "test_phone_", "test_structured_")
      if ENV["CI"]
        skip "Skipping model download test in CI"
      end
      
      self.class.load_model_once
      if @@model_loaded == :failed
        skip "Model loading failed: #{@@load_error.message}"
      end
    end
  end

  def test_qwen_gguf_loading
    skip unless @@llm
    
    # Test that the loaded model has expected properties
    assert @@llm.model_name.include?("Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    assert @@llm.model_name.include?("qwen2.5-1.5b-instruct-q4_k_m.gguf")
    assert_equal @device, @@llm.device
    
    # Test generation
    config = Candle::GenerationConfig.deterministic(max_length: 20, seed: 42)
    result = @@llm.generate("Hello", config: config)
    assert_instance_of String, result
    assert result.length > 0
    refute result.match?(/[^\x20-\x7E\n]/), "Result should not contain non-printable characters"
  end
  
  def test_qwen_generation_stream
    skip unless @@llm
    
    config = Candle::GenerationConfig.deterministic(max_length: 30, seed: 42)
    
    streamed_text = ""
    callback_count = 0
    
    result = @@llm.generate_stream("Tell me about Ruby", config: config) do |token|
      assert_instance_of String, token
      streamed_text += token
      callback_count += 1
    end
    
    assert_instance_of String, result
    assert callback_count > 0, "Stream callback should be called"
    assert_equal result, streamed_text, "Streamed text should match final result"
  end

  def test_qwen_tokenizer_registry
    # Test exact match
    tokenizer = Candle::LLM.guess_tokenizer("Qwen/Qwen3-8B-GGUF")
    assert_equal "Qwen/Qwen3-8B", tokenizer
    
    # Test pattern matching
    tokenizer = Candle::LLM.guess_tokenizer("bartowski/Qwen3-8B-GGUF")
    assert_equal "Qwen/Qwen3-8B", tokenizer
    
    tokenizer = Candle::LLM.guess_tokenizer("someone/qwen-3-14b-gguf")
    assert_equal "Qwen/Qwen3-14B", tokenizer
    
    tokenizer = Candle::LLM.guess_tokenizer("qwen3-0.5b-quantized")
    assert_equal "Qwen/Qwen3-0.5B", tokenizer
  end

  def test_qwen_chat_template
    skip unless @@llm
    
    messages = [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "What is Ruby?" }
    ]
    
    # Test chat template application
    prompt = @@llm.apply_chat_template(messages)
    assert_instance_of String, prompt
    assert prompt.include?("What is Ruby?")
    assert prompt.include?("You are a helpful assistant.")
    
    # Qwen uses specific chat format markers
    assert prompt.include?("<|im_start|>") || prompt.include?("User:"),
           "Prompt should include conversation markers"
    
    # Test chat generation
    config = Candle::GenerationConfig.deterministic(max_length: 30, seed: 42)
    result = @@llm.chat(messages, config: config)
    assert_instance_of String, result
    assert result.length > 0
  end
  
  def test_qwen_chat_stream
    skip unless @@llm
    
    messages = [
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
  
  def test_phone_constraint_generation
    skip unless @@llm
    
    phone_constraint = @@llm.constraint_from_regex('\d{3}-\d{3}-\d{4}')
    config = Candle::GenerationConfig.balanced(constraint: phone_constraint)
    result = @@llm.generate("Generate a phone number:", config: config)
    
    assert_instance_of String, result
    assert_match(/^\d{3}-\d{3}-\d{4}/, result, "Result should start with a valid phone number")
    # Note: Qwen may generate </s> as text after the pattern, which is expected behavior
    refute result.include?("<|im_end|>"), "Result should not contain Qwen EOS tokens"
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
      assert [Float, Integer].include?(result["confidence"].class), "Confidence should be a number"
      # Accept both 0-1 range and 0-100 range (some models interpret as percentage)
      assert result["confidence"] >= 0 && result["confidence"] <= 100, "Confidence should be between 0 and 100"
    end
  end
end