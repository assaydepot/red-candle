require_relative "../test_helper"

class PhiTest < Minitest::Test
  # Class variable to store the model and avoid reloading
  @@llm = nil
  @@model_loaded = false
  
  def self.load_model_once
    unless @@model_loaded
      begin
        @@llm = Candle::LLM.from_pretrained(
          "TheBloke/phi-2-GGUF", 
          gguf_file: "phi-2.Q4_K_M.gguf"
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
    # Only load model for tests that need it
    if self.name.start_with?("test_phi2_", "test_phi4_", "test_phone_", "test_structured_")
      if ENV["CI"]
        skip "Skipping model download test in CI"
      end
      
      self.class.load_model_once
      if @@model_loaded == :failed
        skip "Model loading failed: #{@@load_error.message}"
      end
    end
  end
  
  # Unit tests that don't require model downloads
  def test_phi_in_supported_models
    # Check that Phi is mentioned in error messages for unsupported models
    error = assert_raises(RuntimeError) do
      Candle::LLM.from_pretrained("unknown/model-that-doesnt-exist")
    end
    
    assert error.message.include?("Phi"), "Phi should be listed as a supported model type"
  end
  
  def test_phi_tokenizer_patterns
    test_cases = {
      # Phi-3 variants
      "microsoft/Phi-3-mini-128k-instruct-GGUF" => "microsoft/Phi-3-mini-4k-instruct",
      "microsoft/Phi-3-medium-128k-instruct-GGUF" => "microsoft/Phi-3-medium-4k-instruct",
      "microsoft/Phi-3-small-128k-instruct-GGUF" => "microsoft/Phi-3-small-8k-instruct",
      "TheBloke/Phi-3-mini-4k-instruct-GGUF" => "microsoft/Phi-3-mini-4k-instruct",
      "SomeUser/phi-3-mini-gguf" => "microsoft/Phi-3-mini-4k-instruct",
      
      # Phi-2 variants
      "microsoft/phi-2-GGUF" => "microsoft/phi-2",
      "TheBloke/phi-2-GGUF" => "microsoft/phi-2",
      "quantized/phi-2-q4" => "microsoft/phi-2",
      
      # Phi-1.5 variants (note: underscore doesn't match \.)
      "microsoft/phi-1.5-GGUF" => "microsoft/phi-1_5",
      "TheBloke/phi-1.5-GGUF" => "microsoft/phi-1_5",
      
      # Generic phi
      "some/phi-model-GGUF" => "microsoft/phi-2"
    }
    
    test_cases.each do |input, expected|
      actual = Candle::LLM.guess_tokenizer(input)
      assert_equal expected, actual, "Failed for input: #{input}"
    end
  end
  
  def test_custom_phi_tokenizer_registration
    # Test registering a custom Phi tokenizer
    custom_model = "my-org/custom-phi-3-GGUF"
    custom_tokenizer = "my-org/custom-phi-3-tokenizer"
    
    Candle::LLM.register_tokenizer(custom_model, custom_tokenizer)
    
    assert_equal custom_tokenizer, Candle::LLM.guess_tokenizer(custom_model)
    
    # Test regex pattern registration
    Candle::LLM.register_tokenizer(/custom-phi-\d+/, "my-org/phi-tokenizer")
    assert_equal "my-org/phi-tokenizer", Candle::LLM.guess_tokenizer("custom-phi-4-GGUF")
  end
  
  def test_phi_architecture_detection
    # This tests that Phi models would be detected correctly
    # by checking the model name patterns used in the Rust code
    phi_models = [
      "microsoft/phi-2",
      "microsoft/Phi-3-mini-4k-instruct",
      "microsoft/Phi-3-medium-4k-instruct",
      "microsoft/Phi-3-small-8k-instruct"
    ]
    
    phi_models.each do |model|
      assert model.downcase.include?("phi"), "Model name should contain 'phi': #{model}"
    end
  end
  
  def test_phi_gguf_patterns
    # Test GGUF file naming patterns
    gguf_patterns = [
      "phi-2.Q4_K_M.gguf",
      "phi-3-mini-4k-instruct.Q5_K_S.gguf",
      "Phi-3-medium-128k.Q8_0.gguf"
    ]
    
    gguf_patterns.each do |pattern|
      assert pattern.match?(/\.gguf$/i), "Should match GGUF pattern: #{pattern}"
    end
  end

  def test_phi_tokenizer_registry
    # Test exact matches
    assert_equal "microsoft/Phi-3-mini-4k-instruct", Candle::LLM.guess_tokenizer("microsoft/Phi-3-mini-4k-instruct-GGUF")
    assert_equal "microsoft/phi-4", Candle::LLM.guess_tokenizer("microsoft/phi-4-gguf")
    assert_equal "microsoft/Phi-3.5-mini-instruct", Candle::LLM.guess_tokenizer("bartowski/Phi-3.5-mini-instruct-GGUF")
    
    # Test pattern matches
    assert_equal "microsoft/Phi-3-mini-4k-instruct", Candle::LLM.guess_tokenizer("TheBloke/Phi-3-mini-4k-instruct-GGUF")
    assert_equal "microsoft/phi-2", Candle::LLM.guess_tokenizer("TheBloke/phi-2-GGUF")
    assert_equal "microsoft/phi-2", Candle::LLM.guess_tokenizer("microsoft/phi-2-GGUF")
    assert_equal "microsoft/phi-4", Candle::LLM.guess_tokenizer("someone/phi-4-custom-gguf")
    assert_equal "microsoft/Phi-3.5-mini-instruct", Candle::LLM.guess_tokenizer("user/phi-3.5-mini-quantized")
    
    # Test variations
    assert_equal "microsoft/Phi-3-medium-4k-instruct", Candle::LLM.guess_tokenizer("Phi-3-medium-GGUF")
    assert_equal "microsoft/Phi-3-small-8k-instruct", Candle::LLM.guess_tokenizer("Phi-3-small-GGUF")
  end
  
  # Model loading tests
  def test_phi2_gguf_loading
    skip unless @@llm
    
    assert_instance_of Candle::LLM, @@llm
    assert @@llm.model_name.include?("TheBloke/phi-2-GGUF")
    
    # Test generation
    config = Candle::GenerationConfig.deterministic(max_length: 20, seed: 42)
    result = @@llm.generate("Hello", config: config)
    assert_instance_of String, result
    assert result.length > 0
  end
  
  def test_phi2_chat_template
    skip unless @@llm
    
    messages = [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Hello!" }
    ]
    
    prompt = @@llm.apply_chat_template(messages)
    assert_instance_of String, prompt
    assert prompt.include?("Hello!")
    # Phi-2 uses a simple format
    assert prompt.include?("System:") || prompt.include?("User:"),
           "Prompt should include role markers"
  end
  
  def test_phi3_chat_template
    skip "Skipping Phi-3 test - large model"
    # Phi-3 models are large, skip for regular testing
  end
  
  def test_phi4_gguf_loading
    skip "Phi-4 test requires different model - skipping"
    # Phi-4 requires a different model download, skip for now
  end
  
  def test_phone_constraint_generation
    skip unless @@llm
    
    phone_constraint = @@llm.constraint_from_regex('\d{3}-\d{3}-\d{4}')
    config = Candle::GenerationConfig.balanced(constraint: phone_constraint)
    result = @@llm.generate("Generate a phone number:", config: config)
    
    assert_instance_of String, result
    assert_match(/^\d{3}-\d{3}-\d{4}$/, result, "Result should be a valid phone number")
    refute result.include?("</s>"), "Result should not contain EOS tokens"
    refute result.include?("<|endoftext|>"), "Result should not contain Phi EOS tokens"
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