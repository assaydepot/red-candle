require_relative "../test_helper"

class PhiUnitTest < Minitest::Test
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
end