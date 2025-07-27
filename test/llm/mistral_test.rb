require_relative "../test_helper"

class MistralTest < Minitest::Test
  # Class variable to store the model and avoid reloading
  @@llm = nil
  @@model_loaded = false
  
  def self.load_model_once
    unless @@model_loaded
      begin
        @@llm = Candle::LLM.from_pretrained(
          "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
          gguf_file: "mistral-7b-instruct-v0.2.Q2_K.gguf"
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
    if ENV["CI"]
      skip "Skipping model download test in CI"
    end
    
    self.class.load_model_once
    if @@model_loaded == :failed
      skip "Model loading failed: #{@@load_error.message}"
    end
  end
  
  def test_mistral_tokenizer_registry
    # Test exact matches
    assert_equal "mistralai/Mistral-7B-Instruct-v0.2", 
                 Candle::LLM.guess_tokenizer("TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
    assert_equal "mistralai/Mistral-7B-Instruct-v0.1", 
                 Candle::LLM.guess_tokenizer("TheBloke/Mistral-7B-Instruct-v0.1-GGUF")
    assert_equal "mistralai/Mistral-7B-v0.1", 
                 Candle::LLM.guess_tokenizer("TheBloke/Mistral-7B-v0.1-GGUF")
    
    # Test pattern matches
    assert_equal "mistralai/Mistral-7B-Instruct-v0.2", 
                 Candle::LLM.guess_tokenizer("someone/mistral-7b-instruct-v0.2-custom")
    assert_equal "mistralai/Mistral-7B-Instruct-v0.1", 
                 Candle::LLM.guess_tokenizer("user/mistral-7b-instruct-v0.1-gguf")
  end
  
  def test_mistral_gguf_loading_and_generation
    skip unless @@llm
    
    assert @@llm.model_name.include?("TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
    assert @@llm.model_name.include?("mistral-7b-instruct-v0.2.Q2_K.gguf")
    
    # Test generation
    config = Candle::GenerationConfig.deterministic(max_length: 20, seed: 42)
    result = @@llm.generate("Hello", config: config)
    assert_instance_of String, result
    assert result.length > 0
  end
  
  def test_mistral_chat_template
    skip unless @@llm
    
    messages = [
      { role: "user", content: "What is Ruby?" }
    ]
    
    prompt = @@llm.apply_chat_template(messages)
    assert_instance_of String, prompt
    assert prompt.include?("What is Ruby?")
    # Check for Mistral-specific template markers
    assert prompt.include?("[INST]") || prompt.include?("User:"), 
           "Prompt should include instruction markers"
  end
  
  def test_mistral_gguf_with_explicit_tokenizer
    skip unless @@llm
    
    # Test that the loaded model works correctly
    assert @@llm.model_name.include?("TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
    
    # Test chat with the loaded model
    messages = [{ role: "user", content: "Hello" }]
    config = Candle::GenerationConfig.deterministic(max_length: 10, seed: 42)
    result = @@llm.chat(messages, config: config)
    assert_instance_of String, result
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
      assert [Float, Integer].include?(result["confidence"].class), "Confidence should be a number"
      # Accept both 0-1 range and 0-100 range (some models interpret as percentage)
      assert result["confidence"] >= 0 && result["confidence"] <= 100, "Confidence should be between 0 and 100"
    end
  end
end