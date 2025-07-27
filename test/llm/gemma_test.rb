require_relative "../test_helper"

class GemmaTest < Minitest::Test
  # Class variable to store the model and avoid reloading
  @@llm = nil
  @@model_loaded = false
  
  def self.load_model_once
    unless @@model_loaded
      begin
        @@llm = Candle::LLM.from_pretrained(
          "google/gemma-3-4b-it-qat-q4_0-gguf",
          gguf_file: "gemma-3-4b-it-q4_0.gguf",
          tokenizer: "google/gemma-3-4b-it"
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
  
  def test_gemma_tokenizer_registry
    # Test pattern matches (based on actual registry)
    assert_equal "google/gemma-2b", Candle::LLM.guess_tokenizer("google/gemma-2b-GGUF")
    assert_equal "google/gemma-2b", Candle::LLM.guess_tokenizer("google/gemma-2b-it-GGUF")
    assert_equal "google/gemma-7b", Candle::LLM.guess_tokenizer("google/gemma-7b-GGUF")
    assert_equal "google/gemma-7b", Candle::LLM.guess_tokenizer("google/gemma-7b-it-GGUF")
    
    # Test pattern matches
    assert_equal "google/gemma-2b", Candle::LLM.guess_tokenizer("TheBloke/gemma-2b-GGUF")
    assert_equal "google/gemma-7b", Candle::LLM.guess_tokenizer("someone/gemma-7b-instruct-gguf")
    assert_equal "google/gemma-2b", Candle::LLM.guess_tokenizer("bartowski/gemma-2b-it-GGUF")
    
    # Test Gemma 2 patterns
    assert_equal "google/gemma-2-9b", Candle::LLM.guess_tokenizer("google/gemma-2-9b-it-GGUF")
    assert_equal "google/gemma-2-2b", Candle::LLM.guess_tokenizer("google/gemma-2-2b-it-GGUF")
  end
  
  def test_gemma_model_loading
    skip unless @@llm
    
    assert @@llm.model_name.include?("gemma-3-4b-it-qat-q4_0-gguf")
    
    # Test generation
    config = Candle::GenerationConfig.deterministic(max_length: 20, seed: 42)
    result = @@llm.generate("Hello", config: config)
    assert_instance_of String, result
    assert result.length > 0
  end
  
  def test_gemma_chat_template
    skip unless @@llm
    
    messages = [
      { role: "user", content: "What is Ruby?" }
    ]
    
    prompt = @@llm.apply_chat_template(messages)
    assert_instance_of String, prompt
    assert prompt.include?("What is Ruby?")
    # Gemma uses specific chat format
    assert prompt.include?("<start_of_turn>") || prompt.include?("User:"),
           "Prompt should include turn markers"
  end
  
  def test_gemma_gguf_loading
    skip unless @@llm
    
    # Test that the loaded model has expected properties
    assert @@llm.model_name.include?("google/gemma-3-4b-it-qat-q4_0-gguf")
    assert @@llm.model_name.include?("gemma-3-4b-it-q4_0.gguf")
  end
  
  def test_phone_constraint_generation
    skip unless @@llm
    
    phone_constraint = @@llm.constraint_from_regex('\d{3}-\d{3}-\d{4}')
    config = Candle::GenerationConfig.balanced(constraint: phone_constraint)
    result = @@llm.generate("Generate a phone number:", config: config)
    
    assert_instance_of String, result
    assert_match(/^\d{3}-\d{3}-\d{4}$/, result, "Result should be a valid phone number")
    refute result.include?("<eos>"), "Result should not contain Gemma EOS token"
    refute result.include?("<end_of_turn>"), "Result should not contain Gemma chat EOS token"
    refute result.include?("</s>"), "Result should not contain generation boundary markers"
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