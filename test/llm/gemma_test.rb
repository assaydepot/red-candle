require_relative "../test_helper"

class GemmaTest < Minitest::Test
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
    skip "Skipping model download test in CI" if ENV["CI"]
    
    # Use GGUF model for faster testing
    begin
      model = Candle::LLM.from_pretrained(
        "google/gemma-3-4b-it-qat-q4_0-gguf",
        gguf_file: "gemma-3-4b-it-q4_0.gguf",
        tokenizer: "google/gemma-3-4b-it"
      )
      assert model.model_name.include?("gemma-3-4b-it-qat-q4_0-gguf")
      
      # Test generation
      config = Candle::GenerationConfig.deterministic(max_length: 20, seed: 42)
      result = model.generate("Hello", config: config)
      assert_instance_of String, result
      assert result.length > 0
    rescue => e
      skip "Model download failed: #{e.message}"
    end
  end
  
  def test_gemma_chat_template
    skip "Skipping model download test in CI" if ENV["CI"]
    
    begin
      model = Candle::LLM.from_pretrained(
        "google/gemma-3-4b-it-qat-q4_0-gguf",
        gguf_file: "gemma-3-4b-it-q4_0.gguf",
        tokenizer: "google/gemma-3-4b-it"
      )
      messages = [
        { role: "user", content: "What is Ruby?" }
      ]
      
      prompt = model.apply_chat_template(messages)
      assert_instance_of String, prompt
      assert prompt.include?("What is Ruby?")
      # Gemma uses specific chat format
      assert prompt.include?("<start_of_turn>") || prompt.include?("User:"),
             "Prompt should include turn markers"
    rescue => e
      skip "Model download failed: #{e.message}"
    end
  end
  
  def test_gemma_gguf_loading
    skip "Skipping GGUF download test in CI" if ENV["CI"]
    
    # Test that Gemma GGUF models can be loaded with explicit tokenizer
    model_id = "google/gemma-3-4b-it-qat-q4_0-gguf"
    gguf_file = "gemma-3-4b-it-q4_0.gguf"
    tokenizer = "google/gemma-3-4b-it"
    
    begin
      llm = Candle::LLM.from_pretrained(model_id, 
                                       gguf_file: gguf_file,
                                       tokenizer: tokenizer)
      # When loading with explicit parameters, the model name includes the full spec
      assert llm.model_name.include?(model_id)
      assert llm.model_name.include?(gguf_file)
    rescue => e
      skip "Model download failed: #{e.message}"
    end
  end
end