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
    skip "Skipping model download test - set DOWNLOAD_MODELS=true to enable" unless ENV['DOWNLOAD_MODELS'] == 'true'
    skip "Skipping model download test in CI" if ENV["CI"]
    
    # Test loading Gemma model
    model = Candle::LLM.from_pretrained("google/gemma-2b-it")
    assert_equal "google/gemma-2b-it", model.model_name
    
    # Test generation
    config = Candle::GenerationConfig.deterministic(max_length: 20, seed: 42)
    result = model.generate("Hello", config: config)
    assert_instance_of String, result
    assert result.length > 0
  end
  
  def test_gemma_chat_template
    skip "Skipping model download test - set DOWNLOAD_MODELS=true to enable" unless ENV['DOWNLOAD_MODELS'] == 'true'
    skip "Skipping model download test in CI" if ENV["CI"]
    
    model = Candle::LLM.from_pretrained("google/gemma-2b-it")
    messages = [
      { role: "user", content: "What is Ruby?" }
    ]
    
    prompt = model.apply_chat_template(messages)
    assert prompt.include?("<start_of_turn>user")
    assert prompt.include?("<end_of_turn>")
    assert prompt.include?("<start_of_turn>model")
  end
  
  def test_gemma_gguf_loading
    skip "Skipping GGUF download test in CI" if ENV["CI"]
    skip "Skipping GGUF download test - run individual test to enable"
    
    # Test that Gemma GGUF models can be loaded with explicit tokenizer
    model_id = "google/gemma-3-4b-it-qat-q4_0-gguf"
    gguf_file = "gemma-3-4b-it-q4_0.gguf"
    
    begin
      llm = Candle::LLM.from_pretrained(model_id, 
                                       gguf_file: gguf_file,
                                       tokenizer: "google/gemma-3-4b-it")
      # When loading with explicit parameters, the model name includes the full spec
      assert llm.model_name.include?(model_id)
      assert llm.model_name.include?(gguf_file)
    rescue => e
      skip "Model download failed: #{e.message}"
    end
  end
end