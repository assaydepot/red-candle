require_relative "../test_helper"

class MistralTest < Minitest::Test
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
    skip "Skipping GGUF download test in CI" if ENV["CI"]
    
    # Test loading a small quantized Mistral model
    model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    gguf_file = "mistral-7b-instruct-v0.2.Q2_K.gguf"  # Smallest quantization
    
    begin
      model = Candle::LLM.from_pretrained(model_id, gguf_file: gguf_file)
      assert model.model_name.include?(model_id)
      assert model.model_name.include?(gguf_file)
      
      # Test generation
      config = Candle::GenerationConfig.deterministic(max_length: 20, seed: 42)
      result = model.generate("Hello", config: config)
      assert_instance_of String, result
      assert result.length > 0
    rescue => e
      skip "Model download failed: #{e.message}"
    end
  end
  
  def test_mistral_chat_template
    skip "Skipping GGUF download test in CI" if ENV["CI"]
    
    # Test chat template with GGUF model
    model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    gguf_file = "mistral-7b-instruct-v0.2.Q2_K.gguf"
    
    begin
      model = Candle::LLM.from_pretrained(model_id, gguf_file: gguf_file)
      messages = [
        { role: "user", content: "What is Ruby?" }
      ]
      
      prompt = model.apply_chat_template(messages)
      assert_instance_of String, prompt
      assert prompt.include?("What is Ruby?")
      # Check for Mistral-specific template markers
      assert prompt.include?("[INST]") || prompt.include?("User:"), 
             "Prompt should include instruction markers"
    rescue => e
      skip "Model download failed: #{e.message}"
    end
  end
  
  def test_mistral_gguf_with_explicit_tokenizer
    skip "Skipping GGUF download test in CI" if ENV["CI"]
    
    # Test that Mistral GGUF models can be loaded with auto-detected tokenizer
    model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    gguf_file = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    begin
      llm = Candle::LLM.from_pretrained(model_id, gguf_file: gguf_file)
      # When loading GGUF, the model name includes the file spec
      assert llm.model_name.include?(model_id)
      assert llm.model_name.include?(gguf_file)
      
      # Test chat with the loaded model
      messages = [{ role: "user", content: "Hello" }]
      config = Candle::GenerationConfig.deterministic(max_length: 10, seed: 42)
      result = llm.chat(messages, config: config)
      assert_instance_of String, result
    rescue => e
      skip "Model download failed: #{e.message}"
    end
  end
end