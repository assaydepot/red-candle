require_relative "../test_helper"

class LlamaTest < Minitest::Test
  def test_llama_tokenizer_registry
    # Test exact matches
    assert_equal "meta-llama/Llama-2-7b-hf", 
                 Candle::LLM.guess_tokenizer("TheBloke/Llama-2-7B-GGUF")
    assert_equal "meta-llama/Llama-2-7b-chat-hf", 
                 Candle::LLM.guess_tokenizer("TheBloke/Llama-2-7B-Chat-GGUF")
    assert_equal "meta-llama/Llama-2-13b-hf", 
                 Candle::LLM.guess_tokenizer("TheBloke/Llama-2-13B-GGUF")
    assert_equal "meta-llama/Llama-2-13b-chat-hf", 
                 Candle::LLM.guess_tokenizer("TheBloke/Llama-2-13B-Chat-GGUF")
    
    # Test pattern matches
    assert_equal "meta-llama/Llama-2-7b-hf", 
                 Candle::LLM.guess_tokenizer("someone/llama-2-7b-custom-gguf")
    assert_equal "meta-llama/Llama-2-7b-chat-hf", 
                 Candle::LLM.guess_tokenizer("user/llama-2-7b-chat-quantized")
    
    # Test Llama 3 patterns
    assert_equal "meta-llama/Meta-Llama-3-8B", 
                 Candle::LLM.guess_tokenizer("meta-llama/Meta-Llama-3-8B-GGUF")
    assert_equal "meta-llama/Meta-Llama-3-8B-Instruct", 
                 Candle::LLM.guess_tokenizer("NousResearch/Meta-Llama-3-8B-Instruct-GGUF")
  end
  
  def test_llama2_model_loading
    skip "Skipping model download test - set DOWNLOAD_MODELS=true to enable" unless ENV['DOWNLOAD_MODELS'] == 'true'
    skip "Skipping model download test in CI" if ENV["CI"]
    
    # Test loading Llama 2 model
    model = Candle::LLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    assert_equal "meta-llama/Llama-2-7b-hf", model.model_name
    
    # Test generation
    config = Candle::GenerationConfig.deterministic(max_length: 20, seed: 42)
    result = model.generate("Hello", config: config)
    assert_instance_of String, result
    assert result.length > 0
  end
  
  def test_llama2_chat_template
    skip "Skipping model download test - set DOWNLOAD_MODELS=true to enable" unless ENV['DOWNLOAD_MODELS'] == 'true'
    skip "Skipping model download test in CI" if ENV["CI"]
    
    model = Candle::LLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    messages = [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "What is Ruby?" }
    ]
    
    prompt = model.apply_chat_template(messages)
    assert prompt.include?("[INST]")
    assert prompt.include?("[/INST]")
    assert prompt.include?("<<SYS>>")
    assert prompt.include?("<</SYS>>")
  end
  
  def test_llama3_chat_template
    skip "Skipping model download test - set DOWNLOAD_MODELS=true to enable" unless ENV['DOWNLOAD_MODELS'] == 'true'
    skip "Skipping model download test in CI" if ENV["CI"]
    
    model = Candle::LLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    messages = [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "What is Ruby?" }
    ]
    
    prompt = model.apply_chat_template(messages)
    assert prompt.include?("<|begin_of_text|>")
    assert prompt.include?("<|start_header_id|>")
    assert prompt.include?("<|end_header_id|>")
    assert prompt.include?("<|eot_id|>")
  end
  
  def test_llama_gguf_loading
    skip "Skipping GGUF download test in CI" if ENV["CI"]
    skip "Skipping GGUF download test - run individual test to enable"
    
    # Test that Llama GGUF models can be loaded with auto-detected tokenizer
    model_id = "TheBloke/Llama-2-7B-Chat-GGUF"
    gguf_file = "llama-2-7b-chat.Q4_K_M.gguf"
    
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