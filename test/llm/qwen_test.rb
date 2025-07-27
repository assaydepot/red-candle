require_relative "../test_helper"

class QwenTest < Minitest::Test
  def setup
    @device = Candle::Device.cpu
  end

  def test_qwen_gguf_loading
    skip "Skipping GGUF download test in CI" if ENV["CI"]
    
    # Test that Qwen GGUF models can be loaded with auto-detected tokenizer
    # model_id = "Qwen/Qwen3-4B-GGUF"
    # gguf_file = "qwen3-4b-q4_k_m.gguf"
    model_id = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    gguf_file = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

    # This should auto-detect the tokenizer
    begin
      llm = Candle::LLM.from_pretrained(model_id, device: @device, gguf_file: gguf_file)
      # When loading GGUF, the model name includes the file spec
      assert llm.model_name.include?(model_id)
      assert llm.model_name.include?(gguf_file)
      assert_equal @device, llm.device
      
      # Test generation
      config = Candle::GenerationConfig.deterministic(max_length: 20, seed: 42)
      result = llm.generate("Hello", config: config)
      assert_instance_of String, result
      assert result.length > 0
      refute result.match?(/[^\x20-\x7E\n]/), "Result should not contain non-printable characters"
    rescue => e
      skip "Model download failed: #{e.message}"
    end
  end
  
  def test_qwen_generation_stream
    skip "Skipping GGUF download test in CI" if ENV["CI"]
    
    # model_id = "Qwen/Qwen3-4B-GGUF"
    # gguf_file = "qwen3-4b-q4_k_m.gguf"
    model_id = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    gguf_file = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    
    begin
      llm = Candle::LLM.from_pretrained(model_id, device: @device, gguf_file: gguf_file)
      
      config = Candle::GenerationConfig.deterministic(max_length: 30, seed: 42)
      
      streamed_text = ""
      callback_count = 0
      
      result = llm.generate_stream("Tell me about Ruby", config: config) do |token|
        assert_instance_of String, token
        streamed_text += token
        callback_count += 1
      end
      
      assert_instance_of String, result
      assert callback_count > 0, "Stream callback should be called"
      assert_equal result, streamed_text, "Streamed text should match final result"
    rescue => e
      skip "Model download failed: #{e.message}"
    end
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
    skip "Skipping model download test in CI" if ENV["CI"]
    
    # model_id = "Qwen/Qwen3-4B-GGUF"
    # gguf_file = "qwen3-4b-q4_k_m.gguf"
    model_id = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    gguf_file = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    
    begin
      llm = Candle::LLM.from_pretrained(model_id, device: @device, gguf_file: gguf_file)
      
      messages = [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "What is Ruby?" }
      ]
      
      # Test chat template application
      prompt = llm.apply_chat_template(messages)
      assert_instance_of String, prompt
      assert prompt.include?("What is Ruby?")
      assert prompt.include?("You are a helpful assistant.")
      
      # Qwen uses specific chat format markers
      assert prompt.include?("<|im_start|>") || prompt.include?("User:"),
             "Prompt should include conversation markers"
      
      # Test chat generation
      config = Candle::GenerationConfig.deterministic(max_length: 30, seed: 42)
      result = llm.chat(messages, config: config)
      assert_instance_of String, result
      assert result.length > 0
    rescue => e
      skip "Model download failed: #{e.message}"
    end
  end
  
  def test_qwen_chat_stream
    skip "Skipping model download test in CI" if ENV["CI"]
    
    # model_id = "Qwen/Qwen3-4B-GGUF"
    # gguf_file = "qwen3-4b-q4_k_m.gguf"
    model_id = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    gguf_file = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    
    begin
      llm = Candle::LLM.from_pretrained(model_id, device: @device, gguf_file: gguf_file)
      
      messages = [
        { role: "user", content: "Count to 3" }
      ]
      
      config = Candle::GenerationConfig.new(
        max_length: 30,
        temperature: 0.0,
        seed: 42
      )
      
      callback_count = 0
      
      result = llm.chat_stream(messages, config: config) do |token|
        callback_count += 1
      end
      
      assert_instance_of String, result
      assert callback_count > 0, "Stream callback should be called during chat"
    rescue => e
      skip "Model download failed: #{e.message}"
    end
  end
end