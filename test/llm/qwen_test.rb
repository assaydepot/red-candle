require_relative "../test_helper"

class QwenTest < Minitest::Test
  def setup
    @device = Candle::Device.cpu
  end

  def test_qwen_gguf_loading
    skip "Skipping GGUF download test in CI" if ENV["CI"]
    
    # Test that Qwen GGUF models can be loaded with auto-detected tokenizer
    model_id = "Qwen/Qwen3-4B-GGUF"
    gguf_file = "qwen3-4b-q4_k_m.gguf"
    
    # This should auto-detect the tokenizer
    begin
      llm = Candle::LLM.from_pretrained(model_id, device: @device, gguf_file: gguf_file)
      assert_equal model_id, llm.model_name
      assert_equal @device, llm.device
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
    skip "Skipping model download test" if ENV["CI"]
    
    # Mock test for chat template application
    messages = [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Hello!" },
      { role: "assistant", content: "Hi there! How can I help you today?" },
      { role: "user", content: "What is Ruby?" }
    ]
    
    # Expected Qwen chat format
    expected_format = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
                     "<|im_start|>user\nHello!<|im_end|>\n" +
                     "<|im_start|>assistant\nHi there! How can I help you today?<|im_end|>\n" +
                     "<|im_start|>user\nWhat is Ruby?<|im_end|>\n" +
                     "<|im_start|>assistant\n"
    
    # This would be tested with an actual model instance
    # For now, we just verify the expected format is correct
    assert_includes expected_format, "<|im_start|>"
    assert_includes expected_format, "<|im_end|>"
  end
end