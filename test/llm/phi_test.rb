require_relative "../test_helper"

class PhiTest < Minitest::Test
  def test_phi_tokenizer_registry
    # Test exact matches
    assert_equal "microsoft/Phi-3-mini-4k-instruct", Candle::LLM.guess_tokenizer("microsoft/Phi-3-mini-4k-instruct-GGUF")
    
    # Test pattern matches
    assert_equal "microsoft/Phi-3-mini-4k-instruct", Candle::LLM.guess_tokenizer("TheBloke/Phi-3-mini-4k-instruct-GGUF")
    assert_equal "microsoft/phi-2", Candle::LLM.guess_tokenizer("TheBloke/phi-2-GGUF")
    assert_equal "microsoft/phi-2", Candle::LLM.guess_tokenizer("microsoft/phi-2-GGUF")
    
    # Test variations
    assert_equal "microsoft/Phi-3-medium-4k-instruct", Candle::LLM.guess_tokenizer("Phi-3-medium-GGUF")
    assert_equal "microsoft/Phi-3-small-8k-instruct", Candle::LLM.guess_tokenizer("Phi-3-small-GGUF")
  end
  
  def test_phi_model_type_detection
    skip "Requires Phi model download" unless ENV["RUN_FULL_TESTS"]
    
    # This would test actual model loading, but requires downloading models
    # which is expensive. Only run with RUN_FULL_TESTS=1
    model = Candle::LLM.from_pretrained("microsoft/phi-2")
    assert_equal "microsoft/phi-2", model.model_name
    
    # Test chat template
    messages = [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Hello!" }
    ]
    
    prompt = model.apply_chat_template(messages)
    assert prompt.include?("System:")
    assert prompt.include?("User:")
    assert prompt.include?("Assistant:")
  end
  
  def test_phi3_chat_template
    skip "Requires Phi-3 model download" unless ENV["RUN_FULL_TESTS"]
    
    model = Candle::LLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    messages = [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "What is Ruby?" }
    ]
    
    prompt = model.apply_chat_template(messages)
    assert prompt.include?("<|system|>")
    assert prompt.include?("<|user|>")
    assert prompt.include?("<|assistant|>")
    assert prompt.include?("<|end|>")
  end
  
  def test_phi_gguf_loading
    skip "Requires GGUF file" unless ENV["RUN_FULL_TESTS"]
    
    # Test GGUF loading with automatic tokenizer detection
    model = Candle::LLM.from_pretrained("TheBloke/phi-2-GGUF", gguf_file: "phi-2.Q4_K_M.gguf")
    assert_instance_of Candle::LLM, model
  end
end