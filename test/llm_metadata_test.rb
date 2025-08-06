require_relative "test_helper"

class LLMMetadataTest < Minitest::Test
  # Use TinyLlama for testing as it's the smallest model
  @@llm = nil
  @@gguf_llm = nil
  @@model_loaded = false
  
  def self.load_models_once
    unless @@model_loaded
      begin
        # Load a GGUF model
        @@gguf_llm = Candle::LLM.from_pretrained(
          "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", 
          gguf_file: "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf",
          device: Candle::Device.cpu
        )
        
        # Load a non-GGUF model
        @@llm = Candle::LLM.from_pretrained(
          "microsoft/phi-2",
          device: Candle::Device.cpu
        )
        
        @@model_loaded = true
      rescue => e
        # If model loading fails, we'll still test what we can
        @@model_loaded = :failed
        @@load_error = e
      end
    end
  end
  
  def setup
    self.class.load_models_once
  end
  
  def test_model_id_getter
    skip "Model loading failed: #{@@load_error.message}" if @@model_loaded == :failed
    
    # Test GGUF model_id
    assert_respond_to @@gguf_llm, :model_id
    model_id = @@gguf_llm.model_id
    assert_instance_of String, model_id
    assert model_id.include?("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    assert model_id.include?("tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf")
    
    # Test non-GGUF model_id
    assert_respond_to @@llm, :model_id
    assert_equal "microsoft/phi-2", @@llm.model_id
  end
  
  def test_options_method
    skip "Model loading failed: #{@@load_error.message}" if @@model_loaded == :failed
    
    # Test GGUF model options
    assert_respond_to @@gguf_llm, :options
    options = @@gguf_llm.options
    assert_instance_of Hash, options
    
    # Check expected keys for GGUF model
    assert options.key?("model_id")
    assert options.key?("device")
    assert options.key?("model_type")
    assert options.key?("base_model")
    assert options.key?("gguf_file")
    assert options.key?("architecture")
    assert options.key?("eos_token_id")
    
    # Verify values
    assert_equal "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", options["base_model"]
    assert_equal "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf", options["gguf_file"]
    assert_equal "QuantizedGGUF", options["model_type"]
    assert_equal "llama", options["architecture"]
    assert_equal "cpu", options["device"]
    assert_kind_of Integer, options["eos_token_id"]
    
    # Test non-GGUF model options
    options = @@llm.options
    assert_instance_of Hash, options
    assert_equal "microsoft/phi-2", options["model_id"]
    assert_equal "Phi", options["model_type"]
    assert_equal "cpu", options["device"]
    
    # Non-GGUF shouldn't have GGUF-specific fields
    refute options.key?("base_model")
    refute options.key?("gguf_file")
    refute options.key?("architecture")
  end
  
  def test_options_with_custom_tokenizer
    # Test a model loaded with custom tokenizer
    begin
      model = Candle::LLM.from_pretrained(
        "google/gemma-3-4b-it-qat-q4_0-gguf",
        gguf_file: "gemma-3-4b-it-q4_0.gguf",
        tokenizer: "google/gemma-3-4b-it",
        device: Candle::Device.cpu
      )
      
      options = model.options
      assert_equal "google/gemma-3-4b-it", options["tokenizer_source"]
    rescue => e
      skip "Gemma model test skipped: #{e.message}"
    end
  end
  
  def test_inspect_method
    skip "Model loading failed: #{@@load_error.message}" if @@model_loaded == :failed
    
    # Test GGUF model inspect
    assert_respond_to @@gguf_llm, :inspect
    inspect_str = @@gguf_llm.inspect
    assert_instance_of String, inspect_str
    
    # Check format and content
    assert_match(/^#<Candle::LLM/, inspect_str)
    assert_match(/model=TheBloke\/TinyLlama-1.1B-Chat-v1.0-GGUF/, inspect_str)
    assert_match(/gguf=tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf/, inspect_str)
    assert_match(/device=cpu/, inspect_str)
    assert_match(/type=QuantizedGGUF/, inspect_str)
    assert_match(/arch=llama/, inspect_str)
    assert_match(/>$/, inspect_str)
    
    # Test non-GGUF model inspect
    inspect_str = @@llm.inspect
    assert_instance_of String, inspect_str
    assert_match(/^#<Candle::LLM/, inspect_str)
    assert_match(/model=microsoft\/phi-2/, inspect_str)
    assert_match(/device=cpu/, inspect_str)
    assert_match(/type=Phi/, inspect_str)
    assert_match(/>$/, inspect_str)
    
    # Should not include GGUF-specific fields
    refute_match(/gguf=/, inspect_str)
    refute_match(/arch=/, inspect_str)
  end
  
  def test_inspect_works_with_p_method
    skip "Model loading failed: #{@@load_error.message}" if @@model_loaded == :failed
    
    # Capture output from p method
    output = capture_io do
      p @@gguf_llm
    end.first.strip
    
    # p method should use inspect
    assert_equal @@gguf_llm.inspect, output
  end
  
  def test_device_getter_consistency
    skip "Model loading failed: #{@@load_error.message}" if @@model_loaded == :failed
    
    # Test that device getter returns the same as options["device"]
    device = @@gguf_llm.device
    options_device = @@gguf_llm.options["device"]
    
    # Both should represent the same device
    assert_equal "cpu", options_device
    assert_equal Candle::Device.cpu, device
  end
end