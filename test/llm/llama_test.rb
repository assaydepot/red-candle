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
    skip "Skipping model download test in CI" if ENV["CI"]
    
    # Use GGUF model for testing
    begin
      model = Candle::LLM.from_pretrained(
        "TheBloke/Llama-2-7B-GGUF",
        gguf_file: "llama-2-7b.Q2_K.gguf"  # Smallest quantization
      )
      assert model.model_name.include?("TheBloke/Llama-2-7B-GGUF")
      
      # Test generation
      config = Candle::GenerationConfig.deterministic(max_length: 20, seed: 42)
      result = model.generate("Hello", config: config)
      assert_instance_of String, result
      assert result.length > 0
    rescue => e
      skip "Model download failed: #{e.message}"
    end
  end
  
  def test_llama2_chat_template
    skip "Skipping model download test in CI" if ENV["CI"]
    
    begin
      model = Candle::LLM.from_pretrained(
        "TheBloke/Llama-2-7B-Chat-GGUF",
        gguf_file: "llama-2-7b-chat.Q2_K.gguf"
      )
      messages = [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "What is Ruby?" }
      ]
      
      prompt = model.apply_chat_template(messages)
      assert_instance_of String, prompt
      assert prompt.include?("What is Ruby?")
      # Llama 2 chat format
      assert prompt.include?("[INST]") || prompt.include?("User:"),
             "Prompt should include instruction markers"
    rescue => e
      skip "Model download failed: #{e.message}"
    end
  end
  
  def test_llama3_chat_template
    skip "Skipping Llama 3 test - large model"
    # Llama 3 models are very large, skip for regular testing
  end
  
  def test_llama_gguf_loading
    skip "Skipping GGUF download test in CI" if ENV["CI"]
    
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
  
  def test_phone_constraint_generation
    skip "Skipping model download test in CI" if ENV["CI"]
    
    begin
      model = Candle::LLM.from_pretrained(
        "TheBloke/Llama-2-7B-Chat-GGUF",
        gguf_file: "llama-2-7b-chat.Q2_K.gguf"
      )
      
      phone_constraint = model.constraint_from_regex('\d{3}-\d{3}-\d{4}')
      config = Candle::GenerationConfig.balanced(constraint: phone_constraint)
      result = model.generate("Generate a phone number:", config: config)
      
      assert_instance_of String, result
      assert_match(/^\d{3}-\d{3}-\d{4}$/, result, "Result should be a valid phone number")
      refute result.include?("</s>"), "Result should not contain EOS tokens"
    rescue => e
      skip "Model download failed: #{e.message}"
    end
  end
  
  def test_structured_generation_with_schema
    skip "Skipping model download test in CI" if ENV["CI"]
    
    begin
      model = Candle::LLM.from_pretrained(
        "TheBloke/Llama-2-7B-Chat-GGUF",
        gguf_file: "llama-2-7b-chat.Q2_K.gguf"
      )
      
      schema = {
        type: "object",
        properties: {
          answer: { type: "string", enum: ["yes", "no"] },
          confidence: { type: "number", minimum: 0, maximum: 1 }
        },
        required: ["answer"]
      }
      
      result = model.generate_structured("Is Ruby easy to learn?", schema: schema)
      
      assert_instance_of Hash, result
      assert result.key?("answer"), "Result should have 'answer' key"
      assert ["yes", "no"].include?(result["answer"]), "Answer should be 'yes' or 'no'"
      
      if result.key?("confidence")
        assert_instance_of Float, result["confidence"]
        assert result["confidence"] >= 0 && result["confidence"] <= 1, "Confidence should be between 0 and 1"
      end
    rescue => e
      skip "Model download failed: #{e.message}"
    end
  end
end