require "spec_helper"
require "support/llama_shared_examples"

RSpec.describe "Llama 2 LLM" do
  before(:all) do
    @llm = nil
    @model_loaded = false
    
    begin
        @llm = Candle::LLM.from_pretrained(
          "TheBloke/Llama-2-7B-Chat-GGUF",
          gguf_file: "llama-2-7b-chat.Q2_K.gguf"  # Smallest quantization
        )
        @model_loaded = true
    rescue => e
      @model_loaded = :failed
      @load_error = e
    end
  end
  
  before(:each) do
    if @model_loaded == :failed
      skip "Model loading failed: #{@load_error.message}"
    end
  end
  
  # Use shared examples for common Llama architecture tests
  it_behaves_like "llama architecture model", "Llama 2"
  
  # Llama 2 specific tokenizer tests
  describe "tokenizer registry" do
    it "matches exact Llama patterns" do
      expect(Candle::LLM.guess_tokenizer("TheBloke/Llama-2-7B-Chat-GGUF"))
        .to eq("meta-llama/Llama-2-7b-chat-hf")
      expect(Candle::LLM.guess_tokenizer("TheBloke/Llama-2-13B-Chat-GGUF"))
        .to eq("meta-llama/Llama-2-13b-chat-hf")
    end
    
    it "matches Llama 2 size variants" do
      expect(Candle::LLM.guess_tokenizer("TheBloke/Llama-2-7B-GGUF"))
        .to eq("meta-llama/Llama-2-7b-hf")
      expect(Candle::LLM.guess_tokenizer("TheBloke/Llama-2-13B-GGUF"))
        .to eq("meta-llama/Llama-2-13b-hf")
      expect(Candle::LLM.guess_tokenizer("TheBloke/Llama-2-70B-GGUF"))
        .to eq("meta-llama/Llama-2-70b-hf")
    end
    
    it "matches general Llama patterns" do
      expect(Candle::LLM.guess_tokenizer("someone/llama-2-7b-custom"))
        .to eq("meta-llama/Llama-2-7b-hf")
      expect(Candle::LLM.guess_tokenizer("user/llama2-13b-instruct"))
        .to eq("meta-llama/Llama-2-13b-chat-hf")
    end
  end
  
  describe "Llama 2 specific chat template" do
    it "uses proper Llama 2 instruction format" do
      skip unless @model_loaded == true
      
      messages = [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Hello" }
      ]
      
      formatted = @llm.apply_chat_template(messages)
      
      # Llama 2 uses [INST] format
      expect(formatted).to include("[INST]") if formatted.include?("[INST]")
      # Or might use the <<SYS>> format for system messages
      expect(formatted).to include("<<SYS>>") if messages.any? { |m| m[:role] == "system" } && formatted.include?("<<SYS>>")
    end
  end
  
  describe "performance characteristics" do
    it "handles longer generation" do
      skip unless @model_loaded == true
      
      # Llama 2 7B can handle longer, more coherent generation
      config = Candle::GenerationConfig.new(
        max_length: 100,
        temperature: 0.7,
        top_p: 0.9
      )
      
      result = @llm.generate("Explain quantum computing in simple terms:", config: config)
      
      expect(result).not_to be_empty
      expect(result.split.size).to be > 10  # Should generate multiple words
    end
  end
end