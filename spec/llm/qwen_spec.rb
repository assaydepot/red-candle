require "spec_helper"

RSpec.describe "Qwen LLM" do
  before(:all) do
    @llm = nil
    @model_loaded = false
    @device = Candle::Device.cpu
    
    if ENV["CI"]
      skip "Skipping model download test in CI"
    end
    
    begin
      @llm = Candle::LLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        device: @device,
        gguf_file: "qwen2.5-1.5b-instruct-q4_k_m.gguf"
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
  
  describe "tokenizer registry" do
    it "matches Qwen2 patterns" do
      # Qwen2 models get mapped to Qwen2-1.5B tokenizer (base version)
      expect(Candle::LLM.guess_tokenizer("Qwen/Qwen2-7B-GGUF"))
        .to eq("Qwen/Qwen2-1.5B")
      expect(Candle::LLM.guess_tokenizer("Qwen/Qwen2-7B-Instruct-GGUF"))
        .to eq("Qwen/Qwen2-1.5B")
    end
    
    it "matches Qwen2.5 patterns" do
      expect(Candle::LLM.guess_tokenizer("Qwen/Qwen2.5-0.5B-GGUF"))
        .to eq("Qwen/Qwen2.5-0.5B")
      # All Qwen2.5 models map to the 0.5B tokenizer
      expect(Candle::LLM.guess_tokenizer("Qwen/Qwen2.5-1.5B-Instruct-GGUF"))
        .to eq("Qwen/Qwen2.5-0.5B")
      expect(Candle::LLM.guess_tokenizer("Qwen/Qwen2.5-7B-Instruct-GGUF"))
        .to eq("Qwen/Qwen2.5-0.5B")
    end
    
    it "matches third-party Qwen patterns" do
      # Third-party models map to available Qwen tokenizers
      expect(Candle::LLM.guess_tokenizer("bartowski/Qwen2.5-7B-Instruct-GGUF"))
        .to match(/Qwen2.5/)
      expect(Candle::LLM.guess_tokenizer("lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF"))
        .to match(/Qwen2.5.*Coder|Qwen2.5/)
    end
  end
  
  describe "generation" do
    it "generates text" do
      skip unless @model_loaded == true
      
      prompt = "Write a haiku about Ruby:"
      config = Candle::GenerationConfig.new(max_length: 50)
      result = @llm.generate(prompt, config: config)
      
      expect(result).to be_a(String)
      expect(result).not_to be_empty
    end
    
    it "supports streaming generation" do
      skip unless @model_loaded == true
      
      prompt = "Count to 3:"
      chunks = []
      
      config = Candle::GenerationConfig.new(max_length: 30)
      @llm.generate_stream(prompt, config: config) do |chunk|
        chunks << chunk
      end
      
      expect(chunks).not_to be_empty
      expect(chunks.join).to be_a(String)
    end
  end
  
  describe "chat interface" do
    it "handles chat messages" do
      skip unless @model_loaded == true
      
      messages = [
        { role: "user", content: "Hello" }
      ]
      
      config = Candle::GenerationConfig.new(max_length: 50)
      response = @llm.chat(messages, config: config)
      expect(response).to be_a(String)
      expect(response).not_to be_empty
    end
    
    it "applies Qwen chat template" do
      skip unless @model_loaded == true
      
      messages = [
        { role: "system", content: "You are helpful" },
        { role: "user", content: "Test" },
        { role: "assistant", content: "Response" },
        { role: "user", content: "Follow-up" }
      ]
      
      formatted = @llm.apply_chat_template(messages)
      expect(formatted).to include("<|im_start|>")
      expect(formatted).to include("<|im_end|>")
      expect(formatted).to include("system")
      expect(formatted).to include("user")
      expect(formatted).to include("assistant")
    end
  end
  
  describe "metadata" do
    it "has expected model methods" do
      skip unless @model_loaded == true
      
      expect(@llm).to respond_to(:generate)
      expect(@llm).to respond_to(:chat)
      expect(@llm).to respond_to(:apply_chat_template)
    end
  end
end