require "spec_helper"

RSpec.describe "Llama LLM" do
  before(:all) do
    @llm = nil
    @model_loaded = false
    
    if ENV["CI"]
      skip "Skipping model download test in CI"
    end
    
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
  
  describe "tokenizer registry" do
    it "matches exact Llama patterns" do
      expect(Candle::LLM.guess_tokenizer("TheBloke/Llama-2-7B-GGUF"))
        .to eq("meta-llama/Llama-2-7b-hf")
      expect(Candle::LLM.guess_tokenizer("TheBloke/Llama-2-7B-Chat-GGUF"))
        .to eq("meta-llama/Llama-2-7b-chat-hf")
    end
    
    it "matches Llama 2 size variants" do
      expect(Candle::LLM.guess_tokenizer("TheBloke/Llama-2-13B-GGUF"))
        .to eq("meta-llama/Llama-2-13b-hf")
      expect(Candle::LLM.guess_tokenizer("TheBloke/Llama-2-13B-Chat-GGUF"))
        .to eq("meta-llama/Llama-2-13b-chat-hf")
      expect(Candle::LLM.guess_tokenizer("TheBloke/Llama-2-70B-GGUF"))
        .to eq("meta-llama/Llama-2-70b-hf")
      expect(Candle::LLM.guess_tokenizer("TheBloke/Llama-2-70B-Chat-GGUF"))
        .to eq("meta-llama/Llama-2-70b-chat-hf")
    end
    
    it "matches general Llama patterns" do
      expect(Candle::LLM.guess_tokenizer("someone/llama-2-7b-gguf"))
        .to eq("meta-llama/Llama-2-7b-hf")
      expect(Candle::LLM.guess_tokenizer("bartowski/Meta-Llama-3-8B-Instruct-GGUF"))
        .to eq("meta-llama/Meta-Llama-3-8B-Instruct")
    end
  end
  
  describe "generation" do
    it "generates text" do
      skip unless @model_loaded == true
      
      prompt = "<s>[INST] Write a haiku about Ruby [/INST]"
      config = Candle::GenerationConfig.new(max_length: 50)
      result = @llm.generate(prompt, config: config)
      
      expect(result).to be_a(String)
      expect(result).not_to be_empty
    end
    
    it "supports streaming generation" do
      skip unless @model_loaded == true
      
      prompt = "<s>[INST] Count to 3 [/INST]"
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
    
    it "applies Llama 2 chat template" do
      skip unless @model_loaded == true
      
      messages = [
        { role: "system", content: "You are helpful" },
        { role: "user", content: "Test" },
        { role: "assistant", content: "Response" },
        { role: "user", content: "Follow-up" }
      ]
      
      formatted = @llm.apply_chat_template(messages)
      expect(formatted).to include("[INST]")
      expect(formatted).to include("[/INST]")
      expect(formatted).to include("<<SYS>>") if messages.any? { |m| m[:role] == "system" }
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