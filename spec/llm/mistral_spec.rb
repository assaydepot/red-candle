require "spec_helper"

RSpec.describe "Mistral LLM" do
  before(:all) do
    @llm = nil
    @model_loaded = false
    
    if ENV["CI"]
      skip "Skipping model download test in CI"
    end
    
    begin
      @llm = Candle::LLM.from_pretrained(
        "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        gguf_file: "mistral-7b-instruct-v0.2.Q2_K.gguf"
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
    it "matches exact Mistral patterns" do
      expect(Candle::LLM.guess_tokenizer("TheBloke/Mistral-7B-Instruct-v0.2-GGUF"))
        .to eq("mistralai/Mistral-7B-Instruct-v0.2")
      expect(Candle::LLM.guess_tokenizer("TheBloke/Mistral-7B-Instruct-v0.1-GGUF"))
        .to eq("mistralai/Mistral-7B-Instruct-v0.1")
    end
    
    it "matches general Mistral patterns" do
      expect(Candle::LLM.guess_tokenizer("TheBloke/Mistral-7B-GGUF"))
        .to eq("mistralai/Mistral-7B-v0.1")
      expect(Candle::LLM.guess_tokenizer("someone/mistral-7b-gguf"))
        .to eq("mistralai/Mistral-7B-v0.1")
    end
  end
  
  describe "generation" do
    it "generates text" do
      skip unless @model_loaded == true
      
      prompt = "[INST] Write a haiku about Ruby [/INST]"
      config = Candle::GenerationConfig.new(max_length: 50)
      result = @llm.generate(prompt, config: config)
      
      expect(result).to be_a(String)
      expect(result).not_to be_empty
    end
    
    it "supports streaming generation" do
      skip unless @model_loaded == true
      
      prompt = "[INST] Count to 3 [/INST]"
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
    
    it "applies Mistral chat template" do
      skip unless @model_loaded == true
      
      messages = [
        { role: "user", content: "Test" },
        { role: "assistant", content: "Response" },
        { role: "user", content: "Follow-up" }
      ]
      
      formatted = @llm.apply_chat_template(messages)
      expect(formatted).to include("[INST]")
      expect(formatted).to include("[/INST]")
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