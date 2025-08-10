require "spec_helper"

RSpec.describe "TinyLlama LLM" do
  before(:all) do
    @llm = nil
    @model_loaded = false
    
    begin
      @llm = Candle::LLM.from_pretrained(
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", 
        gguf_file: "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf"
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
  
  describe "generation config" do
    it "creates default config" do
      config = Candle::GenerationConfig.default
      expect(config.max_length).to eq(512)
      expect(config.temperature).to eq(0.7)
    end
    
    it "creates deterministic config" do
      config = Candle::GenerationConfig.deterministic
      expect(config.temperature).to eq(0.0)
      expect(config.seed).not_to be_nil
    end
    
    it "creates creative config" do
      config = Candle::GenerationConfig.creative
      expect(config.temperature).to eq(1.0)
      expect(config.top_p).to be < 1.0
    end
    
    it "creates balanced config" do
      config = Candle::GenerationConfig.balanced
      expect(config.temperature).to be_between(0.5, 1.0)
      expect(config.repetition_penalty).to be > 1.0
    end
  end
  
  describe "generation" do
    it "generates text with default config" do
      skip unless @model_loaded == true
      
      prompt = "Hello, my name is"
      config = Candle::GenerationConfig.new(max_length: 20)
      result = @llm.generate(prompt, config: config)
      
      expect(result).to be_a(String)
      expect(result).not_to be_empty
    end
    
    it "generates with custom config" do
      skip unless @model_loaded == true
      
      config = Candle::GenerationConfig.new(
        max_length: 30,
        temperature: 0.5,
        top_p: 0.9
      )
      
      prompt = "Count to 5:"
      result = @llm.generate(prompt, config: config)
      
      expect(result).to be_a(String)
      expect(result).not_to be_empty
    end
    
    it "supports streaming generation" do
      skip unless @model_loaded == true
      
      prompt = "Once upon a time"
      chunks = []
      
      config = Candle::GenerationConfig.new(max_length: 20)
      @llm.generate_stream(prompt, config: config) do |chunk|
        chunks << chunk
      end
      
      expect(chunks).not_to be_empty
      expect(chunks.join).to be_a(String)
    end
  end
  
  describe "chat interface" do
    it "handles single message" do
      skip unless @model_loaded == true
      
      messages = [
        { role: "user", content: "Hi!" }
      ]
      
      config = Candle::GenerationConfig.new(max_length: 30)
      response = @llm.chat(messages, config: config)
      expect(response).to be_a(String)
      expect(response).not_to be_empty
    end
    
    it "handles conversation" do
      skip unless @model_loaded == true
      
      messages = [
        { role: "user", content: "What's 2+2?" },
        { role: "assistant", content: "2+2 equals 4." },
        { role: "user", content: "And 3+3?" }
      ]
      
      config = Candle::GenerationConfig.new(max_length: 30)
      response = @llm.chat(messages, config: config)
      expect(response).to be_a(String)
    end
    
    it "applies TinyLlama chat template" do
      skip unless @model_loaded == true
      
      messages = [
        { role: "system", content: "You are a helpful assistant" },
        { role: "user", content: "Hello" }
      ]
      
      formatted = @llm.apply_chat_template(messages)
      expect(formatted).to be_a(String)
      # TinyLlama uses Llama format
      expect(formatted).to include("<|")
    end
  end
  
  describe "tokenizer registry" do
    it "finds TinyLlama tokenizer" do
      expect(Candle::LLM.guess_tokenizer("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"))
        .to eq("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
      expect(Candle::LLM.guess_tokenizer("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"))
        .to eq("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    end
  end
  
  describe "metadata" do
    it "has model configuration" do
      skip unless @model_loaded == true
      
      expect(@llm).to respond_to(:generate)
      expect(@llm).to respond_to(:chat)
      expect(@llm).to respond_to(:apply_chat_template)
    end
  end
end