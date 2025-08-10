require "spec_helper"

RSpec.describe "Gemma LLM" do
  # Class variable to store the model and avoid reloading
  before(:all) do
    @llm = nil
    @model_loaded = false
    
    if ENV["CI"]
      skip "Skipping model download test in CI"
    end
    
    begin
      @llm = Candle::LLM.from_pretrained(
        "google/gemma-3-4b-it-qat-q4_0-gguf",
        gguf_file: "gemma-3-4b-it-q4_0.gguf",
        tokenizer: "google/gemma-3-4b-it"
      )
      @model_loaded = true
    rescue => e
      # If model loading fails (e.g., no internet), skip all LLM tests
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
    it "matches Google Gemma patterns" do
      expect(Candle::LLM.guess_tokenizer("google/gemma-2b-GGUF")).to eq("google/gemma-2b")
      expect(Candle::LLM.guess_tokenizer("google/gemma-2b-it-GGUF")).to eq("google/gemma-2b")
      expect(Candle::LLM.guess_tokenizer("google/gemma-7b-GGUF")).to eq("google/gemma-7b")
      expect(Candle::LLM.guess_tokenizer("google/gemma-7b-it-GGUF")).to eq("google/gemma-7b")
    end
    
    it "matches third-party Gemma patterns" do
      expect(Candle::LLM.guess_tokenizer("TheBloke/gemma-2b-GGUF")).to eq("google/gemma-2b")
      expect(Candle::LLM.guess_tokenizer("someone/gemma-7b-instruct-gguf")).to eq("google/gemma-7b")
      expect(Candle::LLM.guess_tokenizer("bartowski/gemma-2b-it-GGUF")).to eq("google/gemma-2b")
    end
    
    it "matches Gemma 2 patterns" do
      expect(Candle::LLM.guess_tokenizer("google/gemma-2-9b-it-GGUF")).to eq("google/gemma-2-9b")
      expect(Candle::LLM.guess_tokenizer("google/gemma-2-2b-it-GGUF")).to eq("google/gemma-2-2b")
      expect(Candle::LLM.guess_tokenizer("bartowski/gemma-2-9b-it-GGUF")).to eq("google/gemma-2-9b")
    end
    
    it "matches lmstudio patterns" do
      expect(Candle::LLM.guess_tokenizer("lmstudio-ai/gemma-2b-it-GGUF")).to eq("google/gemma-2b")
    end
  end
  
  describe "generation" do
    it "generates text" do
      skip unless @model_loaded == true
      
      prompt = "Write a haiku about Ruby"
      config = Candle::GenerationConfig.new(max_length: 50)
      result = @llm.generate(prompt, config: config)
      
      expect(result).to be_a(String)
      expect(result.length).to be > prompt.length
      # Gemma models may not include the prompt in output by default
      # Just check that we got a response
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
    
    it "applies Gemma chat template" do
      skip unless @model_loaded == true
      
      messages = [
        { role: "user", content: "Test" },
        { role: "assistant", content: "Response" },
        { role: "user", content: "Follow-up" }
      ]
      
      formatted = @llm.apply_chat_template(messages)
      expect(formatted).to include("<start_of_turn>")
      expect(formatted).to include("user")
      expect(formatted).to include("model")
      expect(formatted).to include("<end_of_turn>")
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