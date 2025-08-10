RSpec.shared_examples "llama architecture model" do |model_name|
  describe "generation" do
    it "generates text" do
      skip unless @model_loaded == true
      
      prompt = "Hello, my name is"
      config = Candle::GenerationConfig.new(max_length: 20, temperature: 0.7)
      result = @llm.generate(prompt, config: config)
      
      expect(result).to be_a(String)
      expect(result).not_to be_empty
      expect(result.length).to be > 5  # At least some generation happened
    end
    
    it "supports streaming generation" do
      skip unless @model_loaded == true
      
      prompt = "Once upon a time"
      chunks = []
      
      config = Candle::GenerationConfig.new(max_length: 20, temperature: 0.7)
      @llm.generate_stream(prompt, config: config) do |chunk|
        chunks << chunk
      end
      
      expect(chunks).not_to be_empty
      expect(chunks.size).to be >= 2  # At least some streaming happened
      full_text = chunks.join
      expect(full_text).to be_a(String)
      expect(full_text.length).to be > 5
    end
    
    it "respects max_length configuration" do
      skip unless @model_loaded == true
      
      config_short = Candle::GenerationConfig.new(max_length: 10, temperature: 0.0, seed: 42)
      config_long = Candle::GenerationConfig.new(max_length: 50, temperature: 0.0, seed: 42)
      
      result_short = @llm.generate("Tell me", config: config_short)
      result_long = @llm.generate("Tell me", config: config_long)
      
      # The longer config should generate more text
      expect(result_long.length).to be > result_short.length
    end
  end
  
  describe "chat interface" do
    it "handles chat messages" do
      skip unless @model_loaded == true
      
      messages = [
        { role: "user", content: "Hi!" }
      ]
      
      config = Candle::GenerationConfig.new(max_length: 30, temperature: 0.7)
      response = @llm.chat(messages, config: config)
      
      expect(response).to be_a(String)
      expect(response).not_to be_empty
    end
    
    it "applies chat template correctly" do
      skip unless @model_loaded == true
      
      messages = [
        { role: "system", content: "You are helpful" },
        { role: "user", content: "Test" }
      ]
      
      formatted = @llm.apply_chat_template(messages)
      expect(formatted).to be_a(String)
      expect(formatted).to include("Test")
      expect(formatted).to include("helpful")
      
      # Check for Llama-style formatting markers
      expect(formatted).to match(/<\|.*\|>|\[INST\]|User:|### Instruction/)
    end
  end
  
  describe "configuration" do
    it "uses deterministic generation with seed" do
      skip unless @model_loaded == true
      
      config = Candle::GenerationConfig.deterministic.with(max_length: 15)
      result1 = @llm.generate("Hello", config: config)
      result2 = @llm.generate("Hello", config: config)
      
      expect(result1).to eq(result2)  # Same seed should produce same output
    end
    
    it "responds to all expected methods" do
      skip unless @model_loaded == true
      
      expect(@llm).to respond_to(:generate)
      expect(@llm).to respond_to(:generate_stream)
      expect(@llm).to respond_to(:chat)
      expect(@llm).to respond_to(:chat_stream)
      expect(@llm).to respond_to(:apply_chat_template)
      expect(@llm).to respond_to(:tokenizer)
    end
  end
end