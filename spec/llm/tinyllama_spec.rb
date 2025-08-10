require "spec_helper"
require "support/llama_shared_examples"

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
  
  # Use shared examples for common Llama architecture tests
  it_behaves_like "llama architecture model", "TinyLlama"
  
  # TinyLlama-specific tests
  describe "generation config presets" do
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
    
    it "chains config modifications with 'with' method" do
      config = Candle::GenerationConfig.default
      new_config = config.with(temperature: 0.9, max_length: 200)
      
      # Original should be unchanged
      expect(config.temperature).to eq(0.7)
      expect(config.max_length).to eq(512)
      
      # New config should have updated values
      expect(new_config.temperature).to eq(0.9)
      expect(new_config.max_length).to eq(200)
    end
  end
  
  describe "tokenizer registry" do
    it "finds TinyLlama tokenizer automatically" do
      expect(Candle::LLM.guess_tokenizer("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"))
        .to eq("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
      
      # Should also work for variants
      expect(Candle::LLM.guess_tokenizer("someone/tinyllama-1.1b-gguf"))
        .to eq("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    end
    
    it "registers custom tokenizers" do
      # Test exact match registration
      Candle::LLM.register_tokenizer("custom/tinyllama-variant", "custom/tokenizer")
      expect(Candle::LLM.guess_tokenizer("custom/tinyllama-variant"))
        .to eq("custom/tokenizer")
      
      # Test pattern registration
      Candle::LLM.register_tokenizer(/tiny.*custom/i, "custom/tiny-tokenizer")
      expect(Candle::LLM.guess_tokenizer("user/tiny-model-custom"))
        .to eq("custom/tiny-tokenizer")
    end
  end
  
  describe "fast CI verification" do
    it "loads and generates quickly for CI testing" do
      skip unless @model_loaded == true
      
      # This test ensures TinyLlama can be used as a fast smoke test in CI
      start_time = Time.now
      config = Candle::GenerationConfig.new(max_length: 10, temperature: 0.0)
      result = @llm.generate("Test", config: config)
      elapsed = Time.now - start_time
      
      expect(result).not_to be_empty
      expect(elapsed).to be < 2.0  # Should complete quickly
    end
  end
end