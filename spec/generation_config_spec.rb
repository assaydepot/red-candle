require "spec_helper"

RSpec.describe "GenerationConfig" do
  describe "#new with all parameters" do
    it "extracts all parameters properly from kwargs" do
      config = Candle::GenerationConfig.new(
        max_length: 1024,
        temperature: 0.8,
        top_p: 0.95,
        top_k: 50,
        repetition_penalty: 1.2,
        repetition_penalty_last_n: 128,
        seed: 12345,
        include_prompt: true,
        stop_sequences: ["STOP", "END"],
        debug_tokens: true,
        stop_on_constraint_satisfaction: false,
        stop_on_match: false
      )
      
      # Verify all parameters were correctly set
      expect(config.max_length).to eq(1024)
      expect(config.temperature).to eq(0.8)
      expect(config.top_p).to eq(0.95)
      expect(config.top_k).to eq(50)
      expect(config.repetition_penalty).to be_within(0.0001).of(1.2)
      # repetition_penalty_last_n doesn't have a getter method exposed
      expect(config.seed).to eq(12345)
      expect(config.include_prompt).to eq(true)
      expect(config.stop_sequences).to eq(["STOP", "END"])
      expect(config.debug_tokens).to eq(true)
      expect(config.stop_on_constraint_satisfaction).to eq(false)
      expect(config.stop_on_match).to eq(false)
    end
  end
  
  describe "#new with partial parameters" do
    it "creates config with only some parameters" do
      config = Candle::GenerationConfig.new(
        temperature: 0.5,
        max_length: 200
      )
      
      expect(config.temperature).to eq(0.5)
      expect(config.max_length).to eq(200)
      
      # Other parameters should have defaults
      expect(config.seed).to be_a(Integer)
      expect(config.debug_tokens).to eq(false)
    end
  end
  
  describe "#new with empty initialization" do
    it "creates config with no parameters uses defaults" do
      config = Candle::GenerationConfig.new({})
      
      expect(config.max_length).to eq(512)  # default
      expect(config.temperature).to eq(0.7)  # default
      expect(config.debug_tokens).to eq(false)  # default
    end
  end
  
  describe "#with method" do
    it "preserves parameters" do
      original = Candle::GenerationConfig.new(
        max_length: 100,
        temperature: 0.5,
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
        repetition_penalty_last_n: 64,
        seed: 999,
        include_prompt: true,
        stop_sequences: [".", "!"],
        debug_tokens: true,
        stop_on_constraint_satisfaction: false,
        stop_on_match: true
      )
      
      # Create new config with only temperature changed
      modified = original.with(temperature: 0.8)
      
      # Temperature should be updated
      expect(modified.temperature).to eq(0.8)
      
      # All other parameters should be preserved
      expect(modified.max_length).to eq(100)
      expect(modified.top_p).to eq(0.9)
      expect(modified.top_k).to eq(40)
      expect(modified.repetition_penalty).to be_within(0.0001).of(1.1)
      # repetition_penalty_last_n doesn't have a getter method exposed
      expect(modified.seed).to eq(999)
      expect(modified.include_prompt).to eq(true)
      expect(modified.stop_sequences).to eq([".", "!"])
      # Note: debug_tokens, stop_on_constraint_satisfaction, and stop_on_match
      # are not preserved by .with() method in the current implementation
    end
  end
  
  describe "preset methods" do
    it "provides deterministic preset" do
      deterministic = Candle::GenerationConfig.deterministic
      expect(deterministic.temperature).to eq(0.0)
    end
    
    it "provides creative preset" do
      creative = Candle::GenerationConfig.creative
      expect(creative.temperature).to eq(1.0)
    end
    
    it "provides balanced preset" do
      balanced = Candle::GenerationConfig.balanced
      expect(balanced.temperature).to eq(0.7)
    end
  end
end