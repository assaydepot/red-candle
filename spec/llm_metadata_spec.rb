require "spec_helper"

RSpec.describe "LLMMetadata" do
  # Use TinyLlama for testing as it's the smallest model
  # Load models once using ModelCache module
  
  let(:gguf_llm) do
    ModelCache.gguf_llm
  rescue => e
    skip "GGUF model loading failed: #{e.message}"
  end
  
  let(:llm) do
    ModelCache.llm
  rescue => e
    skip "LLM model loading failed: #{e.message}"
  end
  
  describe "#model_id" do
    it "returns correct model_id for GGUF model" do
      expect(gguf_llm).to respond_to(:model_id)
      model_id = gguf_llm.model_id
      expect(model_id).to be_a(String)
      expect(model_id).to include("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
      expect(model_id).to include("tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf")
    end
    
    it "returns correct model_id for non-GGUF model" do
      expect(llm).to respond_to(:model_id)
      expect(llm.model_id).to eq("microsoft/phi-2")
    end
  end
  
  describe "#options" do
    context "for GGUF model" do
      let(:options) { gguf_llm.options }
      
      it "returns a hash of options" do
        expect(gguf_llm).to respond_to(:options)
        expect(options).to be_a(Hash)
      end
      
      it "includes all expected keys" do
        expect(options).to have_key("model_id")
        expect(options).to have_key("device")
        expect(options).to have_key("model_type")
        expect(options).to have_key("base_model")
        expect(options).to have_key("gguf_file")
        expect(options).to have_key("architecture")
        expect(options).to have_key("eos_token_id")
      end
      
      it "has correct values" do
        expect(options["base_model"]).to eq("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
        expect(options["gguf_file"]).to eq("tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf")
        expect(options["model_type"]).to eq("QuantizedGGUF")
        expect(options["architecture"]).to eq("llama")
        expect(options["device"]).to eq("cpu")
        expect(options["eos_token_id"]).to be_an(Integer)
      end
    end
    
    context "for non-GGUF model" do
      let(:options) { llm.options }
      
      it "returns a hash of options" do
        expect(options).to be_a(Hash)
      end
      
      it "has correct values" do
        expect(options["model_id"]).to eq("microsoft/phi-2")
        expect(options["model_type"]).to eq("Phi")
        expect(options["device"]).to eq("cpu")
      end
      
      it "doesn't have GGUF-specific fields" do
        expect(options).not_to have_key("base_model")
        expect(options).not_to have_key("gguf_file")
        expect(options).not_to have_key("architecture")
      end
    end
  end
  
  describe "#options with custom tokenizer" do
    it "includes tokenizer information when custom tokenizer is used" do
      # Test a model loaded with custom tokenizer
      begin
        model = Candle::LLM.from_pretrained(
          "google/gemma-3-4b-it-qat-q4_0-gguf",
          gguf_file: "gemma-3-4b-it-q4_0.gguf",
          tokenizer: "google/gemma-3-4b-it",
          device: Candle::Device.cpu
        )
        
        options = model.options
        expect(options["tokenizer_source"]).to eq("google/gemma-3-4b-it")
      rescue => e
        skip "Custom tokenizer test skipped: #{e.message}"
      end
    end
  end
  
  describe "#inspect" do
    it "provides meaningful inspect output for GGUF model" do
      inspect_str = gguf_llm.inspect
      expect(inspect_str).to include("#<Candle::LLM")
      expect(inspect_str).to include("model=")
      expect(inspect_str).to include("device=cpu")
      expect(inspect_str).to include("GGUF")
    end
    
    it "provides meaningful inspect output for non-GGUF model" do
      inspect_str = llm.inspect
      expect(inspect_str).to include("#<Candle::LLM")
      expect(inspect_str).to include("model=")
      expect(inspect_str).to include("type=Phi")
      expect(inspect_str).to include("device=cpu")
    end
  end
  
  describe "#tokenizer" do
    it "provides access to tokenizer for GGUF model" do
      expect(gguf_llm).to respond_to(:tokenizer)
      tokenizer = gguf_llm.tokenizer
      expect(tokenizer).to be_a(Candle::Tokenizer)
    end
    
    it "provides access to tokenizer for non-GGUF model" do
      expect(llm).to respond_to(:tokenizer)
      tokenizer = llm.tokenizer
      expect(tokenizer).to be_a(Candle::Tokenizer)
    end
  end
  
  describe "#device" do
    it "returns correct device for GGUF model" do
      expect(gguf_llm).to respond_to(:device)
      device = gguf_llm.device
      expect(device.to_s).to eq("cpu")
    end
    
    it "returns correct device for non-GGUF model" do
      expect(llm).to respond_to(:device)
      device = llm.device
      expect(device.to_s).to eq("cpu")
    end
  end
end