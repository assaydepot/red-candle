require "spec_helper"

RSpec.describe "Phi LLM" do
  before(:all) do
    @llm = nil
    @model_loaded = false
    
    if ENV["CI"]
      skip "Skipping model download test in CI"
    end
    
    begin
      @llm = Candle::LLM.from_pretrained(
        "TheBloke/phi-2-GGUF", 
        gguf_file: "phi-2.Q4_K_M.gguf"
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
    it "matches Phi patterns" do
      expect(Candle::LLM.guess_tokenizer("TheBloke/phi-2-GGUF"))
        .to eq("microsoft/phi-2")
      expect(Candle::LLM.guess_tokenizer("microsoft/phi-2-GGUF"))
        .to eq("microsoft/phi-2")
    end
    
    it "matches Phi-3 patterns" do
      expect(Candle::LLM.guess_tokenizer("microsoft/Phi-3-mini-4k-instruct-gguf"))
        .to eq("microsoft/Phi-3-mini-4k-instruct")
      expect(Candle::LLM.guess_tokenizer("bartowski/Phi-3.5-mini-instruct-GGUF"))
        .to eq("microsoft/Phi-3.5-mini-instruct")
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
  
  describe "structured generation" do
    it "generates structured output with JSON schema" do
      skip unless @model_loaded == true
      # Executing without skip to test structured generation
      
      schema = {
        type: "object",
        properties: {
          name: { type: "string" },
          age: { type: "integer", minimum: 0, maximum: 120 }
        },
        required: ["name", "age"]
      }
      
      # Create constraint from schema
      constraint = @llm.constraint_from_schema(schema)
      
      config = Candle::GenerationConfig.new(
        max_length: 100,
        constraint: constraint,
        stop_on_constraint_satisfaction: false  # Important: don't stop early
      )
      
      prompt = "Generate a person with name John and age 30:"
      result = @llm.generate(prompt, config: config)
      
      json_str = result.sub(prompt, "").strip
      parsed = JSON.parse(json_str)
      
      expect(parsed).to be_a(Hash)
      expect(parsed["name"]).to be_a(String)
      expect(parsed["age"]).to be_a(Integer)
    end
  end
  
  describe "metadata" do
    it "has expected model methods" do
      skip unless @model_loaded == true
      
      expect(@llm).to respond_to(:generate)
      expect(@llm).to respond_to(:apply_chat_template)
    end
  end
end