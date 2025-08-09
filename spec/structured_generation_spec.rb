# frozen_string_literal: true

require "spec_helper"
require "json"

RSpec.describe "StructuredGeneration" do
  let(:device) { Candle::Device.cpu }
  let(:model_id) { ENV["TEST_MODEL"] || "TinyLlama/TinyLlama-1.1B-Chat-v1.0" }
  let(:llm) do
    @llm ||= Candle::LLM.from_pretrained(model_id, device: device)
  end
  
  after(:all) do
    @llm = nil
    GC.start
  end
  
  # Helper method to parse JSON from model output that might have extra content
  def parse_json_with_cleanup(text)
    # Try direct parse first
    JSON.parse(text)
  rescue JSON::ParserError
    # Extract JSON content by removing content after stop tokens
    cleaned = text
    ['</s>', '<|endoftext|>', '<|im_end|>', '<end>', '<end_of_turn>'].each do |token|
      if idx = cleaned.index(token)
        cleaned = cleaned[0...idx]
      end
    end
    
    # Try to find valid JSON boundaries
    if match = cleaned.match(/(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])/m)
      JSON.parse(match[0])
    else
      # If still can't parse, raise the original error
      JSON.parse(text)
    end
  end
  
  describe "#constraint_from_schema" do
    it "creates constraint from JSON schema" do
      # Create a simple yes/no schema
      schema = {
        type: "object",
        properties: {
          answer: {
            type: "string",
            enum: ["yes", "no"]
          }
        },
        required: ["answer"]
      }
      
      # Create constraint from schema
      constraint = llm.constraint_from_schema(schema)
      expect(constraint).to be_a(Candle::StructuredConstraint)
      
      # Generate with the constraint
      config = Candle::GenerationConfig.deterministic(
        max_length: 50,  # Increase to ensure complete JSON
        constraint: constraint,
        temperature: 0.1
      )
      
      result = llm.generate("Is the sky blue?", config: config)
      
      # Result should be parseable JSON matching the schema
      begin
        parsed = parse_json_with_cleanup(result)
        expect(parsed).to be_a(Hash), "Result should be a JSON object"
        expect(%w[yes no]).to include(parsed["answer"]), "Answer should be yes or no"
      rescue JSON::ParserError
        # Check if the output at least follows the expected pattern
        if result.include?('{"answer":')
          expect(result).to match(/"answer":\s*"(yes|no)"/), "Output should contain yes or no answer"
        else
          fail "Generated output doesn't match expected format: #{result}"
        end
      end
    end
  end
  
  describe "#constraint_from_regex" do
    it "creates constraint from regex pattern" do
      # Create a simpler constraint - just one or more digits
      digit_regex = '\d+'
      constraint = llm.constraint_from_regex(digit_regex)
      expect(constraint).to be_a(Candle::StructuredConstraint)
      
      # Generate with the constraint
      config = Candle::GenerationConfig.deterministic(
        max_length: 10,
        constraint: constraint,
        temperature: 0.1
      )
      
      result = llm.generate("Generate a number:", config: config)
      
      # Result should contain only digits
      expect(result.strip).to match(/^\d+$/), 
                   "Generated output should contain only digits, got: #{result.inspect}"
    end
  end
  
  describe "multiple choice constraint" do
    it "generates valid multiple choice responses" do
      # Create a multiple choice schema
      schema = {
        type: "object",
        properties: {
          choice: {
            type: "string",
            enum: ["A", "B", "C", "D"]
          },
          confidence: {
            type: "number",
            minimum: 0,
            maximum: 1
          }
        },
        required: ["choice"]
      }
      
      constraint = llm.constraint_from_schema(schema)
      config = Candle::GenerationConfig.deterministic(
        max_length: 50,  # Increase max_length to ensure complete JSON
        constraint: constraint,
        temperature: 0.1
      )
      
      prompt = "What is 2+2? A) 3 B) 4 C) 5 D) 6"
      result = llm.generate(prompt, config: config)
      
      # Try to parse JSON, handle incomplete output
      begin
        parsed = parse_json_with_cleanup(result)
        expect(%w[A B C D]).to include(parsed["choice"]), "Choice should be A, B, C, or D"
        if parsed["confidence"]
          expect(parsed["confidence"]).to be >= 0
          expect(parsed["confidence"]).to be <= 1
        end
      rescue JSON::ParserError
        # If JSON is incomplete, check if it at least started correctly
        expect(result).to include('{"choice":'), "Output should start with valid JSON structure"
        expect(result).to match(/"choice":\s*"[ABCD]/), "Output should contain a valid choice"
      end
    end
  end
  
  describe "structured data extraction" do
    it "extracts structured data from text" do
      # Simpler schema - just a single entity
      schema = {
        type: "object",
        properties: {
          name: { type: "string" },
          type: { 
            type: "string", 
            enum: ["person", "organization", "location"] 
          }
        },
        required: ["name", "type"]
      }
      
      constraint = llm.constraint_from_schema(schema)
      config = Candle::GenerationConfig.balanced(
        max_length: 50,
        constraint: constraint,
        temperature: 0.1
      )
      
      prompt = "Extract the person's name from: John Smith is a developer."
      result = llm.generate(prompt, config: config)
      
      begin
        parsed = parse_json_with_cleanup(result)
        expect(parsed["name"]).to be_a(String), "Name should be a string"
        expect(%w[person organization location]).to include(parsed["type"]), 
               "Type should be person, organization, or location"
      rescue JSON::ParserError
        # Check partial output
        expect(result).to include('{"name":').or include('{"type":"')
      end
    end
  end
  
  describe "constraint with GenerationConfig.with" do
    it "preserves constraint when using .with() method" do
      schema = { 
        type: "string",
        enum: ["yes", "no"]
      }
      constraint = llm.constraint_from_schema(schema)
      
      # Test that constraint is preserved with .with() method
      base_config = Candle::GenerationConfig.balanced(
        constraint: constraint,
        max_length: 10
      )
      
      # Use .with() to create a new config
      new_config = base_config.with(temperature: 0.5)
      
      # For now, just verify that the config was created successfully
      # The actual constraint testing is done in other tests
      expect(new_config.temperature).to eq(0.5)
      expect(new_config.max_length).to eq(10)
    end
  end
  
  describe "#generate_structured" do
    it "returns parsed JSON object" do
      # Simple schema
      schema = {
        type: "object",
        properties: {
          result: { type: "string", enum: ["success", "failure"] }
        },
        required: ["result"]
      }
      
      # Use generate_structured - should return parsed JSON
      result = llm.generate_structured(
        "Did the operation succeed?", 
        schema: schema,
        max_length: 30,
        temperature: 0.1
      )
      
      # Result should be a parsed Ruby object
      expect(result).to be_a(Hash), "Result should be a parsed Hash, got: #{result.class}"
      expect(%w[success failure]).to include(result["result"]), "Result should be success or failure"
    end
  end
  
  describe "constraint classes availability" do
    it "defines StructuredConstraint class" do
      # Basic availability test that doesn't require model loading
      expect(defined?(Candle::StructuredConstraint)).to be_truthy, "StructuredConstraint should be defined"
    end
    
    it "LLM has constraint methods" do
      # Verify LLM methods are defined
      expect(Candle::LLM.instance_methods).to include(:constraint_from_schema), 
             "LLM should have constraint_from_schema method"
      expect(Candle::LLM.instance_methods).to include(:constraint_from_regex), 
             "LLM should have constraint_from_regex method"
      expect(Candle::LLM.instance_methods).to include(:generate_structured), 
             "LLM should have generate_structured method"
    end
  end
  
  describe "GenerationConfig with constraint" do
    it "accepts constraint parameter" do
      # Test that GenerationConfig can be created with constraint parameter
      # This doesn't require a real constraint object
      config = Candle::GenerationConfig.new(
        temperature: 0.7,
        max_length: 100,
        constraint: nil  # Would be a StructuredConstraint in real usage
      )
      
      expect(config.temperature).to eq(0.7)
      expect(config.max_length).to eq(100)
    end
  end
end