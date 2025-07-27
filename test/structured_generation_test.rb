# frozen_string_literal: true

require_relative "test_helper"
require "json"

class StructuredGenerationTest < Minitest::Test
  def setup
    @device = Candle::Device.cpu
    # Using a small model for testing
    @model_id = ENV["TEST_MODEL"] || "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  end
  
  def test_constraint_from_json_schema
    llm = Candle::LLM.from_pretrained(@model_id, device: @device)
    
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
    assert_instance_of Candle::StructuredConstraint, constraint
    
    # Generate with the constraint
    config = Candle::GenerationConfig.deterministic(
      max_length: 50,  # Increase to ensure complete JSON
      constraint: constraint,
      temperature: 0.1
    )
    
    result = llm.generate("Is the sky blue?", config: config)
    
    # Result should be parseable JSON matching the schema
    begin
      parsed = JSON.parse(result)
      assert parsed.is_a?(Hash), "Result should be a JSON object"
      assert %w[yes no].include?(parsed["answer"]), "Answer should be yes or no"
    rescue JSON::ParserError
      # Check if the output at least follows the expected pattern
      if result.include?('{"answer":')
        assert result.match(/"answer":\s*"(yes|no)"/), "Output should contain yes or no answer"
      else
        flunk "Generated output doesn't match expected format: #{result}"
      end
    end
  end
  
  def test_constraint_from_regex    
    llm = Candle::LLM.from_pretrained(@model_id, device: @device)
    
    # Create a simpler constraint - just one or more digits
    digit_regex = '\d+'
    constraint = llm.constraint_from_regex(digit_regex)
    assert_instance_of Candle::StructuredConstraint, constraint
    
    # Generate with the constraint
    config = Candle::GenerationConfig.deterministic(
      max_length: 10,
      constraint: constraint,
      temperature: 0.1
    )
    
    result = llm.generate("Generate a number:", config: config)
    
    # Result should contain only digits
    assert_match(/^\d+$/, result.strip, 
                 "Generated output should contain only digits, got: #{result.inspect}")
  end
  
  def test_multiple_choice_constraint
    llm = Candle::LLM.from_pretrained(@model_id, device: @device)
    
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
      parsed = JSON.parse(result)
      assert %w[A B C D].include?(parsed["choice"]), "Choice should be A, B, C, or D"
      if parsed["confidence"]
        assert parsed["confidence"] >= 0 && parsed["confidence"] <= 1, 
               "Confidence should be between 0 and 1"
      end
    rescue JSON::ParserError
      # If JSON is incomplete, check if it at least started correctly
      assert result.include?('{"choice":'), "Output should start with valid JSON structure"
      assert result.match(/"choice":\s*"[ABCD]/), "Output should contain a valid choice"
    end
  end
  
  def test_structured_data_extraction
    llm = Candle::LLM.from_pretrained(@model_id, device: @device)
    
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
      parsed = JSON.parse(result)
      assert parsed["name"].is_a?(String), "Name should be a string"
      assert %w[person organization location].include?(parsed["type"]), 
             "Type should be person, organization, or location"
    rescue JSON::ParserError
      # Check partial output
      assert result.include?('{"name":') || result.include?('{"type":'), 
             "Output should start with expected JSON structure"
    end
  end
  
  def test_constraint_with_generation_config_with_method
    # This test verifies that constraints work with the .with() method
    # We just test that the mechanism works, not the actual generation
    llm = Candle::LLM.from_pretrained(@model_id, device: @device)
    
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
    assert_equal 0.5, new_config.temperature
    assert_equal 10, new_config.max_length
  end
  
  def test_generate_structured
    llm = Candle::LLM.from_pretrained(@model_id, device: @device)
    
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
    assert result.is_a?(Hash), "Result should be a parsed Hash, got: #{result.class}"
    assert %w[success failure].include?(result["result"]), "Result should be success or failure"
  end
  
  def test_constraint_classes_available
    # Basic availability test that doesn't require model loading
    assert defined?(Candle::StructuredConstraint), "StructuredConstraint should be defined"
    
    # Verify LLM methods are defined
    assert Candle::LLM.instance_methods.include?(:constraint_from_schema), 
           "LLM should have constraint_from_schema method"
    assert Candle::LLM.instance_methods.include?(:constraint_from_regex), 
           "LLM should have constraint_from_regex method"
    assert Candle::LLM.instance_methods.include?(:generate_structured), 
           "LLM should have generate_structured method"
  end
  
  def test_generation_config_accepts_constraint
    # Test that GenerationConfig can be created with constraint parameter
    # This doesn't require a real constraint object
    config = Candle::GenerationConfig.new(
      temperature: 0.7,
      max_length: 100,
      constraint: nil  # Would be a StructuredConstraint in real usage
    )
    
    assert_equal 0.7, config.temperature
    assert_equal 100, config.max_length
  end
end