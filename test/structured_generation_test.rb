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
    skip "Skipping integration test - set TEST_STRUCTURED=1 to run" unless ENV["TEST_STRUCTURED"]
    
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
      max_length: 20,
      constraint: constraint
    )
    
    result = llm.generate("Is the sky blue?", config: config)
    
    # Result should be parseable JSON matching the schema
    begin
      parsed = JSON.parse(result)
      assert parsed.is_a?(Hash), "Result should be a JSON object"
      assert %w[yes no].include?(parsed["answer"]), "Answer should be yes or no"
    rescue JSON::ParserError
      flunk "Generated output should be valid JSON: #{result}"
    end
  end
  
  def test_constraint_from_regex
    skip "Skipping integration test - set TEST_STRUCTURED=1 to run" unless ENV["TEST_STRUCTURED"]
    
    llm = Candle::LLM.from_pretrained(@model_id, device: @device)
    
    # Create a constraint for phone numbers
    phone_regex = '\+?1?\s?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    constraint = llm.constraint_from_regex(phone_regex)
    assert_instance_of Candle::StructuredConstraint, constraint
    
    # Generate with the constraint
    config = Candle::GenerationConfig.deterministic(
      max_length: 20,
      constraint: constraint
    )
    
    result = llm.generate("Generate a US phone number:", config: config)
    
    # Result should match phone pattern
    assert_match(/\+?1?\s?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/, result, 
                 "Generated output should match phone number pattern")
  end
  
  def test_multiple_choice_constraint
    skip "Skipping integration test - set TEST_STRUCTURED=1 to run" unless ENV["TEST_STRUCTURED"]
    
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
      max_length: 30,
      constraint: constraint
    )
    
    prompt = "What is 2+2? A) 3 B) 4 C) 5 D) 6"
    result = llm.generate(prompt, config: config)
    
    parsed = JSON.parse(result)
    assert %w[A B C D].include?(parsed["choice"]), "Choice should be A, B, C, or D"
    if parsed["confidence"]
      assert parsed["confidence"] >= 0 && parsed["confidence"] <= 1, 
             "Confidence should be between 0 and 1"
    end
  end
  
  def test_structured_data_extraction
    skip "Skipping integration test - set TEST_STRUCTURED=1 to run" unless ENV["TEST_STRUCTURED"]
    
    llm = Candle::LLM.from_pretrained(@model_id, device: @device)
    
    # Schema for extracting entities
    schema = {
      type: "object",
      properties: {
        entities: {
          type: "array",
          items: {
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
        }
      },
      required: ["entities"]
    }
    
    constraint = llm.constraint_from_schema(schema)
    config = Candle::GenerationConfig.balanced(
      max_length: 100,
      constraint: constraint
    )
    
    prompt = "Extract entities from: John Smith works at Apple Inc. in Cupertino."
    result = llm.generate(prompt, config: config)
    
    parsed = JSON.parse(result)
    assert parsed["entities"].is_a?(Array), "Entities should be an array"
    
    parsed["entities"].each do |entity|
      assert entity["name"].is_a?(String), "Entity name should be a string"
      assert %w[person organization location].include?(entity["type"]), 
             "Entity type should be person, organization, or location"
    end
  end
  
  def test_constraint_with_generation_config_with_method
    skip "Skipping integration test - set TEST_STRUCTURED=1 to run" unless ENV["TEST_STRUCTURED"]
    
    llm = Candle::LLM.from_pretrained(@model_id, device: @device)
    
    schema = { type: "boolean" }
    constraint = llm.constraint_from_schema(schema)
    
    # Test that constraint is preserved with .with() method
    base_config = Candle::GenerationConfig.balanced(constraint: constraint)
    new_config = base_config.with(temperature: 0.5)
    
    # Generate with new config
    result = llm.generate("Is 2 > 1?", config: new_config)
    
    # Should generate just "true" or "false"
    assert %w[true false].include?(result.strip), "Output should be boolean"
  end
  
  def test_constraint_classes_available
    # Basic availability test that doesn't require model loading
    assert defined?(Candle::StructuredConstraint), "StructuredConstraint should be defined"
    
    # Verify LLM methods are defined
    assert Candle::LLM.instance_methods.include?(:constraint_from_schema), 
           "LLM should have constraint_from_schema method"
    assert Candle::LLM.instance_methods.include?(:constraint_from_regex), 
           "LLM should have constraint_from_regex method"
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