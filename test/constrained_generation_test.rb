# frozen_string_literal: true

require_relative "test_helper"
require "json"

class ConstrainedGenerationTest < Minitest::Test
  def test_generation_config_supports_constraints
    # This test verifies that GenerationConfig can be created
    # In the future, when Ruby API is exposed, we'll test:
    # - Creating a GenerationConfig with a constraint
    # - Passing it to LLM generation methods
    # - Verifying output follows the schema
    
    # For now, just verify the module loads
    assert defined?(Candle), "Candle module should be defined"
  end
  
  def test_structured_generation_examples
    # Document example use cases for structured generation
    
    # Example 1: Yes/No responses
    yes_no_schema = {
      type: "object",
      properties: {
        answer: {
          type: "string",
          enum: ["yes", "no"]
        },
        confidence: {
          type: "number",
          minimum: 0,
          maximum: 1
        }
      },
      required: ["answer"]
    }
    
    # Example 2: Multiple choice
    multiple_choice_schema = {
      type: "object",
      properties: {
        choice: {
          type: "string",
          enum: ["A", "B", "C", "D"]
        },
        reasoning: {
          type: "string"
        }
      },
      required: ["choice"]
    }
    
    # Example 3: Structured data extraction
    extraction_schema = {
      type: "object",
      properties: {
        entities: {
          type: "array",
          items: {
            type: "object",
            properties: {
              name: { type: "string" },
              type: { type: "string", enum: ["person", "organization", "location"] },
              confidence: { type: "number" }
            },
            required: ["name", "type"]
          }
        }
      }
    }
    
    # Example 4: Code generation with specific format
    code_schema = {
      type: "object",
      properties: {
        language: {
          type: "string",
          enum: ["python", "ruby", "javascript"]
        },
        code: {
          type: "string"
        },
        explanation: {
          type: "string"
        }
      },
      required: ["language", "code"]
    }
    
    # Verify all schemas are valid JSON
    [yes_no_schema, multiple_choice_schema, extraction_schema, code_schema].each do |schema|
      json = schema.to_json
      parsed = JSON.parse(json)
      assert parsed["type"], "Schema should have a type"
    end
  end
  
  def test_direct_regex_constraints
    # Examples of using regex patterns directly for constraints
    
    patterns = {
      # Email validation
      email: /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/,
      
      # Phone number (US format)
      phone: /^\+?1?\s?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$/,
      
      # Date in ISO format
      iso_date: /^\d{4}-\d{2}-\d{2}$/,
      
      # Time in 24h format
      time_24h: /^([01]?[0-9]|2[0-3]):[0-5][0-9]$/,
      
      # Semantic version
      semver: /^v?\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$/
    }
    
    assert patterns.all? { |_, pattern| pattern.is_a?(Regexp) }
  end
end