# frozen_string_literal: true

require_relative "test_helper"

class NERIntegrationTest < Minitest::Test
  def setup
    @model_available = false    
    begin
      # Try to load a small NER model for testing
      @ner = Candle::NER.from_pretrained("dslim/bert-base-NER", tokenizer: "bert-base-cased")
      @model_available = true
    rescue => e
      puts "NER model not available: #{e.message}"
    end
  end
  
  def test_entity_types
    skip "NER model not available" unless @model_available
    
    entity_types = @ner.entity_types
    
    assert_instance_of Array, entity_types
    # Standard NER models usually have PER, ORG, LOC
    assert entity_types.any? { |t| ["PER", "ORG", "LOC", "MISC"].include?(t) }
  end
  
  def test_supports_entity
    skip "NER model not available" unless @model_available
    
    # Most NER models support these basic types
    if @ner.entity_types.include?("PER")
      assert @ner.supports_entity?("PER")
      assert @ner.supports_entity?("per") # Case insensitive
    end
    
    # Should not support made-up types
    refute @ner.supports_entity?("FAKETYPE")
  end
  
  def test_extract_entity_type
    skip "NER model not available" unless @model_available
    
    text = "Apple Inc. was founded by Steve Jobs in Cupertino."
    
    if @ner.supports_entity?("ORG")
      orgs = @ner.extract_entity_type(text, "ORG")
      assert_instance_of Array, orgs
      # Should find Apple
      assert orgs.any? { |e| e[:text].downcase.include?("apple") }
    end
  end
  
  def test_analyze
    skip "NER model not available" unless @model_available
    
    text = "Microsoft is a technology company."
    result = @ner.analyze(text)
    
    assert_instance_of Hash, result
    assert result.key?(:entities)
    assert result.key?(:tokens)
    assert_instance_of Array, result[:entities]
    assert_instance_of Array, result[:tokens]
  end
  
  def test_format_entities
    skip "NER model not available" unless @model_available
    
    text = "Google was founded by Larry Page."
    formatted = @ner.format_entities(text)
    
    assert_instance_of String, formatted
    # Should contain entity labels in brackets
    assert formatted =~ /\[[A-Z]+:\d\.\d+\]/ if @ner.extract_entities(text).any?
  end
  
  def test_inspect
    skip "NER model not available" unless @model_available
    
    inspect_str = @ner.inspect
    assert_instance_of String, inspect_str
    assert inspect_str.start_with?("#<Candle::NER")
    assert_equal @ner.to_s, inspect_str
  end
end