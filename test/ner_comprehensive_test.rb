# frozen_string_literal: true

require_relative "test_helper"

class NERComprehensiveTest < Minitest::Test  
  def test_suggested_models
    models = Candle::NER.suggested_models
    
    assert_instance_of Hash, models
    assert models.key?(:general)
    assert models.key?(:biomedical)
    assert models.key?(:multilingual)
    
    # Check structure
    assert_equal "Babelscape/wikineural-multilingual-ner", models[:general][:model]
    assert models[:general][:note].include?("tokenizer")
  end
  
  # Load a real NER model for testing
  @@ner = nil
  @@model_loaded = false
  
  def self.load_model_once
    unless @@model_loaded
      begin
        # Use the Babelscape model which has tokenizer.json included
        @@ner = Candle::NER.from_pretrained("Babelscape/wikineural-multilingual-ner")
        @@model_loaded = true
      rescue => e
        @@model_loaded = :failed
        @@load_error = e
      end
    end
  end
  

  def setup
    super
    self.class.load_model_once
    if @@model_loaded == :failed
      skip "NER model loading failed: #{@@load_error.message}"
    end
    @sample_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
  end
  
  def test_entity_types
    skip "NER model not loaded" unless @@ner
    entity_types = @@ner.entity_types
    
    assert_instance_of Array, entity_types
    # WikiNeural model has PER, ORG, LOC, MISC
    assert entity_types.include?("PER")
    assert entity_types.include?("ORG")
    assert entity_types.include?("LOC")
    assert entity_types.include?("MISC")
    refute entity_types.include?("O")
    refute entity_types.any? { |t| t.start_with?("B-") || t.start_with?("I-") }
  end
  
  def test_supports_entity
    skip "NER model not loaded" unless @@ner
    assert @@ner.supports_entity?("PER")
    assert @@ner.supports_entity?("per") # Should handle lowercase
    assert @@ner.supports_entity?("ORG")
    assert @@ner.supports_entity?("LOC")
    assert @@ner.supports_entity?("MISC")
    refute @@ner.supports_entity?("GENE")
    refute @@ner.supports_entity?("INVALID")
  end
  
  def test_extract_entity_type
    skip "NER model not loaded" unless @@ner
    # Test extracting only ORG entities
    orgs = @@ner.extract_entity_type(@sample_text, "ORG", confidence_threshold: 0.5)
    assert_instance_of Array, orgs
    assert orgs.all? { |e| e["label"] == "ORG" }
    
    # Test extracting only PER entities
    people = @@ner.extract_entity_type(@sample_text, "PER", confidence_threshold: 0.5)
    assert_instance_of Array, people
    assert people.all? { |e| e["label"] == "PER" }
    
    # Test with lowercase entity type (should be uppercased)
    locs = @@ner.extract_entity_type(@sample_text, "loc", confidence_threshold: 0.5)
    assert_instance_of Array, locs
    assert locs.all? { |e| e["label"] == "LOC" }
  end
  
  def test_analyze_method
    skip "NER model not loaded" unless @@ner
    result = @@ner.analyze(@sample_text, confidence_threshold: 0.5)
    
    assert_instance_of Hash, result
    assert result.key?(:entities)
    assert result.key?(:tokens)
    assert_instance_of Array, result[:entities]
    assert_instance_of Array, result[:tokens]
    
    # Both should have processed the text
    assert result[:tokens].length > 0
  end
  
  def test_format_entities
    skip "NER model not loaded" unless @@ner
    formatted = @@ner.format_entities(@sample_text, confidence_threshold: 0.5)
    
    assert_instance_of String, formatted
    # Should contain entity labels in brackets if entities found
    entities = @@ner.extract_entities(@sample_text, confidence_threshold: 0.5)
    if entities.any?
      assert formatted =~ /\[[A-Z]+:\d\.\d+\]/
      # Should contain original text
      assert formatted.include?("Apple")
      assert formatted.include?("Steve Jobs")
    end
  end
  
  def test_format_entities_empty
    skip "NER model not loaded" unless @@ner
    # Test with text that likely has no entities
    formatted = @@ner.format_entities("The the the the the", confidence_threshold: 0.5)
    assert_equal "The the the the the", formatted
  end
  
  def test_inspect_and_to_s
    skip "NER model not loaded" unless @@ner
    inspect_str = @@ner.inspect
    to_s_str = @@ner.to_s
    
    assert_instance_of String, inspect_str
    assert inspect_str.start_with?("#<Candle::NER")
    assert_equal inspect_str, to_s_str
  end
  
  def test_gazetteer_add_terms
    recognizer = Candle::GazetteerEntityRecognizer.new("DRUG")
    
    # Test adding single term
    recognizer.add_terms("aspirin")
    assert recognizer.terms.include?("aspirin")
    
    # Test adding array of terms
    recognizer.add_terms(["ibuprofen", "acetaminophen"])
    assert recognizer.terms.include?("ibuprofen")
    assert recognizer.terms.include?("acetaminophen")
    
    # Test method chaining
    result = recognizer.add_terms("naproxen")
    assert_equal recognizer, result
  end
  
  def test_gazetteer_load_from_file
    # Create a temporary file with terms
    require 'tempfile'
    
    file = Tempfile.new(['drug_list', '.txt'])
    file.write("aspirin\n")
    file.write("ibuprofen\n")
    file.write("# This is a comment\n")
    file.write("acetaminophen\n")
    file.write("\n") # Empty line
    file.write("naproxen\n")
    file.close
    
    recognizer = Candle::GazetteerEntityRecognizer.new("DRUG")
    result = recognizer.load_from_file(file.path)
    
    # Check method chaining
    assert_equal recognizer, result
    
    # Check loaded terms
    assert recognizer.terms.include?("aspirin")
    assert recognizer.terms.include?("ibuprofen")
    assert recognizer.terms.include?("acetaminophen")
    assert recognizer.terms.include?("naproxen")
    
    # Comments and empty lines should be ignored
    refute recognizer.terms.include?("# This is a comment")
    refute recognizer.terms.include?("")
    
    # Cleanup
    file.unlink
  end
  
  def test_hybrid_ner_with_all_components
    # Create hybrid with pattern and gazetteer recognizers
    hybrid = Candle::HybridNER.new
    
    # Add email pattern
    hybrid.add_pattern_recognizer("EMAIL", [/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/])
    
    # Add company gazetteer
    hybrid.add_gazetteer_recognizer("COMPANY", ["Apple", "Google", "Microsoft"])
    
    text = "Contact john@apple.com or mary@google.com about Microsoft products."
    entities = hybrid.extract_entities(text)
    
    # Should find emails
    emails = entities.select { |e| e["label"] == "EMAIL" }
    assert_equal 2, emails.length
    assert emails.any? { |e| e["text"] == "john@apple.com" }
    assert emails.any? { |e| e["text"] == "mary@google.com" }
    
    # Should find company
    companies = entities.select { |e| e["label"] == "COMPANY" }
    assert companies.any? { |e| e["text"] == "Microsoft" }
  end
  
  def test_pattern_recognizer_string_patterns
    recognizer = Candle::PatternEntityRecognizer.new("ID")
    
    # Test adding string pattern (should be converted to regex)
    recognizer.add_pattern("ID-\\d{4}")
    
    text = "User ID-1234 and ID-5678 are active"
    entities = recognizer.recognize(text)
    
    assert_equal 2, entities.length
    assert entities.all? { |e| e["label"] == "ID" }
    assert entities.any? { |e| e["text"] == "ID-1234" }
    assert entities.any? { |e| e["text"] == "ID-5678" }
  end
  
  def test_merge_entities_complex_overlaps
    hybrid = Candle::HybridNER.new
    
    # Test the private merge_entities method indirectly through overlapping patterns
    hybrid.add_pattern_recognizer("LONG", [/Steve Jobs/])
    hybrid.add_pattern_recognizer("SHORT", [/Steve/])
    hybrid.add_pattern_recognizer("ANOTHER", [/Jobs/])
    
    text = "Steve Jobs founded Apple"
    entities = hybrid.extract_entities(text)
    
    # Should only have one entity due to overlap resolution
    # First match (LONG) should win because it appears first in processing
    assert_equal 1, entities.length
    assert_equal "Steve Jobs", entities.first["text"]
    assert_equal "LONG", entities.first["label"]
  end
end