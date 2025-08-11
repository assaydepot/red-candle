# frozen_string_literal: true

require "spec_helper"
require 'tempfile'

RSpec.describe "NERComprehensive" do
  describe ".suggested_models" do
    it "returns suggested model information" do
      models = Candle::NER.suggested_models
      
      expect(models).to be_a(Hash)
      expect(models).to have_key(:general)
      expect(models).to have_key(:biomedical)
      expect(models).to have_key(:multilingual)
      
      # Check structure
      expect(models[:general][:model]).to eq("Babelscape/wikineural-multilingual-ner")
      expect(models[:general][:note]).to include("tokenizer")
    end
  end
  
  # Load a real NER model for testing
  let(:ner) do
    ModelCache.ner
  rescue => e
    skip "NER model loading failed: #{e.message}"
  end
  
  let(:sample_text) { "Apple Inc. was founded by Steve Jobs in Cupertino, California." }
  
  # Clear cached model after spec completes
  after(:all) do
    ModelCache.clear_model(:ner)
    GC.start
  end
  
  describe "#entity_types" do
    it "returns available entity types" do
      skip "NER model not loaded" unless ner
      
      entity_types = ner.entity_types
      
      expect(entity_types).to be_an(Array)
      # WikiNeural model has PER, ORG, LOC, MISC
      expect(entity_types).to include("PER")
      expect(entity_types).to include("ORG")
      expect(entity_types).to include("LOC")
      expect(entity_types).to include("MISC")
      expect(entity_types).not_to include("O")
      expect(entity_types.none? { |t| t.start_with?("B-") || t.start_with?("I-") }).to be true
    end
  end
  
  describe "#supports_entity?" do
    it "checks if entity type is supported" do
      skip "NER model not loaded" unless ner
      
      expect(ner.supports_entity?("PER")).to be true
      expect(ner.supports_entity?("per")).to be true # Should handle lowercase
      expect(ner.supports_entity?("ORG")).to be true
      expect(ner.supports_entity?("LOC")).to be true
      expect(ner.supports_entity?("MISC")).to be true
      expect(ner.supports_entity?("GENE")).to be false
      expect(ner.supports_entity?("INVALID")).to be false
    end
  end
  
  describe "#extract_entity_type" do
    it "extracts only specified entity type" do
      skip "NER model not loaded" unless ner
      
      # Test extracting only ORG entities
      orgs = ner.extract_entity_type(sample_text, "ORG", confidence_threshold: 0.5)
      expect(orgs).to be_an(Array)
      expect(orgs.all? { |e| e[:label] == "ORG" }).to be true
      
      # Test extracting only PER entities
      people = ner.extract_entity_type(sample_text, "PER", confidence_threshold: 0.5)
      expect(people).to be_an(Array)
      expect(people.all? { |e| e[:label] == "PER" }).to be true
      
      # Test with lowercase entity type (should be uppercased)
      locs = ner.extract_entity_type(sample_text, "loc", confidence_threshold: 0.5)
      expect(locs).to be_an(Array)
      expect(locs.all? { |e| e[:label] == "LOC" }).to be true
    end
  end
  
  describe "#analyze" do
    it "returns both entities and tokens" do
      skip "NER model not loaded" unless ner
      
      result = ner.analyze(sample_text, confidence_threshold: 0.5)
      
      expect(result).to be_a(Hash)
      expect(result).to have_key(:entities)
      expect(result).to have_key(:tokens)
      expect(result[:entities]).to be_an(Array)
      expect(result[:tokens]).to be_an(Array)
      
      # Both should have processed the text
      expect(result[:tokens].length).to be > 0
    end
  end
  
  describe "#format_entities" do
    it "formats entities with labels and confidence" do
      skip "NER model not loaded" unless ner
      
      formatted = ner.format_entities(sample_text, confidence_threshold: 0.5)
      
      expect(formatted).to be_a(String)
      # Should contain entity labels in brackets if entities found
      entities = ner.extract_entities(sample_text, confidence_threshold: 0.5)
      if entities.any?
        expect(formatted).to match(/\[[A-Z]+:\d\.\d+\]/)
        # Should contain original text
        expect(formatted).to include("Apple")
        expect(formatted).to include("Steve Jobs")
      end
    end
    
    it "returns original text when no entities found" do
      skip "NER model not loaded" unless ner
      
      # Test with text that likely has no entities
      formatted = ner.format_entities("The the the the the", confidence_threshold: 0.5)
      expect(formatted).to eq("The the the the the")
    end
  end
  
  describe "#inspect and #to_s" do
    it "provides meaningful string representation" do
      skip "NER model not loaded" unless ner
      
      inspect_str = ner.inspect
      to_s_str = ner.to_s
      
      expect(inspect_str).to be_a(String)
      expect(inspect_str).to start_with("#<Candle::NER")
      expect(to_s_str).to eq(inspect_str)
    end
  end
  
  describe "GazetteerEntityRecognizer" do
    describe "#add_terms" do
      it "adds terms to the recognizer" do
        recognizer = Candle::GazetteerEntityRecognizer.new("DRUG")
        
        # Test adding single term
        recognizer.add_terms("aspirin")
        expect(recognizer.terms).to include("aspirin")
        
        # Test adding array of terms
        recognizer.add_terms(["ibuprofen", "acetaminophen"])
        expect(recognizer.terms).to include("ibuprofen")
        expect(recognizer.terms).to include("acetaminophen")
        
        # Test method chaining
        result = recognizer.add_terms("naproxen")
        expect(result).to eq(recognizer)
      end
    end
    
    describe "#load_from_file" do
      it "loads terms from file" do
        # Create a temporary file with terms
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
        expect(result).to eq(recognizer)
        
        # Check loaded terms
        expect(recognizer.terms).to include("aspirin")
        expect(recognizer.terms).to include("ibuprofen")
        expect(recognizer.terms).to include("acetaminophen")
        expect(recognizer.terms).to include("naproxen")
        
        # Comments and empty lines should be ignored
        expect(recognizer.terms).not_to include("# This is a comment")
        expect(recognizer.terms).not_to include("")
        
        # Cleanup
        file.unlink
      end
    end
  end
  
  describe "HybridNER" do
    it "combines pattern and gazetteer recognizers" do
      # Create hybrid with pattern and gazetteer recognizers
      hybrid = Candle::HybridNER.new
      
      # Add email pattern
      hybrid.add_pattern_recognizer("EMAIL", [/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/])
      
      # Add company gazetteer
      hybrid.add_gazetteer_recognizer("COMPANY", ["Apple", "Google", "Microsoft"])
      
      text = "Contact john@apple.com or mary@google.com about Microsoft products."
      entities = hybrid.extract_entities(text)
      
      # Should find emails
      emails = entities.select { |e| e[:label] == "EMAIL" }
      expect(emails.length).to eq(2)
      expect(emails.any? { |e| e[:text] == "john@apple.com" }).to be true
      expect(emails.any? { |e| e[:text] == "mary@google.com" }).to be true
      
      # Should find company
      companies = entities.select { |e| e[:label] == "COMPANY" }
      expect(companies.any? { |e| e[:text] == "Microsoft" }).to be true
    end
    
    it "resolves complex overlaps" do
      hybrid = Candle::HybridNER.new
      
      # Test the private merge_entities method indirectly through overlapping patterns
      hybrid.add_pattern_recognizer("LONG", [/Steve Jobs/])
      hybrid.add_pattern_recognizer("SHORT", [/Steve/])
      hybrid.add_pattern_recognizer("ANOTHER", [/Jobs/])
      
      text = "Steve Jobs founded Apple"
      entities = hybrid.extract_entities(text)
      
      # Should only have one entity due to overlap resolution
      # First match (LONG) should win because it appears first in processing
      expect(entities.length).to eq(1)
      expect(entities.first[:text]).to eq("Steve Jobs")
      expect(entities.first[:label]).to eq("LONG")
    end
  end
  
  describe "PatternEntityRecognizer with string patterns" do
    it "accepts string patterns and converts to regex" do
      recognizer = Candle::PatternEntityRecognizer.new("ID")
      
      # Test adding string pattern (should be converted to regex)
      recognizer.add_pattern("ID-\\d{4}")
      
      text = "User ID-1234 and ID-5678 are active"
      entities = recognizer.recognize(text)
      
      expect(entities.length).to eq(2)
      expect(entities.all? { |e| e[:label] == "ID" }).to be true
      expect(entities.any? { |e| e[:text] == "ID-1234" }).to be true
      expect(entities.any? { |e| e[:text] == "ID-5678" }).to be true
    end
  end
end