# frozen_string_literal: true

require "spec_helper"

RSpec.describe "NERIntegration" do
  # Try to load a small NER model for testing
  let(:ner) do
    @ner ||= begin
      Candle::NER.from_pretrained("dslim/bert-base-NER", tokenizer: "bert-base-cased")
    rescue => e
      puts "NER model not available: #{e.message}"
      nil
    end
  end
  
  let(:model_available) { !ner.nil? }
  
  after(:all) do
    @ner = nil
    GC.start
  end
  
  describe "#entity_types" do
    it "returns available entity types" do
      skip "NER model not available" unless model_available
      
      entity_types = ner.entity_types
      
      expect(entity_types).to be_an(Array)
      # Standard NER models usually have PER, ORG, LOC
      expect(entity_types.any? { |t| ["PER", "ORG", "LOC", "MISC"].include?(t) }).to be true
    end
  end
  
  describe "#supports_entity?" do
    it "checks entity type support" do
      skip "NER model not available" unless model_available
      
      # Most NER models support these basic types
      if ner.entity_types.include?("PER")
        expect(ner.supports_entity?("PER")).to be true
        expect(ner.supports_entity?("per")).to be true # Case insensitive
      end
      
      # Should not support made-up types
      expect(ner.supports_entity?("FAKETYPE")).to be false
    end
  end
  
  describe "#extract_entity_type" do
    it "extracts specific entity type" do
      skip "NER model not available" unless model_available
      
      text = "Apple Inc. was founded by Steve Jobs in Cupertino."
      
      if ner.supports_entity?("ORG")
        orgs = ner.extract_entity_type(text, "ORG")
        expect(orgs).to be_an(Array)
        # Should find Apple
        expect(orgs.any? { |e| e[:text].downcase.include?("apple") }).to be true
      end
    end
  end
  
  describe "#analyze" do
    it "returns entities and tokens" do
      skip "NER model not available" unless model_available
      
      text = "Microsoft is a technology company."
      result = ner.analyze(text)
      
      expect(result).to be_a(Hash)
      expect(result).to have_key(:entities)
      expect(result).to have_key(:tokens)
      expect(result[:entities]).to be_an(Array)
      expect(result[:tokens]).to be_an(Array)
    end
  end
  
  describe "#format_entities" do
    it "formats text with entity labels" do
      skip "NER model not available" unless model_available
      
      text = "Google was founded by Larry Page."
      formatted = ner.format_entities(text)
      
      expect(formatted).to be_a(String)
      # Should contain entity labels in brackets
      expect(formatted).to match(/\[[A-Z]+:\d\.\d+\]/) if ner.extract_entities(text).any?
    end
  end
  
  describe "#inspect" do
    it "provides string representation" do
      skip "NER model not available" unless model_available
      
      inspect_str = ner.inspect
      expect(inspect_str).to be_a(String)
      expect(inspect_str).to start_with("#<Candle::NER")
      expect(ner.to_s).to eq(inspect_str)
    end
  end
end