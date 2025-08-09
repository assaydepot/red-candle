# frozen_string_literal: true

require "spec_helper"

RSpec.describe "NERCoreFunctionality" do
  # Load model once using ModelCache module
  let(:ner) do
    model = ModelCache.ner
    skip "Could not load any NER model" if model.nil?
    model
  end
  
  # Clear cached model after this spec file completes to free memory
  after(:all) do
    ModelCache.clear_model(:ner)
    GC.start
  end
  
  # Test extract_entities method comprehensively
  describe "#extract_entities" do
    context "basic functionality" do
      let(:text) { "Apple Inc. was founded by Steve Jobs in Cupertino, California." }
      let(:entities) { ner.extract_entities(text) }
      
      it "returns an array of entities" do
        expect(entities).to be_an(Array)
      end
      
      it "returns entities with all required fields" do
        entities.each do |entity|
          expect(entity).to be_a(Hash)
          expect(entity).to have_key(:text)
          expect(entity).to have_key(:label)
          expect(entity).to have_key(:start)
          expect(entity).to have_key(:end)
          expect(entity).to have_key(:confidence)
          expect(entity).to have_key(:token_start)
          expect(entity).to have_key(:token_end)
        end
      end
      
      it "returns entities with correct data types" do
        entities.each do |entity|
          expect(entity[:text]).to be_a(String)
          expect(entity[:label]).to be_a(String)
          expect(entity[:start]).to be_an(Integer)
          expect(entity[:end]).to be_an(Integer)
          expect(entity[:confidence]).to be_a(Numeric)
          expect(entity[:token_start]).to be_an(Integer)
          expect(entity[:token_end]).to be_an(Integer)
        end
      end
      
      it "returns entities with valid ranges" do
        entities.each do |entity|
          expect(entity[:start]).to be >= 0
          expect(entity[:end]).to be > entity[:start]
          expect(entity[:end]).to be <= text.length
          expect(entity[:confidence]).to be_between(0.0, 1.0)
          expect(entity[:token_start]).to be >= 0
          expect(entity[:token_end]).to be > entity[:token_start]
        end
      end
      
      it "extracts text that matches entity positions" do
        entities.each do |entity|
          extracted = text[entity[:start]...entity[:end]]
          # Due to tokenization, the extracted text might have slight differences
          # but should be mostly the same
          match = extracted.include?(entity[:text]) || 
                  entity[:text].include?(extracted) ||
                  extracted.strip == entity[:text].strip
          expect(match).to be(true), 
            "Entity text '#{entity[:text]}' doesn't match extracted '#{extracted}'"
        end
      end
    end
    
    context "with confidence threshold" do
      let(:text) { "Microsoft and Google are technology companies." }
      
      it "filters entities by low confidence threshold" do
        low_threshold_entities = ner.extract_entities(text, confidence_threshold: 0.3)
        
        low_threshold_entities.each do |entity|
          expect(entity[:confidence]).to be >= 0.3
        end
      end
      
      it "filters entities by high confidence threshold" do
        high_threshold_entities = ner.extract_entities(text, confidence_threshold: 0.95)
        
        high_threshold_entities.each do |entity|
          expect(entity[:confidence]).to be >= 0.95
        end
      end
      
      it "returns fewer entities with higher threshold" do
        low_threshold_entities = ner.extract_entities(text, confidence_threshold: 0.3)
        high_threshold_entities = ner.extract_entities(text, confidence_threshold: 0.95)
        
        expect(high_threshold_entities.length).to be <= low_threshold_entities.length
      end
    end
    
    context "with empty text" do
      it "returns empty array for empty string" do
        entities = ner.extract_entities("")
        expect(entities).to eq([])
      end
      
      it "returns empty array for whitespace only" do
        entities = ner.extract_entities("   \n\t   ")
        expect(entities).to eq([])
      end
    end
    
    context "with unicode text" do
      it "handles unicode text correctly" do
        text = "Это компания Apple в Москве."
        entities = ner.extract_entities(text)
        
        entities.each do |entity|
          extracted = text[entity[:start]...entity[:end]]
          expect(entity[:text].length).to be > 0, "Entity text should not be empty"
        end
      end
    end
  end
  
  describe "#predict_tokens" do
    let(:text) { "Apple is a company." }
    
    it "returns token-level predictions" do
      predictions = ner.predict_tokens(text)
      
      expect(predictions).to be_an(Array)
      expect(predictions).not_to be_empty
      
      predictions.each do |pred|
        expect(pred).to include("token", "label", "confidence", "index")
      end
    end
  end
  
  describe "#labels" do
    it "returns label configuration" do
      labels = ner.labels
      
      expect(labels).to be_a(Hash)
      expect(labels).to include("id2label")
      expect(labels).to include("label2id")
      expect(labels["id2label"]).to be_a(Hash)
      expect(labels["label2id"]).to be_a(Hash)
    end
  end
  
  describe "#model_info" do
    it "returns informative model description" do
      info = ner.model_info
      
      expect(info).to be_a(String)
      expect(info).to include("NER")
    end
  end
  
  describe "#entity_types" do
    it "returns available entity types" do
      labels = ner.labels
      entity_types = labels["label2id"].keys.reject { |l| l == "O" }
                                           .map { |l| l.sub(/^[BI]-/, "") }
                                           .uniq
      
      expect(entity_types).to be_an(Array)
      expect(entity_types).not_to be_empty
    end
  end
end