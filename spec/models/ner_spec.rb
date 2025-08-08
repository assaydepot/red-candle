require 'spec_helper'

RSpec.describe Candle::NER do
  include_context "cached models"
  
  describe ".from_pretrained" do
    it "loads a model from HuggingFace" do
      # This uses the cached model - loaded only once for all specs
      expect(ner).to be_a(Candle::NER)
    end
    
    it "defaults to Device.best" do
      # Note: ner is memoized and cached
      # Device.best returns the best available device (Metal > CUDA > CPU)
      available = DeviceHelpers.available_devices
      expect(available.any? { |d| ner.device.to_s.include?(d.to_s) }).to be true
    end
    
    it_behaves_like "a model with inspect" do
      subject { ner }
    end
    it_behaves_like "a model with device support" do
      let(:model_id) { "Babelscape/wikineural-multilingual-ner" }
    end
  end
  
  describe "#extract_entities" do
    # Using let! to force evaluation and cache the model
    let!(:cached_ner) { ner }
    
    context "with typical text" do
      let(:text) { "Apple Inc. was founded by Steve Jobs in Cupertino." }
      let(:entities) { cached_ner.extract_entities(text) }
      
      it "extracts organization entities" do
        org_entities = entities.select { |e| e[:label] == "ORG" }
        expect(org_entities).not_to be_empty
        expect(org_entities.first[:text]).to include("Apple")
      end
      
      it "extracts person entities" do
        per_entities = entities.select { |e| e[:label] == "PER" }
        expect(per_entities).not_to be_empty
        expect(per_entities.first[:text]).to include("Steve Jobs")
      end
      
      it "extracts location entities" do
        loc_entities = entities.select { |e| e[:label] == "LOC" }
        expect(loc_entities).not_to be_empty
        expect(loc_entities.first[:text]).to include("Cupertino")
      end
      
      it "returns entities with symbol keys" do
        expect(entities.first.keys).to all(be_a(Symbol))
        expect(entities.first).to include(:text, :label, :start, :end, :confidence)
      end
    end
    
    context "with confidence threshold" do
      let(:text) { "John works at Google." }
      
      it "filters entities by confidence" do
        high_conf = cached_ner.extract_entities(text, confidence_threshold: 0.95)
        low_conf = cached_ner.extract_entities(text, confidence_threshold: 0.5)
        
        expect(low_conf.length).to be >= high_conf.length
      end
    end
    
    context "with unicode text" do
      let(:text) { "北京で山田太郎さんに会いました。" }
      
      it "handles unicode characters correctly" do
        entities = cached_ner.extract_entities(text)
        entities.each do |entity|
          expect(entity[:start]).to be >= 0
          expect(entity[:end]).to be > entity[:start]
          expect(entity[:text]).not_to be_empty
        end
      end
    end
  end
  
  describe "#predict_tokens" do
    let(:text) { "Apple is a company." }
    let(:predictions) { ner.predict_tokens(text) }
    
    it "returns token-level predictions" do
      expect(predictions).to be_an(Array)
      expect(predictions).not_to be_empty
    end
    
    it "includes token information" do
      predictions.each do |pred|
        expect(pred).to include("token", "label", "confidence", "index")
      end
    end
  end
  
  describe "#labels" do
    it "returns label configuration" do
      labels = ner.labels
      expect(labels).to be_a(Hash)
      expect(labels).to include("id2label", "label2id")
    end
  end
  
  describe "#tokenizer" do
    it "returns the model's tokenizer" do
      tokenizer = ner.tokenizer
      expect(tokenizer).to be_a(Candle::Tokenizer)
    end
  end
  
  # Performance specs
  describe "performance", :performance do
    it "caches the model across examples" do
      # First access loads the model
      start_time = Time.now
      first_ner = model_cache.ner
      first_load_time = Time.now - start_time
      
      # Second access should be instant (cached)
      start_time = Time.now
      second_ner = model_cache.ner
      second_load_time = Time.now - start_time
      
      # Second load should be much faster (at least 10x)
      # But with very fast first loads, use a minimum threshold
      expect(second_load_time).to be < [first_load_time * 0.5, 0.001].max
      expect(first_ner).to equal(second_ner) # Same object
    end
  end
end