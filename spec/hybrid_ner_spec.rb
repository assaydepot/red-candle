require "spec_helper"

RSpec.describe Candle::HybridNER do
  describe "#initialize" do
    it "creates hybrid NER without model" do
      hybrid = described_class.new
      expect(hybrid.model_ner).to be_nil
      expect(hybrid.pattern_recognizers).to be_empty
      expect(hybrid.gazetteer_recognizers).to be_empty
    end
    
    it "creates hybrid NER with model" do
      # Executing without skip to test model loading
      hybrid = described_class.new("Babelscape/wikineural-multilingual-ner")
      expect(hybrid.model_ner).not_to be_nil
    end
  end
  
  describe "#add_pattern_recognizer" do
    let(:hybrid) { described_class.new }
    
    it "adds pattern recognizer" do
      hybrid.add_pattern_recognizer("GENE", [/TP53/, /BRCA\d/])
      expect(hybrid.pattern_recognizers.length).to eq(1)
      expect(hybrid.pattern_recognizers.first).to be_a(Candle::PatternEntityRecognizer)
      expect(hybrid.pattern_recognizers.first.entity_type).to eq("GENE")
    end
    
    it "returns self for chaining" do
      result = hybrid
        .add_pattern_recognizer("GENE", [/TP53/])
        .add_pattern_recognizer("EMAIL", [/@/])
      
      expect(result).to eq(hybrid)
      expect(hybrid.pattern_recognizers.length).to eq(2)
    end
  end
  
  describe "#add_gazetteer_recognizer" do
    let(:hybrid) { described_class.new }
    
    it "adds gazetteer recognizer" do
      hybrid.add_gazetteer_recognizer("DRUG", ["aspirin", "ibuprofen"])
      expect(hybrid.gazetteer_recognizers.length).to eq(1)
      expect(hybrid.gazetteer_recognizers.first).to be_a(Candle::GazetteerEntityRecognizer)
      expect(hybrid.gazetteer_recognizers.first.entity_type).to eq("DRUG")
    end
    
    it "passes options to gazetteer" do
      hybrid.add_gazetteer_recognizer("TERM", ["Test"], case_sensitive: true)
      recognizer = hybrid.gazetteer_recognizers.first
      expect(recognizer.case_sensitive).to be true
    end
    
    it "returns self for chaining" do
      result = hybrid
        .add_gazetteer_recognizer("DRUG", ["aspirin"])
        .add_gazetteer_recognizer("COMPANY", ["Apple"])
      
      expect(result).to eq(hybrid)
      expect(hybrid.gazetteer_recognizers.length).to eq(2)
    end
  end
  
  describe "#extract_entities" do
    context "without model" do
      let(:hybrid) { described_class.new }
      
      before do
        hybrid.add_pattern_recognizer("GENE", [/\b[A-Z]{2,6}\d*\b/])
        hybrid.add_gazetteer_recognizer("DRUG", ["aspirin", "ibuprofen"])
      end
      
      it "combines pattern and gazetteer results" do
        text = "TP53 mutations respond to aspirin treatment"
        entities = hybrid.extract_entities(text)
        
        expect(entities.length).to eq(2)
        expect(entities.map { |e| e[:label] }).to contain_exactly("GENE", "DRUG")
        expect(entities.map { |e| e[:text] }).to contain_exactly("TP53", "aspirin")
      end
      
      it "returns entities sorted by position" do
        text = "aspirin and TP53"
        entities = hybrid.extract_entities(text)
        
        expect(entities.map { |e| e[:text] }).to eq(["aspirin", "TP53"])
        expect(entities.first[:start]).to be < entities.last[:start]
      end
    end
    
    context "overlapping entities" do
      let(:hybrid) { described_class.new }
      
      it "prefers higher confidence when entities overlap" do
        # Add recognizers that will create overlapping entities
        hybrid.add_pattern_recognizer("LONG", [/Steve Jobs/])
        hybrid.add_pattern_recognizer("SHORT", [/Steve/])
        
        entities = hybrid.extract_entities("Steve Jobs founded Apple")
        
        # Should only keep "Steve Jobs" (first added, same confidence)
        expect(entities.length).to eq(1)
        expect(entities.first[:text]).to eq("Steve Jobs")
        expect(entities.first[:label]).to eq("LONG")
      end
      
      it "handles partial overlaps" do
        hybrid.add_pattern_recognizer("A", [/test word/])
        hybrid.add_pattern_recognizer("B", [/word example/])
        
        entities = hybrid.extract_entities("test word example")
        
        # These partially overlap - "word" is in both
        # The merge_entities method removes overlapping entities, keeping higher confidence/first
        expect(entities.length).to eq(1)
        expect(entities.first[:text]).to eq("test word")
        expect(entities.first[:label]).to eq("A")
      end
      
      it "handles identical spans from different sources" do
        hybrid.add_pattern_recognizer("TYPE1", [/test/])
        hybrid.add_gazetteer_recognizer("TYPE2", ["test"])
        
        entities = hybrid.extract_entities("test")
        
        # Same span, should keep only one (first by position, then by confidence)
        expect(entities.length).to eq(1)
      end
    end
    
    context "multiple recognizers of same type" do
      let(:hybrid) { described_class.new }
      
      before do
        hybrid.add_pattern_recognizer("GENE", [/TP\d+/])
        hybrid.add_pattern_recognizer("PROTEIN", [/CD\d+/])
        hybrid.add_gazetteer_recognizer("DRUG", ["aspirin"])
        hybrid.add_gazetteer_recognizer("DISEASE", ["cancer"])
      end
      
      it "applies all recognizers" do
        text = "TP53 and CD4 in cancer treated with aspirin"
        entities = hybrid.extract_entities(text)
        
        expect(entities.length).to eq(4)
        labels = entities.map { |e| e[:label] }
        expect(labels).to contain_exactly("GENE", "PROTEIN", "DISEASE", "DRUG")
      end
    end
    
    context "confidence threshold" do
      let(:hybrid) { described_class.new }
      
      before do
        # Pattern and gazetteer matches always have confidence 1.0
        hybrid.add_pattern_recognizer("HIGH", [/test/])
      end
      
      it "respects confidence threshold for pattern matches" do
        # Pattern matches have confidence 1.0, so threshold 0.9 should include them
        entities = hybrid.extract_entities("test", confidence_threshold: 0.9)
        expect(entities.length).to eq(1)
        
        # Even with threshold 1.0, pattern matches should be included
        entities = hybrid.extract_entities("test", confidence_threshold: 1.0)
        expect(entities.length).to eq(1)
      end
    end
    
    context "empty text" do
      let(:hybrid) { described_class.new }
      
      before do
        hybrid.add_pattern_recognizer("TEST", [/test/])
      end
      
      it "returns empty array for empty text" do
        expect(hybrid.extract_entities("")).to be_empty
      end
    end
    
    context "complex example" do
      let(:hybrid) { described_class.new }
      
      before do
        # Email patterns
        hybrid.add_pattern_recognizer("EMAIL", [
          /\b[\w._%+-]+@[\w.-]+\.[A-Za-z]{2,}\b/
        ])
        
        # Phone patterns
        hybrid.add_pattern_recognizer("PHONE", [
          /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/,
          /\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b/
        ])
        
        # Company names from gazetteer
        hybrid.add_gazetteer_recognizer("COMPANY", [
          "Apple", "Google", "Microsoft"
        ])
        
        # Person names from gazetteer
        hybrid.add_gazetteer_recognizer("PERSON", [
          "Tim Cook", "Steve Jobs"
        ])
      end
      
      it "extracts multiple entity types from complex text" do
        text = "Contact Tim Cook at Apple via tcook@apple.com or 555-123-4567"
        entities = hybrid.extract_entities(text)
        
        expect(entities.map { |e| e[:label] }).to contain_exactly(
          "PERSON", "COMPANY", "EMAIL", "PHONE"
        )
        expect(entities.map { |e| e[:text] }).to contain_exactly(
          "Tim Cook", "Apple", "tcook@apple.com", "555-123-4567"
        )
      end
    end
  end
end