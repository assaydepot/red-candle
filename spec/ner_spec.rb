# frozen_string_literal: true

require "spec_helper"

RSpec.describe "NER" do
  describe "PatternEntityRecognizer" do
    it "recognizes entities based on patterns" do
      recognizer = Candle::PatternEntityRecognizer.new("GENE")
      recognizer.add_pattern(/\b[A-Z][A-Z0-9]{2,}\b/)
      recognizer.add_pattern(/\bCD\d+\b/)
      
      text = "TP53 and BRCA1 mutations are common. CD4+ cells express CD8."
      entities = recognizer.recognize(text)
      
      expect(entities.any? { |e| e[:text] == "TP53" }).to be true
      expect(entities.any? { |e| e[:text] == "BRCA1" }).to be true
      expect(entities.any? { |e| e[:text] == "CD4" }).to be true
      expect(entities.any? { |e| e[:text] == "CD8" }).to be true
      
      entities.each do |e|
        expect(e[:label]).to eq("GENE")
        expect(e[:confidence]).to eq(1.0)
        expect(e[:source]).to eq("pattern")
      end
    end
  end
  
  describe "GazetteerEntityRecognizer" do
    it "recognizes entities from dictionary" do
      genes = ["TP53", "BRCA1", "EGFR"]
      recognizer = Candle::GazetteerEntityRecognizer.new("CANCER_GENE", genes)
      
      text = "TP53 mutations affect BRCA1 function but not ABC1."
      entities = recognizer.recognize(text)
      
      expect(entities.length).to eq(2)
      expect(entities.any? { |e| e[:text] == "TP53" }).to be true
      expect(entities.any? { |e| e[:text] == "BRCA1" }).to be true
      # ABC1 not in gazetteer, so not recognized
      expect(entities.any? { |e| e[:text] == "ABC1" }).to be false
    end
    
    it "respects case sensitivity setting" do
      # Case insensitive
      recognizer_ci = Candle::GazetteerEntityRecognizer.new("GENE", ["TP53"], case_sensitive: false)
      entities = recognizer_ci.recognize("tp53 and TP53 and Tp53")
      expect(entities.length).to eq(3)
      
      # Case sensitive  
      recognizer_cs = Candle::GazetteerEntityRecognizer.new("GENE", ["TP53"], case_sensitive: true)
      entities = recognizer_cs.recognize("tp53 and TP53 and Tp53")
      expect(entities.length).to eq(1)
      expect(entities.first[:text]).to eq("TP53")
    end
  end
  
  describe "HybridNER" do
    it "works without model using only pattern/gazetteer recognizers" do
      # Test hybrid NER with only pattern/gazetteer recognizers
      hybrid = Candle::HybridNER.new
      
      # Add pattern recognizer
      hybrid.add_pattern_recognizer("GENE", [/\b[A-Z]{2,}\d*[A-Z]?\d*\b/])
      
      # Add gazetteer
      hybrid.add_gazetteer_recognizer("PROTEIN", ["p53", "p16", "p21"])
      
      text = "TP53 encodes the p53 protein, while CDKN2A encodes p16."
      entities = hybrid.extract_entities(text)
      
      # Should find pattern-based genes
      genes = entities.select { |e| e[:label] == "GENE" }
      expect(genes.any? { |e| e[:text] == "TP53" }).to be true
      expect(genes.any? { |e| e[:text] == "CDKN2A" }).to be true
      
      # Should find gazetteer-based proteins
      proteins = entities.select { |e| e[:label] == "PROTEIN" }
      expect(proteins.any? { |e| e[:text] == "p53" }).to be true
      expect(proteins.any? { |e| e[:text] == "p16" }).to be true
    end
    
    it "resolves entity overlaps" do
      hybrid = Candle::HybridNER.new
      
      # Add overlapping patterns
      hybrid.add_pattern_recognizer("ALPHANUM", [/[A-Z]+\d+/])
      hybrid.add_pattern_recognizer("GENE", [/TP53/])
      
      text = "TP53 is important"
      entities = hybrid.extract_entities(text)
      
      # Should only have one entity due to overlap resolution
      expect(entities.length).to eq(1)
      # First match wins (ALPHANUM comes first)
      expect(entities.first[:label]).to eq("ALPHANUM")
    end
  end
  
  describe "word boundary detection" do
    it "respects word boundaries" do
      recognizer = Candle::GazetteerEntityRecognizer.new("GENE", ["TP5"])
      
      # Should match with word boundaries
      entities = recognizer.recognize("TP5 gene")
      expect(entities.length).to eq(1)
      
      # Should not match within words
      entities = recognizer.recognize("TP53 gene")
      expect(entities.length).to eq(0)
    end
  end
  
  describe ".from_pretrained" do
    let(:ner) do
      @ner ||= Candle::NER.from_pretrained("dslim/bert-base-NER", tokenizer: "bert-base-cased")
    end
    
    after(:all) do
      @ner = nil
      GC.start
    end
    
    it "loads pretrained NER model" do
      expect(ner).to be_a(Candle::NER)
      
      # Test entity extraction
      text = "Apple Inc. was founded by Steve Jobs in Cupertino."
      entities = ner.extract_entities(text)
      
      # Check for expected entities
      expect(entities.any? { |e| e[:label] == "ORG" && e[:text].include?("Apple") }).to be true
      expect(entities.any? { |e| e[:label] == "PER" && e[:text].include?("Steve Jobs") }).to be true
      expect(entities.any? { |e| e[:label] == "LOC" && e[:text].include?("Cupertino") }).to be true
    end
  end
end