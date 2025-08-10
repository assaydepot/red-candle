require "spec_helper"
require "tempfile"

RSpec.describe Candle::GazetteerEntityRecognizer do
  describe "#initialize" do
    it "creates recognizer with entity type and terms" do
      recognizer = described_class.new("DRUG", ["aspirin", "ibuprofen"])
      expect(recognizer.entity_type).to eq("DRUG")
      expect(recognizer.terms).to include("aspirin", "ibuprofen")
    end
    
    it "handles case sensitivity option" do
      recognizer = described_class.new("TEST", ["Word"], case_sensitive: true)
      expect(recognizer.case_sensitive).to be true
    end
    
    it "defaults to case insensitive" do
      recognizer = described_class.new("TEST", ["word"])
      expect(recognizer.case_sensitive).to be false
    end
    
    it "normalizes terms when case insensitive" do
      recognizer = described_class.new("TEST", ["WORD", "Word", "word"])
      # Should deduplicate to single lowercase term
      expect(recognizer.terms.size).to eq(1)
      expect(recognizer.terms).to include("word")
    end
    
    it "preserves case when case sensitive" do
      recognizer = described_class.new("TEST", ["WORD", "Word", "word"], case_sensitive: true)
      expect(recognizer.terms.size).to eq(3)
      expect(recognizer.terms).to include("WORD", "Word", "word")
    end
  end
  
  describe "#add_terms" do
    let(:recognizer) { described_class.new("DRUG") }
    
    it "adds single term" do
      recognizer.add_terms("metformin")
      expect(recognizer.terms).to include("metformin")
    end
    
    it "adds array of terms" do
      recognizer.add_terms(["drug1", "drug2"])
      expect(recognizer.terms).to include("drug1", "drug2")
    end
    
    it "returns self for chaining" do
      result = recognizer.add_terms("drug1").add_terms("drug2")
      expect(result).to eq(recognizer)
      expect(recognizer.terms.size).to eq(2)
    end
    
    it "deduplicates terms when case insensitive" do
      recognizer.add_terms(["aspirin", "ASPIRIN", "Aspirin"])
      expect(recognizer.terms.size).to eq(1)
    end
  end
  
  describe "#load_from_file" do
    let(:recognizer) { described_class.new("COMPANY") }
    let(:temp_file) { Tempfile.new(["gazetteer", ".txt"]) }
    
    after { temp_file.unlink }
    
    it "loads terms from file" do
      temp_file.write("Apple\nGoogle\nMicrosoft\n")
      temp_file.flush
      
      recognizer.load_from_file(temp_file.path)
      expect(recognizer.terms).to include("apple", "google", "microsoft")
    end
    
    it "ignores empty lines" do
      temp_file.write("Apple\n\n\nGoogle\n")
      temp_file.flush
      
      recognizer.load_from_file(temp_file.path)
      expect(recognizer.terms.size).to eq(2)
    end
    
    it "ignores comment lines starting with #" do
      temp_file.write("# Companies\nApple\n# Tech companies\nGoogle\n")
      temp_file.flush
      
      recognizer.load_from_file(temp_file.path)
      expect(recognizer.terms).to include("apple", "google")
      expect(recognizer.terms).not_to include("# companies", "# tech companies")
    end
    
    it "returns self for chaining" do
      temp_file.write("Apple\n")
      temp_file.flush
      
      result = recognizer.load_from_file(temp_file.path).add_terms("Google")
      expect(result).to eq(recognizer)
      expect(recognizer.terms.size).to eq(2)
    end
  end
  
  describe "#recognize" do
    context "case insensitive matching" do
      let(:recognizer) { described_class.new("COMPANY", ["Apple", "Google", "Microsoft"]) }
      
      it "finds exact matches" do
        entities = recognizer.recognize("Apple makes iPhones")
        expect(entities.length).to eq(1)
        expect(entities.first[:text]).to eq("Apple")
        expect(entities.first[:label]).to eq("COMPANY")
        expect(entities.first[:source]).to eq("gazetteer")
        expect(entities.first[:confidence]).to eq(1.0)
      end
      
      it "finds case-insensitive matches" do
        entities = recognizer.recognize("APPLE and google work with MICROSOFT")
        expect(entities.length).to eq(3)
        expect(entities.map { |e| e[:text] }).to eq(["APPLE", "google", "MICROSOFT"])
      end
      
      it "respects word boundaries" do
        entities = recognizer.recognize("Apples are not Apple devices")
        expect(entities.length).to eq(1)
        expect(entities.first[:text]).to eq("Apple")
        expect(entities.first[:start]).to eq(15)  # Position of "Apple" not "Apples"
      end
      
      it "finds multiple occurrences" do
        entities = recognizer.recognize("Apple vs Apple: Apple wins")
        expect(entities.length).to eq(3)
        expect(entities.all? { |e| e[:text] == "Apple" }).to be true
      end
    end
    
    context "case sensitive matching" do
      let(:recognizer) do
        described_class.new("TERM", ["Apple", "apple"], case_sensitive: true)
      end
      
      it "distinguishes case" do
        entities = recognizer.recognize("Apple apple APPLE")
        expect(entities.length).to eq(2)
        expect(entities.map { |e| e[:text] }).to eq(["Apple", "apple"])
      end
    end
    
    context "multi-word terms" do
      let(:recognizer) do
        described_class.new("PERSON", ["Steve Jobs", "Tim Cook", "Bill Gates"])
      end
      
      it "matches multi-word terms" do
        entities = recognizer.recognize("Steve Jobs founded Apple, now led by Tim Cook")
        expect(entities.length).to eq(2)
        expect(entities.map { |e| e[:text] }).to eq(["Steve Jobs", "Tim Cook"])
      end
      
      it "respects word boundaries for multi-word terms" do
        # Should not match "Steve JobsX" or "XSteve Jobs"
        entities = recognizer.recognize("Steve JobsX and XSteve Jobs")
        expect(entities).to be_empty
      end
    end
    
    context "position information" do
      let(:recognizer) { described_class.new("FRUIT", ["apple", "banana"]) }
      
      it "provides correct start and end positions" do
        text = "I like apple and banana"
        entities = recognizer.recognize(text)
        
        expect(entities[0][:start]).to eq(7)
        expect(entities[0][:end]).to eq(12)
        expect(text[7...12]).to eq("apple")
        
        expect(entities[1][:start]).to eq(17)
        expect(entities[1][:end]).to eq(23)
        expect(text[17...23]).to eq("banana")
      end
    end
    
    context "edge cases" do
      let(:recognizer) { described_class.new("TEST", ["test"]) }
      
      it "handles empty text" do
        expect(recognizer.recognize("")).to be_empty
      end
      
      it "handles text without matches" do
        expect(recognizer.recognize("no matches here")).to be_empty
      end
      
      it "handles terms at start of text" do
        entities = recognizer.recognize("test this")
        expect(entities.first[:start]).to eq(0)
      end
      
      it "handles terms at end of text" do
        entities = recognizer.recognize("this test")
        expect(entities.first[:start]).to eq(5)
        expect(entities.first[:end]).to eq(9)
      end
    end
    
    context "performance" do
      it "handles large gazetteers efficiently" do
        terms = (1..1000).map { |i| "term#{i}" }
        recognizer = described_class.new("TEST", terms)
        text = "This text contains term500 and term999"
        
        start_time = Time.now
        entities = recognizer.recognize(text)
        elapsed = Time.now - start_time
        
        expect(entities.length).to eq(2)
        expect(elapsed).to be < 0.1
      end
      
      it "handles long text efficiently" do
        recognizer = described_class.new("TEST", ["specific"])
        text = "word " * 10000
        
        start_time = Time.now
        entities = recognizer.recognize(text)
        elapsed = Time.now - start_time
        
        expect(entities).to be_empty
        expect(elapsed).to be < 0.5
      end
    end
  end
end