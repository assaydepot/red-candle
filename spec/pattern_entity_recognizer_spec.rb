require "spec_helper"

RSpec.describe Candle::PatternEntityRecognizer do
  describe "#initialize" do
    it "creates recognizer with entity type and patterns" do
      recognizer = described_class.new("GENE", [/TP53/, /BRCA1/])
      expect(recognizer.entity_type).to eq("GENE")
      expect(recognizer.patterns).to eq([/TP53/, /BRCA1/])
    end
    
    it "creates recognizer without patterns" do
      recognizer = described_class.new("TEST")
      expect(recognizer.patterns).to be_empty
    end
  end
  
  describe "#add_pattern" do
    let(:recognizer) { described_class.new("GENE") }
    
    it "adds regex patterns" do
      recognizer.add_pattern(/\b[A-Z]{3}\d+\b/)
      expect(recognizer.patterns).to include(/\b[A-Z]{3}\d+\b/)
    end
    
    it "adds string patterns" do
      recognizer.add_pattern("ID-\\d{4}")
      expect(recognizer.patterns).to include("ID-\\d{4}")
    end
    
    it "returns self for chaining" do
      result = recognizer.add_pattern(/test/).add_pattern(/another/)
      expect(result).to eq(recognizer)
      expect(recognizer.patterns.length).to eq(2)
    end
  end
  
  describe "#recognize" do
    context "with regex patterns" do
      let(:recognizer) do
        described_class.new("GENE", [
          /\b[A-Z][A-Z0-9]{2,10}\b/,  # Safe bounded pattern
          /\bCD\d+\b/
        ])
      end
      
      it "finds matching entities" do
        text = "TP53 and BRCA1 mutations affect CD4+ cells"
        entities = recognizer.recognize(text)
        
        # CD4 matches both patterns, so we get 4 entities total
        expect(entities.length).to eq(4)
        # But unique texts are only 3
        expect(entities.map { |e| e[:text] }.uniq).to contain_exactly("TP53", "BRCA1", "CD4")
        expect(entities.all? { |e| e[:label] == "GENE" }).to be true
        expect(entities.all? { |e| e[:source] == "pattern" }).to be true
        expect(entities.all? { |e| e[:confidence] == 1.0 }).to be true
      end
      
      it "returns empty array for no matches" do
        entities = recognizer.recognize("no genes here")
        expect(entities).to be_empty
      end
      
      it "includes correct position information" do
        text = "TP53 is here"
        entities = recognizer.recognize(text)
        
        expect(entities.first[:start]).to eq(0)
        expect(entities.first[:end]).to eq(4)
        expect(text[entities.first[:start]...entities.first[:end]]).to eq("TP53")
      end
    end
    
    context "with string patterns" do
      let(:recognizer) { described_class.new("ID") }
      
      before do
        recognizer.add_pattern("ID-\\d{4}")
      end
      
      it "converts string patterns to regex" do
        text = "User ID-1234 and ID-5678"
        entities = recognizer.recognize(text)
        
        expect(entities.length).to eq(2)
        expect(entities.map { |e| e[:text] }).to eq(["ID-1234", "ID-5678"])
      end
    end
    
    context "with overlapping patterns" do
      let(:recognizer) do
        described_class.new("TEST", [/test/, /test\w+/])
      end
      
      it "finds all matching patterns" do
        entities = recognizer.recognize("testing")
        # Should find both "test" and "testing"
        expect(entities.length).to eq(2)
        expect(entities.map { |e| e[:text] }).to contain_exactly("test", "testing")
      end
    end
    
    context "with multiple occurrences" do
      let(:recognizer) { described_class.new("EMAIL", [/\b[\w.]+@[\w.]+\b/]) }
      
      it "finds all occurrences" do
        text = "Contact alice@example.com or bob@test.org"
        entities = recognizer.recognize(text)
        
        expect(entities.length).to eq(2)
        expect(entities.map { |e| e[:text] }).to contain_exactly("alice@example.com", "bob@test.org")
      end
    end
    
    context "edge cases" do
      let(:recognizer) { described_class.new("TEST", [/test/i]) }
      
      it "handles empty text" do
        expect(recognizer.recognize("")).to be_empty
      end
      
      it "handles case-insensitive patterns" do
        entities = recognizer.recognize("Test TEST test")
        expect(entities.length).to eq(3)
      end
    end
    
    context "performance" do
      it "handles long text efficiently" do
        recognizer = described_class.new("WORD", [/\b\w{5}\b/])
        text = "word " * 10000  # 50,000 character text
        
        start_time = Time.now
        entities = recognizer.recognize(text)
        elapsed = Time.now - start_time
        
        expect(entities).to be_empty  # "word " has 4 letters, not 5
        expect(elapsed).to be < 1.0
      end
      
      it "handles bounded quantifiers efficiently" do
        recognizer = described_class.new("ID", [/[A-Z]{2,10}\d{2,10}/])
        text = "AA11 BBB222 CCCC3333 " * 100
        
        start_time = Time.now
        entities = recognizer.recognize(text)
        elapsed = Time.now - start_time
        
        expect(entities.length).to eq(300)
        expect(elapsed).to be < 0.5
      end
    end
  end
  
  describe "ReDoS protection" do
    context "with problematic patterns from security alert" do
      let(:recognizer) do
        described_class.new("GENE", [
          # CodeQL Alert: These patterns are intentionally included to test ReDoS protection
          /\b[A-Z][A-Z0-9]{2,}\b/,  # This could cause ReDoS without protection
          /\bCD\d+\b/               # This is safe
        ])
      end
      
      it "handles normal input correctly" do
        text = "TP53 and CD4 are important"
        entities = recognizer.recognize(text)
        # CD4 matches both patterns
        expect(entities.length).to eq(3)
        expect(entities.map { |e| e[:text] }.uniq).to contain_exactly("TP53", "CD4")
      end
      
      it "completes quickly even with problematic input" do
        # This would cause exponential time without protections
        problematic = "test A" + "0" * 1000 + "00000 test"
        
        start_time = Time.now
        entities = recognizer.recognize(problematic)
        elapsed = Time.now - start_time
        
        # Should match the very long string
        expect(entities.length).to eq(1)
        expect(entities[0][:text]).to start_with("A000")
        expect(elapsed).to be < 1.0
      end
    end
    
    context "with nested quantifier patterns" do
      it "handles alternation patterns efficiently" do
        recognizer = described_class.new("TEST", [/(ab|cd)+/])
        text = "abcdabcd" * 100
        
        start_time = Time.now
        entities = recognizer.recognize(text)
        elapsed = Time.now - start_time
        
        expect(entities).not_to be_empty
        expect(elapsed).to be < 1.0
      end
      
      it "handles character class patterns efficiently" do
        recognizer = described_class.new("ALPHANUM", [/[A-Za-z0-9]+/])
        text = "a1b2c3" * 1000
        
        start_time = Time.now
        entities = recognizer.recognize(text)
        elapsed = Time.now - start_time
        
        expect(entities.length).to eq(1)  # One long match
        expect(elapsed).to be < 0.5
      end
    end
    
    context "with text length protection" do
      it "truncates very long text" do
        recognizer = described_class.new("TEST", [/test/])
        long_text = "test" + "a" * 1_000_000
        
        # Should process but truncate
        entities = recognizer.recognize(long_text)
        
        # The test pattern should be found since it's at the beginning
        expect(entities.length).to eq(1)
        expect(entities.first[:text]).to eq("test")
      end
    end
    
    context "with pattern validation" do
      it "handles patterns with nested quantifiers" do
        recognizer = described_class.new("TEST")
        # Intentionally problematic pattern for testing ReDoS protection
        # CodeQL Alert: This pattern is intentionally vulnerable to test our protection
        recognizer.add_pattern(/(\w+)*$/)
        
        # Even with nested quantifiers, should complete quickly
        start_time = Time.now
        entities = recognizer.recognize("test text")
        elapsed = Time.now - start_time
        
        expect(elapsed).to be < 0.1
        expect(entities).not_to be_empty
      end
      
      it "handles safe patterns efficiently" do
        recognizer = described_class.new("GENE")
        recognizer.add_pattern(/\b[A-Z][A-Z0-9]{2,10}\b/)
        
        entities = recognizer.recognize("TP53 and BRCA1")
        expect(entities.length).to eq(2)
      end
    end
  end
end