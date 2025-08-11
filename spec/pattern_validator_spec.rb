require "spec_helper"

RSpec.describe Candle::PatternValidator do
  describe ".validate" do
    context "with safe patterns" do
      it "marks simple patterns as safe" do
        result = described_class.validate(/test/)
        expect(result[:safe]).to be true
        expect(result[:warnings]).to be_empty
      end
      
      it "marks bounded quantifiers as safe" do
        result = described_class.validate(/[A-Z]{2,10}/)
        expect(result[:safe]).to be true
        expect(result[:warnings]).to be_empty
      end
      
      it "marks anchored patterns as safe" do
        result = described_class.validate(/^test$/)
        expect(result[:safe]).to be true
        expect(result[:warnings]).to be_empty
      end
    end
    
    context "with nested quantifiers" do
      it "detects nested quantifiers" do
        result = described_class.validate(/(\w+)*/)
        expect(result[:safe]).to be false
        expect(result[:warnings]).to include(/nested quantifiers/)
      end
      
      it "provides suggestions for nested quantifiers" do
        result = described_class.validate(/(\w+)*/)
        expect(result[:suggestions]).not_to be_empty
        expect(result[:suggestions].first).to include("atomic")
      end
    end
    
    context "with unbounded quantifiers" do
      it "warns about unbounded character class quantifiers" do
        result = described_class.validate(/[A-Z]+/)
        expect(result[:warnings]).to include(/unbounded quantifier/)
        expect(result[:suggestions]).to include(/upper bound/)
      end
      
      it "warns about {n,} quantifiers" do
        result = described_class.validate(/[A-Z]{2,}/)
        expect(result[:warnings]).to include(/unbounded quantifier/)
      end
      
      it "accepts reasonably bounded quantifiers" do
        result = described_class.validate(/[A-Z]{2,100}/)
        expect(result[:warnings]).to be_empty
      end
    end
    
    context "with alternation and quantifiers" do
      it "warns about alternation with quantifiers" do
        result = described_class.validate(/(ab|cd)+/)
        expect(result[:warnings]).to include(/alternation with quantifier/)
      end
    end
    
    context "with unanchored wildcards" do
      it "warns about leading .*" do
        result = described_class.validate(/.*test/)
        expect(result[:warnings]).to include(/unanchored/)
      end
      
      it "warns about trailing .*" do
        result = described_class.validate(/test.*/)
        expect(result[:warnings]).to include(/unanchored/)
      end
      
      it "accepts anchored wildcards" do
        result = described_class.validate(/^.*test$/)
        expect(result[:warnings]).not_to include(/unanchored/)
      end
    end
  end
  
  describe ".safe?" do
    it "returns true for safe patterns" do
      expect(described_class.safe?(/test/)).to be true
      expect(described_class.safe?(/[A-Z]{2,10}/)).to be true
    end
    
    it "returns false for unsafe patterns" do
      expect(described_class.safe?(/(\w+)*/)).to be false
    end
  end
  
  describe ".make_bounded" do
    it "bounds unbounded {n,} quantifiers" do
      result = described_class.make_bounded(/[A-Z]{2,}/)
      expect(result).to eq("[A-Z]{2,100}")
    end
    
    it "bounds + quantifiers" do
      result = described_class.make_bounded(/[A-Z]+/)
      expect(result).to eq("[A-Z]{1,100}")
    end
    
    it "bounds * quantifiers" do
      result = described_class.make_bounded(/[A-Z]*/)
      expect(result).to eq("[A-Z]{0,100}")
    end
    
    it "accepts custom max length" do
      result = described_class.make_bounded(/[A-Z]+/, 50)
      expect(result).to eq("[A-Z]{1,50}")
    end
    
    it "handles multiple quantifiers" do
      result = described_class.make_bounded(/[A-Z]+[0-9]*/)
      expect(result).to eq("[A-Z]{1,100}[0-9]{0,100}")
    end
  end
  
  describe "specific gene pattern" do
    it "detects issues with unbounded gene pattern" do
      pattern = /\b[A-Z][A-Z0-9]{2,}\b/
      result = described_class.validate(pattern)
      
      expect(result[:warnings]).not_to be_empty
      expect(result[:warnings].first).to include("unbounded")
    end
    
    it "approves bounded gene pattern" do
      pattern = /\b[A-Z][A-Z0-9]{2,10}\b/
      result = described_class.validate(pattern)
      
      expect(result[:warnings]).to be_empty
      expect(result[:safe]).to be true
    end
  end
end