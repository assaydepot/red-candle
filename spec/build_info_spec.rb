require "spec_helper"

RSpec.describe "BuildInfo" do
  describe ".cuda_available?" do
    it "returns boolean" do
      result = Candle::BuildInfo.cuda_available?
      expect([true, false]).to include(result)
    end
  end
  
  describe ".metal_available?" do
    it "returns boolean" do
      result = Candle::BuildInfo.metal_available?
      expect([true, false]).to include(result)
    end
  end
  
  describe ".mkl_available?" do
    it "returns boolean" do
      result = Candle::BuildInfo.mkl_available?
      expect([true, false]).to include(result)
    end
  end
  
  describe ".accelerate_available?" do
    it "returns boolean" do
      result = Candle::BuildInfo.accelerate_available?
      expect([true, false]).to include(result)
    end
  end
  
  describe ".cudnn_available?" do
    it "returns boolean" do
      result = Candle::BuildInfo.cudnn_available?
      expect([true, false]).to include(result)
    end
  end
  
  describe ".summary" do
    let(:summary) { Candle::BuildInfo.summary }
    
    it "returns a hash with all expected keys" do
      expect(summary).to be_a(Hash)
      expect(summary.keys).to include(:default_device)
      expect(summary.keys).to include(:available_backends)
      expect(summary.keys).to include(:cuda_available)
      expect(summary.keys).to include(:metal_available)
      expect(summary.keys).to include(:mkl_available)
      expect(summary.keys).to include(:accelerate_available)
      expect(summary.keys).to include(:cudnn_available)
    end
    
    it "always includes CPU in available_backends" do
      expect(summary[:available_backends]).to include("CPU")
    end
    
    it "has consistent Metal backend reporting" do
      if summary[:metal_available]
        expect(summary[:available_backends]).to include("Metal")
      else
        expect(summary[:available_backends]).not_to include("Metal")
      end
    end
    
    it "has consistent CUDA backend reporting" do
      if summary[:cuda_available]
        expect(summary[:available_backends]).to include("CUDA")
      else
        expect(summary[:available_backends]).not_to include("CUDA")
      end
    end
    
    context "backend consistency" do
      let(:backends) { summary[:available_backends] }
      
      it "returns an array of backends" do
        expect(backends).to be_an(Array)
      end
      
      it "has at least CPU backend" do
        expect(backends.length).to be >= 1
      end
      
      it "has string backend names" do
        backends.each do |backend|
          expect(backend).to be_a(String)
        end
      end
      
      it "has only valid backend names" do
        valid_backends = ["CPU", "CUDA", "Metal"]
        backends.each do |backend|
          expect(valid_backends).to include(backend)
        end
      end
    end
  end
end