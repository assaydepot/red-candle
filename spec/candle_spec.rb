require "spec_helper"

RSpec.describe Candle do
  describe "VERSION" do
    it "has a version number" do
      expect(::Candle::VERSION).not_to be_nil
    end
  end

  describe "basic tensor operations" do
    it "creates and manipulates tensors" do
      t = Candle::Tensor.new([3.0, 1, 4, 1, 5, 9, 2, 6], :f32)
      expect(t).to be_a(Candle::Tensor)
      expect(t.shape).to eq([8])
      
      t = t.reshape([2, 4])
      expect(t.shape).to eq([2, 4])
      
      t = t.t
      expect(t.shape).to eq([4, 2])

      t = Candle::Tensor.randn([5, 3])
      expect(t.shape).to eq([5, 3])
      expect(t.dtype).to be_a(Candle::DType)
    end
  end
end