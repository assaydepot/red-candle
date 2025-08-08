require 'spec_helper'

RSpec.describe Candle::Tensor do
  describe "scalar operations" do
    describe "#to_f" do
      context "with scalar tensor" do
        it "converts float scalar to float" do
          scalar = Candle::Tensor.new([3.14]).reshape([])
          expect(scalar.to_f).to be_within(0.001).of(3.14)
        end
        
        it "converts integer scalar to float" do
          int_scalar = Candle::Tensor.new([42], :i64).reshape([])
          expect(int_scalar.to_f).to eq(42.0)
        end
      end
      
      context "with non-scalar tensor" do
        it "raises ArgumentError" do
          tensor = Candle::Tensor.new([1, 2, 3])
          expect { tensor.to_f }.to raise_error(ArgumentError, /to_f can only be called on scalar tensors/)
        end
      end
    end
    
    describe "#to_i" do
      it "converts float scalar to integer" do
        float_scalar = Candle::Tensor.new([3.14]).reshape([])
        expect(float_scalar.to_i).to eq(3)
      end
      
      it "converts integer scalar to integer" do
        int_scalar = Candle::Tensor.new([42], :i64).reshape([])
        expect(int_scalar.to_i).to eq(42)
      end
    end
  end
  
  describe "#each" do
    context "with scalar tensor" do
      it "yields the single value" do
        scalar = Candle::Tensor.new([42.0]).reshape([])
        values = []
        scalar.each { |v| values << v }
        expect(values).to eq([42.0])
      end
    end
    
    context "with 1D tensor" do
      let(:tensor) { Candle::Tensor.new([1.0, 2.0, 3.0]) }
      
      it "yields each value" do
        values = []
        tensor.each { |v| values << v }
        expect(values).to eq([1.0, 2.0, 3.0])
      end
      
      it "is enumerable" do
        expect(tensor).to respond_to(:map, :select, :reduce)
      end
    end
    
    context "with 2D tensor" do
      let(:tensor) do
        Candle::Tensor.new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape([2, 3])
      end
      
      it "yields sub-tensors for each row" do
        sub_tensors = []
        tensor.each { |t| sub_tensors << t }
        
        expect(sub_tensors.length).to eq(2)
        expect(sub_tensors).to all(be_a(Candle::Tensor))
        expect(sub_tensors[0].values).to eq([1.0, 2.0, 3.0])
        expect(sub_tensors[1].values).to eq([4.0, 5.0, 6.0])
      end
    end
  end
  
  describe "Enumerable methods" do
    let(:tensor) { Candle::Tensor.new([1.0, 2.0, 3.0, 4.0, 5.0]) }
    
    it "supports map" do
      doubled = tensor.map { |v| v * 2 }
      expect(doubled).to eq([2.0, 4.0, 6.0, 8.0, 10.0])
    end
    
    it "supports select" do
      filtered = tensor.select { |v| v > 2.5 }
      expect(filtered).to eq([3.0, 4.0, 5.0])
    end
    
    it "supports reduce" do
      sum = tensor.reduce(0) { |acc, v| acc + v }
      expect(sum).to eq(15.0)
    end
  end
  
  describe "class method overrides" do
    describe ".new" do
      it "accepts device keyword argument" do
        tensor = Candle::Tensor.new([1, 2, 3], nil, device: Candle::Device.cpu)
        expect(tensor).to be_on_device(:cpu)
      end
    end
    
    describe ".ones" do
      it "accepts device keyword argument" do
        ones = Candle::Tensor.ones([2, 2], device: Candle::Device.cpu)
        expect(ones).to be_on_device(:cpu)
        expect(ones).to have_shape([2, 2])
      end
      
      it "creates a tensor filled with ones" do
        ones = Candle::Tensor.ones([2, 2])
        expect(ones.values.flatten).to all(eq(1.0))
      end
    end
    
    describe ".zeros" do
      it "accepts device keyword argument" do
        zeros = Candle::Tensor.zeros([2, 2], device: Candle::Device.cpu)
        expect(zeros).to be_on_device(:cpu)
        expect(zeros).to have_shape([2, 2])
      end
      
      it "creates a tensor filled with zeros" do
        zeros = Candle::Tensor.zeros([2, 2])
        expect(zeros.values.flatten).to all(eq(0.0))
      end
    end
    
    describe ".randn" do
      it "accepts device keyword argument" do
        randn = Candle::Tensor.randn([2, 2], device: Candle::Device.cpu)
        expect(randn).to be_on_device(:cpu)
        expect(randn).to have_shape([2, 2])
      end
      
      it "creates a tensor with random normal values" do
        randn = Candle::Tensor.randn([100])
        values = randn.values
        mean = values.sum / values.length
        expect(mean).to be_within(0.5).of(0.0)
      end
    end
  end
  
  # Testing across devices
  describe "device-specific operations" do
    DeviceHelpers.devices_to_test.each do |device_type|
      context "on #{device_type.to_s.upcase}" do
        let(:device) { create_device(device_type) }
        
        before do
          skip_unless_device_available(device_type)
        end
        
        it "creates tensors on the specified device" do
          tensor = Candle::Tensor.new([1, 2, 3], nil, device: device)
          expect(tensor).to be_on_device(device_type.to_s)
        end
        
        it "performs operations on the device" do
          a = Candle::Tensor.new([1.0, 2.0, 3.0], nil, device: device)
          b = Candle::Tensor.new([4.0, 5.0, 6.0], nil, device: device)
          result = a + b
          expect(result.values).to eq([5.0, 7.0, 9.0])
          expect(result).to be_on_device(device_type.to_s)
        end
      end
    end
  end
end