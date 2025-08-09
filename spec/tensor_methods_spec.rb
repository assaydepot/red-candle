require "spec_helper"

RSpec.describe "TensorMethods" do
  describe "#to_f" do
    it "converts scalar tensor to float" do
      # Create a scalar tensor
      scalar = Candle::Tensor.new([3.14])
      scalar = scalar.reshape([])  # Make it a scalar
      expect(scalar.to_f).to be_within(0.001).of(3.14)
      
      # Test with integer scalar
      int_scalar = Candle::Tensor.new([42], :i64)
      int_scalar = int_scalar.reshape([])  # Make it a scalar
      expect(int_scalar.to_f).to eq(42.0)
    end
    
    it "raises error for non-scalar tensor" do
      # Create a 1D tensor
      tensor = Candle::Tensor.new([1, 2, 3])
      
      expect { tensor.to_f }.to raise_error(ArgumentError, /to_f can only be called on scalar tensors/)
    end
  end
  
  describe "#to_i" do
    it "converts scalar tensor to integer" do
      # Test float to int conversion
      float_scalar = Candle::Tensor.new([3.14])
      float_scalar = float_scalar.reshape([])  # Make it a scalar
      expect(float_scalar.to_i).to eq(3)
      
      # Test integer scalar
      int_scalar = Candle::Tensor.new([42], :i64)
      int_scalar = int_scalar.reshape([])  # Make it a scalar
      expect(int_scalar.to_i).to eq(42)
    end
  end
  
  describe "#each" do
    it "iterates over scalar tensor" do
      scalar = Candle::Tensor.new([42.0])
      scalar = scalar.reshape([])  # Make it a scalar
      values = []
      
      scalar.each { |v| values << v }
      
      expect(values).to eq([42.0])
    end
    
    it "iterates over 1D tensor" do
      tensor = Candle::Tensor.new([1.0, 2.0, 3.0])
      values = []
      
      tensor.each { |v| values << v }
      
      expect(values).to eq([1.0, 2.0, 3.0])
    end
    
    it "iterates over 2D tensor" do
      # Create a 2x3 tensor by flattening and reshaping
      tensor = Candle::Tensor.new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      tensor = tensor.reshape([2, 3])
      sub_tensors = []
      
      tensor.each { |t| sub_tensors << t }
      
      expect(sub_tensors.length).to eq(2)
      expect(sub_tensors[0]).to be_a(Candle::Tensor)
      expect(sub_tensors[1]).to be_a(Candle::Tensor)
      
      # Check the values of sub-tensors
      expect(sub_tensors[0].values).to eq([1.0, 2.0, 3.0])
      expect(sub_tensors[1].values).to eq([4.0, 5.0, 6.0])
    end
    
    it "iterates over different dtypes" do
      # Test with integer tensor
      int_tensor = Candle::Tensor.new([1, 2, 3], :i64)
      int_values = []
      int_tensor.each { |v| int_values << v }
      expect(int_values).to eq([1, 2, 3])
      
      # Test with float64 tensor if supported
      begin
        f64_tensor = Candle::Tensor.new([1.1, 2.2, 3.3], :f64)
        f64_values = []
        f64_tensor.each { |v| f64_values << v }
        expect(f64_values.length).to eq(3)
        expect(f64_values[0]).to be_within(0.001).of(1.1)
        expect(f64_values[1]).to be_within(0.001).of(2.2)
        expect(f64_values[2]).to be_within(0.001).of(3.3)
      rescue
        # F64 might not be supported on all devices
        skip "F64 dtype not supported on this device"
      end
    end
  end
  
  describe "Enumerable methods" do
    it "supports map, select, and reduce" do
      tensor = Candle::Tensor.new([1.0, 2.0, 3.0, 4.0, 5.0])
      
      # Test map
      doubled = tensor.map { |v| v * 2 }
      expect(doubled).to eq([2.0, 4.0, 6.0, 8.0, 10.0])
      
      # Test select
      filtered = tensor.select { |v| v > 2.5 }
      expect(filtered).to eq([3.0, 4.0, 5.0])
      
      # Test reduce
      sum = tensor.reduce(0) { |acc, v| acc + v }
      expect(sum).to eq(15.0)
    end
  end
  
  describe "class methods with device keyword" do
    it "creates tensors with specified device" do
      # Test new with device keyword
      tensor = Candle::Tensor.new([1, 2, 3], nil, device: Candle::Device.cpu)
      expect(tensor.device.to_s).to eq("cpu")
      
      # Test ones with device keyword
      ones = Candle::Tensor.ones([2, 2], device: Candle::Device.cpu)
      expect(ones.device.to_s).to eq("cpu")
      expect(ones.shape).to eq([2, 2])
      
      # Test zeros with device keyword
      zeros = Candle::Tensor.zeros([3], device: Candle::Device.cpu)
      expect(zeros.device.to_s).to eq("cpu")
      expect(zeros.shape).to eq([3])
      
      # Test rand with device keyword
      rand_tensor = Candle::Tensor.rand([2, 3], device: Candle::Device.cpu)
      expect(rand_tensor.device.to_s).to eq("cpu")
      expect(rand_tensor.shape).to eq([2, 3])
      
      # Test randn with device keyword
      randn_tensor = Candle::Tensor.randn([4], device: Candle::Device.cpu)
      expect(randn_tensor.device.to_s).to eq("cpu")
      expect(randn_tensor.shape).to eq([4])
    end
  end
end