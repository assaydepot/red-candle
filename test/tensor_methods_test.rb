require "test_helper"

class TensorMethodsTest < Minitest::Test
  def test_to_f_with_scalar_tensor
    # Create a scalar tensor
    scalar = Candle::Tensor.new([3.14])
    scalar = scalar.reshape([])  # Make it a scalar
    assert_in_delta 3.14, scalar.to_f, 0.001
    
    # Test with integer scalar
    int_scalar = Candle::Tensor.new([42], :i64)
    int_scalar = int_scalar.reshape([])  # Make it a scalar
    assert_equal 42.0, int_scalar.to_f
  end
  
  def test_to_f_with_non_scalar_raises_error
    # Create a 1D tensor
    tensor = Candle::Tensor.new([1, 2, 3])
    
    error = assert_raises(ArgumentError) do
      tensor.to_f
    end
    
    assert_match(/to_f can only be called on scalar tensors/, error.message)
  end
  
  def test_to_i_with_scalar_tensor
    # Test float to int conversion
    float_scalar = Candle::Tensor.new([3.14])
    float_scalar = float_scalar.reshape([])  # Make it a scalar
    assert_equal 3, float_scalar.to_i
    
    # Test integer scalar
    int_scalar = Candle::Tensor.new([42], :i64)
    int_scalar = int_scalar.reshape([])  # Make it a scalar
    assert_equal 42, int_scalar.to_i
  end
  
  def test_each_with_scalar_tensor
    scalar = Candle::Tensor.new([42.0])
    scalar = scalar.reshape([])  # Make it a scalar
    values = []
    
    scalar.each { |v| values << v }
    
    assert_equal [42.0], values
  end
  
  def test_each_with_1d_tensor
    tensor = Candle::Tensor.new([1.0, 2.0, 3.0])
    values = []
    
    tensor.each { |v| values << v }
    
    assert_equal [1.0, 2.0, 3.0], values
  end
  
  def test_each_with_2d_tensor
    # Create a 2x3 tensor by flattening and reshaping
    tensor = Candle::Tensor.new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    tensor = tensor.reshape([2, 3])
    sub_tensors = []
    
    tensor.each { |t| sub_tensors << t }
    
    assert_equal 2, sub_tensors.length
    assert_instance_of Candle::Tensor, sub_tensors[0]
    assert_instance_of Candle::Tensor, sub_tensors[1]
    
    # Check the values of sub-tensors
    assert_equal [1.0, 2.0, 3.0], sub_tensors[0].values
    assert_equal [4.0, 5.0, 6.0], sub_tensors[1].values
  end
  
  def test_enumerable_methods
    tensor = Candle::Tensor.new([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Test map
    doubled = tensor.map { |v| v * 2 }
    assert_equal [2.0, 4.0, 6.0, 8.0, 10.0], doubled
    
    # Test select
    filtered = tensor.select { |v| v > 2.5 }
    assert_equal [3.0, 4.0, 5.0], filtered
    
    # Test reduce
    sum = tensor.reduce(0) { |acc, v| acc + v }
    assert_equal 15.0, sum
  end
  
  def test_class_method_overrides_with_device
    # Test new with device keyword
    tensor = Candle::Tensor.new([1, 2, 3], nil, device: Candle::Device.cpu)
    assert_equal "cpu", tensor.device.to_s
    
    # Test ones with device keyword
    ones = Candle::Tensor.ones([2, 2], device: Candle::Device.cpu)
    assert_equal "cpu", ones.device.to_s
    assert_equal [2, 2], ones.shape
    
    # Test zeros with device keyword
    zeros = Candle::Tensor.zeros([3], device: Candle::Device.cpu)
    assert_equal "cpu", zeros.device.to_s
    assert_equal [3], zeros.shape
    
    # Test rand with device keyword
    rand_tensor = Candle::Tensor.rand([2, 3], device: Candle::Device.cpu)
    assert_equal "cpu", rand_tensor.device.to_s
    assert_equal [2, 3], rand_tensor.shape
    
    # Test randn with device keyword
    randn_tensor = Candle::Tensor.randn([4], device: Candle::Device.cpu)
    assert_equal "cpu", randn_tensor.device.to_s
    assert_equal [4], randn_tensor.shape
  end
  
  def test_each_with_different_dtypes
    # Test with integer tensor
    int_tensor = Candle::Tensor.new([1, 2, 3], :i64)
    int_values = []
    int_tensor.each { |v| int_values << v }
    assert_equal [1, 2, 3], int_values
    
    # Test with float64 tensor if supported
    begin
      f64_tensor = Candle::Tensor.new([1.1, 2.2, 3.3], :f64)
      f64_values = []
      f64_tensor.each { |v| f64_values << v }
      assert_equal 3, f64_values.length
      assert_in_delta 1.1, f64_values[0], 0.001
      assert_in_delta 2.2, f64_values[1], 0.001
      assert_in_delta 3.3, f64_values[2], 0.001
    rescue
      # F64 might not be supported on all devices
      skip "F64 dtype not supported on this device"
    end
  end
end
