require "test_helper"

class BuildInfoTest < Minitest::Test  
  def test_cuda_available
    # This method calls into Candle.build_info which is from native extension
    result = Candle::BuildInfo.cuda_available?
    assert [true, false].include?(result), "cuda_available? should return boolean"
  end
  
  def test_metal_available
    result = Candle::BuildInfo.metal_available?
    assert [true, false].include?(result), "metal_available? should return boolean"
  end
  
  def test_mkl_available
    result = Candle::BuildInfo.mkl_available?
    assert [true, false].include?(result), "mkl_available? should return boolean"
  end
  
  def test_accelerate_available
    result = Candle::BuildInfo.accelerate_available?
    assert [true, false].include?(result), "accelerate_available? should return boolean"
  end
  
  def test_cudnn_available
    result = Candle::BuildInfo.cudnn_available?
    assert [true, false].include?(result), "cudnn_available? should return boolean"
  end
  
  def test_summary
    summary = Candle::BuildInfo.summary
    
    assert_instance_of Hash, summary
    assert_includes summary.keys, :default_device
    assert_includes summary.keys, :available_backends
    assert_includes summary.keys, :cuda_available
    assert_includes summary.keys, :metal_available
    assert_includes summary.keys, :mkl_available
    assert_includes summary.keys, :accelerate_available
    assert_includes summary.keys, :cudnn_available
    
    # Check available_backends always includes CPU
    assert_includes summary[:available_backends], "CPU"
    
    # Check that available_backends is consistent with flags
    if summary[:metal_available]
      assert_includes summary[:available_backends], "Metal"
    else
      refute_includes summary[:available_backends], "Metal"
    end
    
    if summary[:cuda_available]
      assert_includes summary[:available_backends], "CUDA"
    else
      refute_includes summary[:available_backends], "CUDA"
    end
  end
  
  def test_summary_backend_consistency
    summary = Candle::BuildInfo.summary
    backends = summary[:available_backends]
    
    # Backends should be an array
    assert_instance_of Array, backends
    
    # Should always have at least CPU
    assert backends.length >= 1, "Should have at least CPU backend"
    
    # Each backend should be a string
    backends.each do |backend|
      assert_instance_of String, backend
    end
    
    # Valid backend names
    valid_backends = ["CPU", "CUDA", "Metal"]
    backends.each do |backend|
      assert_includes valid_backends, backend, "Unknown backend: #{backend}"
    end
  end
end