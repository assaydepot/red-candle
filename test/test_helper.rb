require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"

require "candle"

# Device testing helpers
module DeviceTestHelper
  # Cache device availability to avoid repeated checks
  AVAILABLE_DEVICES = begin
    devices = { cpu: true } # CPU is always available
    
    # Check Metal availability
    begin
      Candle::Device.metal
      Candle::Tensor.ones([1], device: Candle::Device.metal)
      devices[:metal] = true
    rescue
      devices[:metal] = false
    end
    
    # Check CUDA availability
    begin
      Candle::Device.cuda
      Candle::Tensor.ones([1], device: Candle::Device.cuda)
      devices[:cuda] = true
    rescue
      devices[:cuda] = false
    end
    
    devices
  end
  
  def available_devices
    AVAILABLE_DEVICES
  end
  
  def skip_unless_device_available(device_type)
    skip("#{device_type.to_s.upcase} device not available") unless AVAILABLE_DEVICES[device_type]
  end
  
  def create_device(device_type)
    case device_type
    when :cpu
      Candle::Device.cpu
    when :metal
      Candle::Device.metal
    when :cuda
      Candle::Device.cuda
    else
      raise ArgumentError, "Unknown device type: #{device_type}"
    end
  end
  
  # Get devices to test based on environment variable or all available
  def self.devices_to_test
    if ENV['CANDLE_TEST_DEVICES']
      # Allow CI/CD to specify which devices to test
      requested = ENV['CANDLE_TEST_DEVICES'].split(',').map(&:downcase).map(&:to_sym)
      requested.select { |d| AVAILABLE_DEVICES[d] }
    else
      # Test all available devices
      AVAILABLE_DEVICES.select { |_, available| available }.keys
    end
  end

  # Should we run benchmarks?
  def self.run_benchmarks?
    ENV['CANDLE_RUN_BENCHMARKS'] == 'true'
  end
end

# Print device availability once at test startup
if ENV['CANDLE_TEST_VERBOSE'] || ARGV.include?('-v') || ARGV.include?('--verbose')
  puts "\nCandle Test Environment:"
  puts "  Available devices: #{DeviceTestHelper::AVAILABLE_DEVICES.select { |_, v| v }.keys.join(', ')}"
  puts "  Testing devices: #{DeviceTestHelper.devices_to_test.join(', ')}"
  puts "  Benchmarks: #{DeviceTestHelper.run_benchmarks? ? 'enabled' : 'disabled'}"
  puts "  LLM tests: #{ENV['CANDLE_TEST_SKIP_LLM'] == 'true' ? 'skipped' : 'enabled'}"
  puts
end
