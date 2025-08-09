module DeviceHelpers
  # Cache device availability to avoid repeated checks
  AVAILABLE_DEVICES = begin
    devices = {}
    
    # CPU is always available
    devices[:cpu] = true
    
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
  
  def self.available_devices
    AVAILABLE_DEVICES.select { |_, available| available }.keys
  end
  
  def self.devices_to_test
    if ENV['CANDLE_TEST_DEVICES']
      # Allow CI/CD to specify which devices to test
      requested = ENV['CANDLE_TEST_DEVICES'].split(',').map(&:downcase).map(&:to_sym)
      requested.select { |d| AVAILABLE_DEVICES[d] }
    else
      # Test all available devices
      available_devices
    end
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
  
  # Helper to test across all available devices
  # This should be used at the describe/context level
  def self.with_each_device(&block)
    DeviceHelpers.devices_to_test.each do |device_type|
      RSpec.describe "on #{device_type.to_s.upcase}" do
        let(:device) { create_device(device_type) }
        
        before do
          skip_unless_device_available(device_type)
        end
        
        instance_eval(&block)
      end
    end
  end
end