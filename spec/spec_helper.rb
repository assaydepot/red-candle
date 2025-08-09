# SimpleCov must be started before requiring the code to track
require 'simplecov'
SimpleCov.start do
  add_filter '/spec/'
  add_filter '/test/'
  add_filter '/vendor/'
  add_filter '/ext/'  # Native extensions can't be tracked
  track_files 'lib/**/*.rb'
  
  # Add groups for better organization
  add_group 'Models', 'lib/candle'
  add_group 'Core', 'lib/candle.rb'
end

require 'bundler/setup'
Bundler.require(:default)

# Load the library
require 'candle'

# Load support files
Dir[File.join(__dir__, 'support', '**', '*.rb')].each { |f| require f }

RSpec.configure do |config|
  # Enable flags like --only-failures and --next-failure
  config.example_status_persistence_file_path = ".rspec_status"

  # Disable monkey patching
  config.disable_monkey_patching!

  # Use expect syntax only
  config.expect_with :rspec do |c|
    c.syntax = :expect
  end

  # Random order with seed
  config.order = :random
  Kernel.srand config.seed

  # Filter examples by tags
  config.filter_run_when_matching :focus
  
  # Run specs in random order to surface order dependencies
  config.order = :random
  
  # Print the slowest examples and example groups
  config.profile_examples = 10 if ENV['PROFILE_SPECS']

  # Include helpers
  config.include DeviceHelpers
  config.include SimpleModelCache
  
  # Suppress warnings during tests (same as Minitest setup)
  config.before(:suite) do
    $VERBOSE = nil
    def Kernel.warn(*args); end
  end
  
  # Print test environment info
  config.before(:suite) do
    if ENV['CANDLE_TEST_VERBOSE'] || ARGV.include?('-v')
      puts "\nCandle RSpec Test Environment:"
      puts "  Ruby version: #{RUBY_VERSION}"
      puts "  Available devices: #{DeviceHelpers.available_devices.join(', ')}"
      puts "  Testing devices: #{DeviceHelpers.devices_to_test.join(', ')}"
      puts
    end
  end
end