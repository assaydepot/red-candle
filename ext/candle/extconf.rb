require "mkmf"
require "rb_sys/mkmf"

# Detect available hardware acceleration
features = []

# Force CPU-only build if requested
if ENV['CANDLE_FORCE_CPU']
  puts "CANDLE_FORCE_CPU is set, building CPU-only version"
else
  # Check for CUDA
  cuda_available = ENV['CUDA_ROOT'] || ENV['CUDA_PATH'] || ENV['CANDLE_CUDA_PATH'] ||
                   File.exist?('/usr/local/cuda') || File.exist?('/opt/cuda') ||
                   (RbConfig::CONFIG['host_os'] =~ /mswin|mingw|cygwin/ && 
                    (File.exist?('C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA') ||
                     File.exist?('C:\CUDA')))
  
  cuda_disabled = ENV['CANDLE_DISABLE_CUDA']
  
  if cuda_available && !cuda_disabled
    puts "CUDA detected, enabling CUDA support"
    features << 'cuda'
    
    # Check if CUDNN should be enabled
    if ENV['CANDLE_CUDNN'] || ENV['CUDNN_ROOT']
      puts "CUDNN support enabled"
      features << 'cudnn'
    end
  elsif cuda_available && cuda_disabled
    puts "=" * 80
    puts "CUDA detected but disabled via CANDLE_DISABLE_CUDA"
    puts "=" * 80
  end

  # Check for Metal (macOS only)
  if RbConfig::CONFIG['host_os'] =~ /darwin/
    puts "macOS detected, enabling Metal support"
    features << 'metal'
    
    # Also enable Accelerate framework on macOS
    puts "Enabling Accelerate framework support"
    features << 'accelerate'
  end

  # Check for Intel MKL
  mkl_available = ENV['MKLROOT'] || ENV['MKL_ROOT'] ||
                  File.exist?('/opt/intel/mkl') || 
                  File.exist?('/opt/intel/oneapi/mkl/latest')
  
  if mkl_available && !features.include?('accelerate')  # Don't use both MKL and Accelerate
    puts "Intel MKL detected, enabling MKL support"
    features << 'mkl'
  end
end

# Allow manual override of features
if ENV['CANDLE_FEATURES']
  manual_features = ENV['CANDLE_FEATURES'].split(',').map(&:strip)
  puts "Manual features override: #{manual_features.join(', ')}"
  features = manual_features
end

# Display selected features
unless features.empty?
  puts "Building with features: #{features.join(', ')}"
else
  puts "Building CPU-only version (no acceleration features detected)"
end

# Create the Rust makefile with proper feature configuration
create_rust_makefile("candle/candle") do |r|
  # Pass the features to rb_sys
  r.features = features unless features.empty?
  
  # Pass through any additional cargo flags
  if ENV['CANDLE_CARGO_FLAGS']
    r.extra_cargo_args = ENV['CANDLE_CARGO_FLAGS'].split(' ')
  end
end
