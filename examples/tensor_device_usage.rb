#!/usr/bin/env ruby

require 'bundler/setup'
require 'candle'

puts "Tensor Device Usage Examples"
puts "=" * 40

# The correct pattern for creating tensors on specific devices
puts "\n1. Creating tensors on CPU (default):"
tensor_cpu = Candle::Tensor.ones([2, 3])
puts "   ones([2, 3]): device = #{tensor_cpu.device}"

tensor_cpu2 = Candle::Tensor.new([1.0, 2.0, 3.0], :f32)
puts "   new([1, 2, 3]): device = #{tensor_cpu2.device}"

puts "\n2. Moving tensors to a specific device:"
begin
  # Try Metal device
  metal_device = Candle::Device.metal
  tensor_metal = tensor_cpu.to_device(metal_device)
  puts "   Moved to Metal: device = #{tensor_metal.device}"
rescue => e
  puts "   Metal not available: #{e.message}"
end

puts "\n3. Creating and moving in one line:"
begin
  metal_device = Candle::Device.metal
  tensor = Candle::Tensor.zeros([3, 3]).to_device(metal_device)
  puts "   zeros([3, 3]).to_device(metal): device = #{tensor.device}"
rescue => e
  puts "   Metal not available: #{e.message}"
end

puts "\n4. New device parameter support:"
puts "   ✓ Candle::Tensor.ones([2, 3], device: device)   # Best - direct creation"
puts "   ✓ Candle::Tensor.ones([2, 3]).to_device(device) # Works but slower"

puts "\n5. Working with models and devices:"
puts "   Models accept device parameter directly:"
puts "   ✓ Candle::EmbeddingModel.new(model_path: 'path', device: device)"
puts "   ✓ Candle::Reranker.new(device: device)"
puts "   ✓ Candle::LLM.from_pretrained('model-id', device)"

puts "\n6. Device utilities:"
best_device = Candle::DeviceUtils.best_device
puts "   Best available device: #{best_device}"

puts "\nSummary:"
puts "- Tensor creation methods now accept optional device parameter"
puts "- Direct creation on device is much faster (avoids CPU->GPU copy)"
puts "- Backward compatible - defaults to CPU if device not specified"
puts "- Models accept device parameter during initialization"
puts "- DeviceUtils.best_device automatically selects the best available device"