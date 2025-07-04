#!/usr/bin/env ruby

require 'bundler/setup'
require 'candle'

puts "Testing Metal Device Operations"
puts "=" * 60

# Test basic tensor operations on Metal
begin
  puts "\n--- Basic Tensor Operations on Metal ---"
  metal = Candle::Device.metal
  
  # Create tensors on CPU first, then move to Metal
  a = Candle::Tensor.new([1.0, 2.0, 3.0, 4.0], :f32).reshape([2, 2]).to_device(metal)
  b = Candle::Tensor.new([5.0, 6.0, 7.0, 8.0], :f32).reshape([2, 2]).to_device(metal)
  
  puts "✓ Created tensors on Metal"
  
  # Test basic operations
  c = a + b
  puts "✓ Addition works: #{c.values}"
  
  d = a.matmul(b)
  puts "✓ Matrix multiplication works: #{d.values}"
  
  e = a.sum(0)
  puts "✓ Sum works: #{e.values}"
  
  f = a.mean(0)
  puts "✓ Mean works: #{f.values}"
  
  # Test more complex operations
  g = a.sqrt()
  puts "✓ Square root works"
  
  h = a.exp()
  puts "✓ Exponential works"
  
rescue => e
  puts "Error: #{e.message}"
end

# Test what happens with models
puts "\n--- Model Operations on Metal ---"
puts "Note: Some operations like RMS normalization are not yet implemented for Metal."
puts "This is a known limitation in the current version of Candle."

# Show available devices
puts "\n--- Available Devices ---"
puts "CPU: #{Candle::Device.cpu.inspect}"
puts "Metal: #{Candle::Device.metal.inspect}"
begin
  puts "CUDA: #{Candle::Device.cuda.inspect}"
rescue => e
  puts "CUDA: Not available (#{e.message})"
end

puts "\n" + "=" * 60
puts "For LLM inference, use CPU until Metal operations are fully implemented."