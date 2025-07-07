#!/usr/bin/env ruby

require 'bundler/setup'
require 'candle'

puts "Testing Direct Tensor Creation on Devices"
puts "=" * 50

# Test 1: Create tensors directly on CPU (default)
puts "\n1. Creating tensors on CPU (default):"
t1 = Candle::Tensor.ones([2, 3])
puts "   ones([2, 3]): device = #{t1.device}"

t2 = Candle::Tensor.new([1.0, 2.0, 3.0])
puts "   new([1, 2, 3]): device = #{t2.device}"

# Test 2: Create tensors directly on Metal
puts "\n2. Creating tensors directly on Metal:"
begin
  metal = Candle::Device.metal
  
  t3 = Candle::Tensor.ones([2, 3], device: metal)
  puts "   ones([2, 3], device: metal): device = #{t3.device}"
  
  t4 = Candle::Tensor.zeros([3, 3], device: metal)
  puts "   zeros([3, 3], device: metal): device = #{t4.device}"
  
  t5 = Candle::Tensor.rand([2, 2], device: metal)
  puts "   rand([2, 2], device: metal): device = #{t5.device}"
  
  t6 = Candle::Tensor.randn([4], device: metal)
  puts "   randn([4], device: metal): device = #{t6.device}"
  
  t7 = Candle::Tensor.new([1.0, 2.0, 3.0], :f32, device: metal)
  puts "   new([1, 2, 3], :f32, device: metal): device = #{t7.device}"
  
  # Verify the tensor is actually on Metal by doing an operation
  result = t3.sum(0)
  puts "   ✓ Operations work on Metal-created tensors"
  
rescue => e
  puts "   Metal not available: #{e.message}"
end

# Test 3: Performance comparison
puts "\n3. Performance comparison (creating large tensors):"
require 'benchmark'

size = [1000, 1000]
iterations = 10

begin
  metal = Candle::Device.metal
  
  cpu_time = Benchmark.realtime do
    iterations.times do
      t = Candle::Tensor.ones(size).to_device(metal)
    end
  end
  
  direct_time = Benchmark.realtime do
    iterations.times do
      t = Candle::Tensor.ones(size, device: metal)
    end
  end
  
  puts "   Create on CPU + move to Metal: #{(cpu_time * 1000).round(2)}ms"
  puts "   Create directly on Metal: #{(direct_time * 1000).round(2)}ms"
  puts "   Speedup: #{(cpu_time / direct_time).round(2)}x"
  
rescue => e
  puts "   Performance test skipped: #{e.message}"
end

# Test 4: Verify backward compatibility
puts "\n4. Backward compatibility:"
t8 = Candle::Tensor.ones([2, 2])
puts "   ones([2, 2]) without device: device = #{t8.device} ✓"

t9 = Candle::Tensor.new([1.0, 2.0], :f32)
puts "   new([1, 2], :f32) without device: device = #{t9.device} ✓"

puts "\nSummary:"
puts "- Tensor creation methods now accept an optional device parameter"
puts "- Creating directly on device avoids CPU->GPU copy overhead"
puts "- Backward compatibility maintained - device defaults to CPU"
puts "- Syntax: Candle::Tensor.ones(shape, device: device)"