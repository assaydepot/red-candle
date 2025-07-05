#!/usr/bin/env ruby

require 'bundler/setup'
require 'candle'

puts "Testing Candle LayerNorm Implementation Details"
puts "=" * 60

# Test 1: Direct tensor operations (should work)
puts "\n--- Test 1: Direct Tensor Operations on Metal ---"
begin
  device = Candle::Device.metal
  
  # Create test data matching the Rust example
  batch_size = 2
  seq_len = 4
  hidden_size = 8
  
  input_data = (0...(batch_size * seq_len * hidden_size)).map { |i| i * 0.1 }
  input = Candle::Tensor.new(input_data, :f32).reshape([batch_size, seq_len, hidden_size]).to_device(device)
  
  gamma = Candle::Tensor.ones([hidden_size]).to_device(device)
  beta = Candle::Tensor.zeros([hidden_size]).to_device(device)
  
  puts "✓ Created tensors on Metal"
  puts "  Input: #{input.shape} on #{input.device}"
  puts "  Gamma: #{gamma.shape} on #{gamma.device}"
  puts "  Beta: #{beta.shape} on #{beta.device}"
  
rescue => e
  puts "✗ Failed: #{e.message}"
end

# Test 2: Check if tensors are contiguous
puts "\n--- Test 2: Tensor Contiguity Check ---"
begin
  device = Candle::Device.metal
  
  # Create a tensor
  t = Candle::Tensor.new([1.0, 2.0, 3.0, 4.0], :f32).reshape([2, 2]).to_device(device)
  puts "Created tensor: #{t.shape}"
  puts "Is contiguous: #{t.is_contiguous}"
  
  # Make it non-contiguous by transposing
  t_transposed = t.transpose(0, 1)
  puts "Transposed tensor: #{t_transposed.shape}"
  puts "Is contiguous: #{t_transposed.is_contiguous}"
  
  # Make it contiguous again
  t_contiguous = t_transposed.contiguous
  puts "Made contiguous: #{t_contiguous.shape}"
  puts "Is contiguous: #{t_contiguous.is_contiguous}"
  
rescue => e
  puts "✗ Failed: #{e.message}"
end

# Test 3: Try to replicate BERT's exact scenario
puts "\n--- Test 3: BERT-like Scenario ---"
begin
  device = Candle::Device.metal
  
  # BERT typically uses hidden_size=768
  batch_size = 1
  seq_len = 128
  hidden_size = 768
  
  # Create input similar to BERT embeddings output
  input_data = Array.new(batch_size * seq_len * hidden_size) { rand - 0.5 }
  input = Candle::Tensor.new(input_data, :f32).reshape([batch_size, seq_len, hidden_size]).to_device(device)
  
  # Layer norm parameters (weight and bias)
  weight_data = Array.new(hidden_size, 1.0)
  bias_data = Array.new(hidden_size, 0.0)
  
  weight = Candle::Tensor.new(weight_data, :f32).to_device(device)
  bias = Candle::Tensor.new(bias_data, :f32).to_device(device)
  
  puts "✓ Created BERT-like tensors on Metal"
  puts "  Input: #{input.shape}, contiguous: #{input.is_contiguous}"
  puts "  Weight: #{weight.shape}, contiguous: #{weight.is_contiguous}"
  puts "  Bias: #{bias.shape}, contiguous: #{bias.is_contiguous}"
  
  # The BERT model would call layer_norm here, which is where it fails
  # But we can't call it directly from Ruby
  
rescue => e
  puts "✗ Failed: #{e.message}"
end

# Test 4: Check what operations are available on Tensor
puts "\n--- Test 4: Available Tensor Operations ---"
begin
  t = Candle::Tensor.new([1.0], :f32)
  methods = t.methods.sort
  
  # Look for normalization-related methods
  norm_methods = methods.select { |m| m.to_s.include?("norm") || m.to_s.include?("layer") }
  puts "Normalization-related methods: #{norm_methods.join(', ')}"
  
  # Look for common operations
  common_ops = [:mean, :sum, :sqrt, :broadcast_add, :broadcast_sub, :broadcast_mul, :broadcast_div]
  available = common_ops.select { |op| methods.include?(op) }
  puts "Available common ops: #{available.join(', ')}"
  
rescue => e
  puts "✗ Failed: #{e.message}"
end

puts "\n" + "=" * 60
puts "The issue appears to be that candle_nn::ops::layer_norm is not exposed to Ruby"
puts "and the BERT models use it internally where it fails on Metal."