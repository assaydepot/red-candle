#!/usr/bin/env ruby

require 'bundler/setup'
require 'candle'

puts "Testing LLM on Metal"
puts "=" * 60

# Test 1: Try Mistral on Metal
puts "\n--- Mistral-7B on Metal ---"
begin
  device = Candle::Device.metal
  puts "Creating Mistral LLM on Metal..."
  llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device)
  puts "✅ Model loaded successfully!"
  
  config = Candle::GenerationConfig.new(
    max_length: 50,
    temperature: 0.7,
    top_p: 0.9
  )
  
  prompt = "The capital of France is"
  puts "\nPrompt: #{prompt}"
  puts "Generating..."
  
  response = llm.generate(prompt, config)
  puts "Response: #{response}"
  puts "\n✅ LLM works on Metal! This is unexpected but great!"
  
rescue => e
  puts "❌ Failed: #{e.message}"
  if e.message.include?("rms-norm")
    puts "This is expected - RMS norm is not implemented for Metal yet"
  end
end

# Test 2: Try with streaming
puts "\n--- Testing Streaming on Metal ---"
begin
  device = Candle::Device.metal
  llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device)
  
  config = Candle::GenerationConfig.new(max_length: 30)
  
  print "Streaming response: "
  llm.generate_stream("Once upon a time", config) do |token|
    print token
  end
  puts "\n✅ Streaming works on Metal too!"
  
rescue => e
  puts "\n❌ Streaming failed: #{e.message}"
end

# Test 3: Check if it's actually using Metal
puts "\n--- Verifying Metal Usage ---"
begin
  # Create tensors on Metal to verify the device is working
  device = Candle::Device.metal
  tensor = Candle::Tensor.new([1.0, 2.0, 3.0], :f32).to_device(device)
  puts "✅ Created tensor on Metal: #{tensor.device}"
  
  # Perform an operation
  result = tensor.sum(0)
  puts "✅ Sum operation on Metal: #{result.to_vec0}"
rescue => e
  puts "❌ Metal tensor operations failed: #{e.message}"
end

puts "\n" + "=" * 60
puts "Summary:"
puts "If the LLM is working on Metal, this means either:"
puts "1. RMS norm has been implemented in candle-metal-kernels"
puts "2. The model is somehow falling back to CPU for unsupported ops"
puts "3. There's a different normalization being used"
puts "\nThis needs further investigation!"