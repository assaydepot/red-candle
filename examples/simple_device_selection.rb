#!/usr/bin/env ruby

require 'bundler/setup'
require 'candle'

puts "Simple Device Selection Example"
puts "=" * 40

# Method 1: Let red-candle pick the best device
puts "\n1. Automatic device selection:"
best = Candle::DeviceUtils.best_device
puts "   Best device: #{best.inspect}"

# Create models with automatic device selection
embedding_model = Candle::DeviceUtils.create_with_best_device(
  Candle::EmbeddingModel,
  model_path: "jinaai/jina-embeddings-v2-base-en"
)
puts "   ✓ Created EmbeddingModel"

reranker = Candle::DeviceUtils.create_with_best_device(Candle::Reranker)
puts "   ✓ Created Reranker"

# Method 2: Explicitly specify device
puts "\n2. Explicit device selection:"
begin
  # Try Metal
  device = Candle::Device.metal
  puts "   Using Metal device"
rescue
  # Fall back to CPU
  device = Candle::Device.cpu
  puts "   Using CPU device"
end

model = Candle::EmbeddingModel.new(
  model_path: "jinaai/jina-embeddings-v2-base-en",
  device: device
)
puts "   ✓ Created model with #{device.inspect}"

# Test the model
puts "\n3. Testing model:"
embedding = model.embedding("Hello world!")
puts "   ✓ Generated embedding with shape: #{embedding.shape}"

puts "\nDone! All models now work on all devices."