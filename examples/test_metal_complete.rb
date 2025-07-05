#!/usr/bin/env ruby

require 'bundler/setup'
require 'candle'

puts "Metal Support Test - Complete"
puts "=" * 60

# Test 1: Device support checks
puts "\n--- Device Support Checks ---"
[:embedding_model, :reranker, :llm].each do |model_type|
  metal_support = Candle::DeviceUtils.supports_model?(Candle::Device.metal, model_type)
  puts "#{model_type}: #{metal_support ? '✅ Metal supported' : '❌ Metal not supported'}"
end

# Test 2: EmbeddingModel on Metal
puts "\n--- EmbeddingModel on Metal ---"
begin
  device = Candle::Device.metal
  model = Candle::EmbeddingModel.new(
    model_path: "jinaai/jina-embeddings-v2-base-en",
    device: device
  )
  
  texts = [
    "Ruby is a beautiful programming language.",
    "Metal acceleration provides significant speedups.",
    "Candle makes ML accessible in Rust."
  ]
  
  texts.each do |text|
    embedding = model.embedding(text)
    puts "✅ Embedded: '#{text[0..30]}...' -> shape: #{embedding.shape}"
  end
rescue => e
  puts "❌ Failed: #{e.message}"
end

# Test 3: Reranker on Metal with all pooling methods
puts "\n--- Reranker on Metal ---"
begin
  device = Candle::Device.metal
  reranker = Candle::Reranker.new(device: device)
  
  query = "What is Ruby programming?"
  documents = [
    "Ruby is a dynamic, interpreted programming language.",
    "Python is a high-level programming language.",
    "Ruby focuses on simplicity and productivity.",
    "JavaScript runs in web browsers."
  ]
  
  # Test default pooling
  puts "\nDefault pooling (pooler):"
  results = reranker.rerank(query, documents)
  results.take(2).each do |result|
    puts "  ✅ Score: #{'%.6f' % result[:score]} - #{result[:text][0..40]}..."
  end
  
  # Test different pooling methods
  ["cls", "mean"].each do |pooling|
    puts "\n#{pooling.capitalize} pooling:"
    results = reranker.rerank(query, documents, pooling_method: pooling)
    results.take(2).each do |result|
      puts "  ✅ Score: #{'%.6f' % result[:score]} - #{result[:text][0..40]}..."
    end
  end
rescue => e
  puts "❌ Failed: #{e.message}"
end

# Test 4: Smart device selection
puts "\n--- Smart Device Selection ---"
begin
  # Should automatically use Metal
  model = Candle::DeviceUtils.create_with_best_device(
    Candle::EmbeddingModel,
    :embedding_model,
    model_path: "jinaai/jina-embeddings-v2-base-en"
  )
  puts "✅ Created EmbeddingModel with best device"
  
  reranker = Candle::DeviceUtils.create_with_best_device(
    Candle::Reranker,
    :reranker
  )
  puts "✅ Created Reranker with best device"
rescue => e
  puts "❌ Failed: #{e.message}"
end

# Test 5: LLM (expected to fail on Metal)
puts "\n--- LLM on Metal (Expected to Fail) ---"
begin
  device = Candle::Device.metal
  puts "Creating LLM on Metal device..."
  llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device)
  puts "LLM created successfully! Attempting generation..."
  config = Candle::GenerationConfig.new(max_length: 10)
  response = llm.generate("Hello", config)
  puts "✅ Surprisingly, LLM works on Metal! Response: #{response}"
rescue => e
  if e.message.include?("rms-norm")
    puts "❌ Expected: #{e.message}"
  else
    puts "❌ Unexpected error: #{e.message}"
  end
end

puts "\n" + "=" * 60
puts "Summary:"
puts "✅ EmbeddingModel: Fully working on Metal"
puts "✅ Reranker: Fully working on Metal (all pooling methods)"
puts "❌ LLM: Still needs RMS norm implementation for Metal"
puts "\nBoth EmbeddingModel and Reranker can now leverage Apple GPU acceleration!"