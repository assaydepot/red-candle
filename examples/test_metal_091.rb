#!/usr/bin/env ruby

require 'bundler/setup'
require 'candle'

puts "Testing Metal Support with Candle 0.9.1"
puts "=" * 60

# Test EmbeddingModel on Metal
puts "\n--- Testing EmbeddingModel on Metal ---"
begin
  device = Candle::Device.metal
  puts "Creating EmbeddingModel on Metal..."
  
  model = Candle::EmbeddingModel.new(
    model_path: "jinaai/jina-embeddings-v2-base-en",
    device: device
  )
  
  puts "Generating embedding..."
  embedding = model.embedding("Hello world!")
  puts "✓ Success! Embedding shape: #{embedding.shape}"
  puts "✓ EmbeddingModel works on Metal with candle 0.9.1!"
  
rescue => e
  puts "✗ Failed: #{e.message}"
  puts "Error class: #{e.class}"
end

# Test Reranker on Metal
puts "\n--- Testing Reranker on Metal ---"
begin
  device = Candle::Device.metal
  puts "Creating Reranker on Metal..."
  
  reranker = Candle::Reranker.new(
    model_path: "cross-encoder/ms-marco-MiniLM-L-12-v2",
    device: device
  )
  
  puts "Testing reranking..."
  query = "What is machine learning?"
  documents = [
    "Machine learning is a subset of artificial intelligence.",
    "I like pizza.",
    "Neural networks are used in deep learning."
  ]
  
  results = reranker.rerank(query, documents)
  puts "✓ Success! Top result: #{results.first[:text]}"
  puts "✓ Reranker works on Metal with candle 0.9.1!"
  
rescue => e
  puts "✗ Failed: #{e.message}"
  puts "Error class: #{e.class}"
end

puts "\n" + "=" * 60
puts "Summary: Testing Metal support with upgraded Candle version"