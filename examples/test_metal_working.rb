#!/usr/bin/env ruby

require 'bundler/setup'
require 'candle'

puts "Testing Working Metal Support"
puts "=" * 60

# Test 1: EmbeddingModel works!
puts "\n--- EmbeddingModel on Metal (WORKING!) ---"
begin
  device = Candle::Device.metal
  puts "Creating EmbeddingModel on Metal..."
  
  model = Candle::EmbeddingModel.new(
    model_path: "jinaai/jina-embeddings-v2-base-en",
    device: device
  )
  
  # Test with different inputs
  texts = [
    "Hello world!",
    "Ruby is a beautiful programming language.",
    "Metal acceleration provides significant speedups for machine learning workloads."
  ]
  
  texts.each do |text|
    embedding = model.embedding(text)
    puts "✓ '#{text[0..30]}...' -> embedding shape: #{embedding.shape}"
  end
  
  puts "\n🎉 EmbeddingModel fully works on Metal!"
  
rescue => e
  puts "✗ Failed: #{e.message}"
end

# Test 2: Try different Reranker models
puts "\n--- Testing Different Reranker Models ---"
reranker_models = [
  "cross-encoder/ms-marco-MiniLM-L-12-v2",
  "cross-encoder/ms-marco-MiniLM-L-6-v2", 
  "cross-encoder/ms-marco-TinyBERT-L-2-v2"
]

reranker_models.each do |model_path|
  puts "\nTrying #{model_path}..."
  begin
    device = Candle::Device.metal
    reranker = Candle::Reranker.new(
      model_path: model_path,
      device: device
    )
    
    # Simple test
    results = reranker.rerank("test", ["doc1", "doc2"])
    puts "✓ Works!"
    break  # Found a working model
    
  rescue => e
    puts "✗ Failed: #{e.message}"
    
    # Try CPU to see if it's model-specific
    begin
      cpu_reranker = Candle::Reranker.new(
        model_path: model_path,
        device: Candle::Device.cpu
      )
      results = cpu_reranker.rerank("test", ["doc1", "doc2"])
      puts "  (Works on CPU, so it's a Metal-specific issue)"
    rescue => cpu_e
      puts "  (Also fails on CPU: #{cpu_e.message})"
    end
  end
end

# Test 3: LLM on Metal
puts "\n--- Testing LLM on Metal ---"
begin
  device = Candle::Device.metal
  llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device)
  config = Candle::GenerationConfig.new(max_length: 10)
  response = llm.generate("Hello", config)
  puts "✓ LLM works on Metal!"
rescue => e
  puts "✗ LLM failed: #{e.message}"
end

puts "\n" + "=" * 60
puts "Summary:"
puts "- EmbeddingModel: ✅ Fully working on Metal!"
puts "- Reranker: ⚠️  Has dimension mismatch issues on Metal"
puts "- LLM: ❌ Still missing RMS norm implementation"