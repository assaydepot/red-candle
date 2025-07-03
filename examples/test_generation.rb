#!/usr/bin/env ruby

require 'bundler/setup'
require 'candle'

puts "Testing Mistral generation..."
puts "=" * 60

begin
  # Note: This assumes you've already downloaded the model
  # If not, it will download ~13GB of data
  puts "\nLoading model (this may take a moment)..."
  llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", nil)
  puts "✓ Model loaded successfully!"
  
  # Test basic generation
  puts "\n--- Basic Generation Test ---"
  config = Candle::GenerationConfig.new(
    temperature: 0.7,
    max_length: 50,
    top_p: 0.9
  )
  
  prompt = "Write a haiku about coding"
  puts "Prompt: #{prompt}"
  print "Response: "
  
  response = llm.generate(prompt, config)
  puts response
  
  # Test streaming generation
  puts "\n--- Streaming Generation Test ---"
  prompt = "Tell me a joke about programming"
  puts "Prompt: #{prompt}"
  print "Response: "
  
  llm.generate_stream(prompt, config) do |token|
    print token
    $stdout.flush
  end
  puts  # New line after streaming
  
  # Test different configurations
  puts "\n--- Deterministic Generation Test ---"
  det_config = Candle::GenerationConfig.deterministic.with(max_length: 30)
  prompt = "The capital of France is"
  puts "Prompt: #{prompt}"
  print "Response: "
  
  response = llm.generate(prompt, det_config)
  puts response
  
rescue => e
  puts "✗ Error: #{e.message}"
  puts e.backtrace.first(5).join("\n")
end

puts "\n" + "=" * 60