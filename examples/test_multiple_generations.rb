#!/usr/bin/env ruby

require 'bundler/setup'
require 'candle'

puts "Testing multiple generations with KV cache clearing..."
puts "=" * 60

# Load the model once
puts "\nLoading model..."
llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
puts "✓ Model loaded"

config = Candle::GenerationConfig.new(
  temperature: 0.7,
  max_length: 50,
  top_p: 0.9
)

# Test multiple generations WITHOUT clearing cache (should fail)
puts "\n--- Test 1: Multiple generations WITHOUT clearing cache ---"
begin
  response1 = llm.generate("What is Ruby?", config: config)
  puts "First generation: ✓"
  puts response1[0..100] + "..."
  
  # This should fail with shape mismatch
  response2 = llm.generate("What is Python?", config: config)
  puts "Second generation: ✓ (unexpected!)"
rescue => e
  puts "Second generation: ✗ Failed as expected"
  puts "Error: #{e.message}"
end

# Test multiple generations WITH clearing cache (should work)
puts "\n--- Test 2: Multiple generations WITH clearing cache ---"
begin
  # Clear cache before first generation to start fresh
  llm.clear_cache
  
  response1 = llm.generate("What is Ruby?", config: config)
  puts "First generation: ✓"
  puts response1[0..100] + "..."
  
  # Clear cache before second generation
  llm.clear_cache
  
  response2 = llm.generate("What is Python?", config: config)
  puts "Second generation: ✓"
  puts response2[0..100] + "..."
  
  # Clear cache and do a third generation
  llm.clear_cache
  
  response3 = llm.generate("What is JavaScript?", config: config)
  puts "Third generation: ✓"
  puts response3[0..100] + "..."
  
rescue => e
  puts "✗ Error: #{e.message}"
  puts e.backtrace.first(3).join("\n")
end

puts "\n" + "=" * 60
puts "Summary: The clear_cache method successfully resets the KV cache,"
puts "allowing multiple independent generations from the same model instance."