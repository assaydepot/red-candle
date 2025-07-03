#!/usr/bin/env ruby

require 'bundler/setup'
require 'candle'

# Minimal test assuming model is already downloaded
puts "Testing minimal generation..."

begin
  llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", nil)
  
  config = Candle::GenerationConfig.new(
    temperature: 0.7,
    max_length: 20  # Keep it short for testing
  )
  
  response = llm.generate("Hello", config)
  puts "Generated: #{response}"
  
  puts "\nSuccess! The model is generating text."
rescue => e
  puts "Error: #{e.message}"
  puts "Backtrace:"
  puts e.backtrace.first(3).join("\n")
end