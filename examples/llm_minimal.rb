#!/usr/bin/env ruby

require 'bundler/setup'
require 'candle'

puts "=== LLM Configuration Test ==="

# Test configuration creation
config = Candle::GenerationConfig.new(
  temperature: 0.7,
  max_length: 50
)
puts "Created config with temperature: #{config.temperature}, max_length: #{config.max_length}"

# Note: Actual model loading requires downloading large models
puts "\n=== Model Loading ==="
puts "Note: Actual model loading would require downloading a ~13GB model."
puts "Example usage would be:"
puts "  llm = Candle::LLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')"
puts "  response = llm.generate('What is Ruby?')"