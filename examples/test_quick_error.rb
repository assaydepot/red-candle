#!/usr/bin/env ruby

require 'bundler/setup'
require 'candle'

puts "Testing model that doesn't exist to see error handling..."

begin
  # Try a model that doesn't exist to see the error quickly
  llm = Candle::LLM.from_pretrained("fake-model/does-not-exist", nil)
rescue => e
  puts "Error: #{e.message}"
end

puts "\nTesting unsupported model type..."
begin
  # Try a non-Mistral model to see the unsupported error
  llm = Candle::LLM.from_pretrained("bert-base-uncased", nil)
rescue => e
  puts "Error: #{e.message}"
end