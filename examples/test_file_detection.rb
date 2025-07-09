#!/usr/bin/env ruby

require 'bundler/setup'
require 'candle'

puts "Testing Mistral file detection (minimal test)"
puts "=" * 60

# This should trigger the file detection logic and fail with a clear error
# about which files it tried to find
begin
  puts "\nAttempting to load mistralai/Mistral-7B-Instruct-v0.1..."
  llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
  puts "Success! (This is unexpected for a 13GB model)"
rescue => e
  puts "Error (expected): #{e.message}"
  
  # Check if we're getting the right kind of error
  if e.message.include?("model-") && e.message.include?("safetensors")
    puts "\n✓ File detection is working correctly!"
    puts "  The error shows it's trying to load sharded safetensors files."
  elsif e.message.include?("Failed to download")
    puts "\n✓ File detection found the files!"
    puts "  Download failed (expected for large models)."
  else
    puts "\n✗ Unexpected error type"
  end
end

puts "\n" + "=" * 60