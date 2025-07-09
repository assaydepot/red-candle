#!/usr/bin/env ruby

require 'bundler/setup'
require 'candle'

puts "Simple LLM Example (CPU)"
puts "=" * 60

# Load model on CPU
puts "\nLoading Mistral 7B on CPU..."
llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device: Candle::Device.cpu)
puts "✓ Model loaded successfully"

# Configure generation
config = Candle::GenerationConfig.new(
  temperature: 0.7,
  max_length: 100,
  top_p: 0.9
)

# Generate text
puts "\nGenerating text..."
prompt = "What are the three main benefits of Ruby programming language?"
response = llm.generate(prompt, config: config)

puts "\nPrompt: #{prompt}"
puts "\nResponse: #{response}"

# Clear cache for next generation
llm.clear_cache

# Try streaming
puts "\n" + "-" * 60
puts "Streaming generation:"
prompt2 = "Write a haiku about programming:"

puts "\nPrompt: #{prompt2}\n\nResponse: "
llm.generate_stream(prompt2, config: config) do |token|
  print token
  $stdout.flush
end
puts "\n\n✓ Generation complete!"