#!/usr/bin/env ruby

require 'bundler/setup'
require 'candle'

# Example 1: Simple text generation with device selection
puts "=== Simple Generation Example ==="
begin
  # Choose device (uncomment one):
  device = Candle::Device.cpu     # Use CPU
  # device = Candle::Device.metal   # Use Apple GPU (Metal)
  # device = Candle::Device.cuda    # Use NVIDIA GPU (CUDA)

  puts "Using device: #{device.inspect}"

  # Load a Mistral model
  llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", device: device)

  # Generate text with default config
  response = llm.generate("What is Ruby programming language?", config: Candle::GenerationConfig.balanced)
  puts "Response: #{response}"

  # Generate with custom configuration
  config = Candle::GenerationConfig.new(
    temperature: 0.7,
    max_length: 100,
    top_p: 0.9
  )
  response = llm.generate("Write a haiku about coding", config: config)
  puts "\nHaiku: #{response}"
rescue => e
  puts "Error loading model: #{e.message}"
  puts "Note: This example requires downloading a large model from HuggingFace."
end

# Example 2: Streaming generation
puts "\n\n=== Streaming Generation Example ==="
begin
  # Use Metal for better performance if available
  device = Candle::Device.metal rescue Candle::Device.cpu
  llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device: device)

  print "Streaming response: "
  llm.generate_stream("Tell me a short story about a robot", config: Candle::GenerationConfig.balanced) do |token|
    print token
    $stdout.flush
  end
  puts
rescue => e
  puts "Streaming error: #{e.message}"
end

# Example 3: Chat interface
puts "\n\n=== Chat Interface Example ==="
begin
  device = Candle::Device.cpu  # Or use Metal/CUDA
  llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device: device)

  messages = [
    { role: "system", content: "You are a helpful Ruby programming assistant." },
    { role: "user", content: "How do I create a hash in Ruby?" }
  ]

  response = llm.chat(messages)
  puts "Chat response: #{response}"
rescue => e
  puts "Chat error: #{e.message}"
end

# Example 4: Different generation configs
puts "\n\n=== Generation Config Examples ==="
begin
  device = Candle::Device.cpu  # Or use Metal/CUDA
  llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device: device)

  # Deterministic (temperature = 0)
  deterministic_config = Candle::GenerationConfig.deterministic
  response = llm.generate("The capital of France is", config: deterministic_config)
  puts "Deterministic: #{response}"

  # Creative (higher temperature)
  creative_config = Candle::GenerationConfig.creative
  response = llm.generate("Once upon a time", config: creative_config)
  puts "Creative: #{response}"

  # Balanced
  balanced_config = Candle::GenerationConfig.balanced
  response = llm.generate("Ruby is", config: balanced_config)
  puts "Balanced: #{response}"
rescue => e
  puts "Config example error: #{e.message}"
end

# Example 5: Model registry
puts "\n\n=== Model Registry Example ==="
puts "Supported models:"
Candle::LLM::ModelRegistry.registered_models.each do |model|
  status = model[:status] || :available
  puts "  - #{model[:name]} (#{model[:size]}) - Status: #{status}"
  puts "    Pattern: #{model[:pattern]}"
  puts "    Context length: #{model[:context_length]}"
  puts "    Supports chat: #{model[:supports_chat]}"
  puts
end