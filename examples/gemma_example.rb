#!/usr/bin/env ruby

require_relative '../lib/candle'

# Example demonstrating Gemma model usage
# Note: This will download the model on first run

puts "Loading Gemma model..."
device = begin
  Candle::Device.metal
rescue
  puts "Metal not available, using CPU"
  Candle::Device.cpu
end

# Load a Gemma model - using the 2B model for faster loading
# Other options:
# - "google/gemma-2b" (2 billion parameters)
# - "google/gemma-7b" (7 billion parameters)
# - "google/gemma-2b-it" (instruction-tuned 2B)
# - "google/gemma-7b-it" (instruction-tuned 7B)

model_id = "google/gemma-2b-it" # Using instruction-tuned model for better chat
llm = Candle::LLM.from_pretrained(model_id, device: device)

puts "Model loaded: #{llm.model_name}"
puts "Device: #{llm.device}"

# Test basic generation
puts "\n=== Basic Generation ==="
prompt = "The Ruby programming language is"
response = llm.generate(prompt, config: Candle::GenerationConfig.balanced(max_length: 50))
puts "Prompt: #{prompt}"
puts "Response: #{response}"

# Test streaming generation
puts "\n=== Streaming Generation ==="
prompt = "Write a haiku about Ruby programming:"
print "Prompt: #{prompt}\n"
print "Response: "
llm.generate_stream(prompt, config: Candle::GenerationConfig.balanced(max_length: 100)) do |token|
  print token
  $stdout.flush
end
puts "\n"

# Test chat interface (Gemma-IT models support chat)
puts "\n=== Chat Interface ==="
messages = [
  { role: "user", content: "What makes Ruby a good programming language?" }
]

response = llm.chat(messages, config: Candle::GenerationConfig.balanced(max_length: 150))
puts "Chat response: #{response}"

# Test multi-turn conversation
puts "\n=== Multi-turn Conversation ==="
messages = [
  { role: "user", content: "Hi! Can you help me learn Ruby?" },
  { role: "assistant", content: "Of course! I'd be happy to help you learn Ruby. Ruby is a dynamic, object-oriented programming language known for its simplicity and productivity. What specific aspect would you like to start with?" },
  { role: "user", content: "How do I define a method in Ruby?" }
]

puts "Conversation:"
messages.each do |msg|
  puts "#{msg[:role].capitalize}: #{msg[:content]}"
end

print "\nAssistant: "
llm.chat_stream(messages, config: Candle::GenerationConfig.balanced(max_length: 200)) do |token|
  print token
  $stdout.flush
end
puts "\n"

# Test different generation configs
puts "\n=== Generation Configs ==="
prompt = "Ruby programming"

puts "\nDeterministic (temperature=0):"
response = llm.generate(prompt, config: Candle::GenerationConfig.deterministic(max_length: 30))
puts response

puts "\nCreative (temperature=1.0):"
response = llm.generate(prompt, config: Candle::GenerationConfig.creative(max_length: 30))
puts response

# Show Gemma-specific chat formatting
puts "\n=== Gemma Chat Template Example ==="
messages = [
  { role: "user", content: "Explain Ruby blocks in one sentence." }
]

# The chat method automatically applies the correct template
# For Gemma it uses: <start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n
formatted_prompt = llm.apply_chat_template(messages)
puts "Formatted prompt for Gemma:"
puts formatted_prompt.inspect

puts "\nDone!"