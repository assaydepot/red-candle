#!/usr/bin/env ruby

require_relative '../lib/candle'

# Example demonstrating Llama model usage
# Note: This will download the model on first run

puts "Loading Llama model..."
device = begin
  Candle::Device.metal
rescue
  puts "Metal not available, using CPU"
  Candle::Device.cpu
end

# Load a small Llama model (you can change this to any Llama model on HuggingFace)
# Example models:
# - "meta-llama/Llama-2-7b-hf" (requires HF token)
# - "NousResearch/Llama-2-7b-hf" (no token required)
# - "meta-llama/Llama-3-8B" (requires HF token)

model_id = "NousResearch/Llama-2-7b-hf" # Using a community model that doesn't require auth
llm = Candle::LLM.from_pretrained(model_id, device: device)

puts "Model loaded: #{llm.model_name}"
puts "Device: #{llm.device}"

# Test basic generation
puts "\n=== Basic Generation ==="
prompt = "The capital of France is"
response = llm.generate(prompt, config: Candle::GenerationConfig.balanced(max_length: 50))
puts "Prompt: #{prompt}"
puts "Response: #{response}"

# Test streaming generation
puts "\n=== Streaming Generation ==="
prompt = "List three interesting facts about Ruby programming language:"
print "Prompt: #{prompt}\n"
print "Response: "
llm.generate_stream(prompt, config: Candle::GenerationConfig.balanced(max_length: 100)) do |token|
  print token
  $stdout.flush
end
puts "\n"

# Test chat interface (for instruction-tuned models)
if model_id.include?("instruct") || model_id.include?("Instruct") || model_id.include?("chat")
  puts "\n=== Chat Interface ==="
  messages = [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "What is Ruby programming language?" }
  ]
  
  response = llm.chat(messages, config: Candle::GenerationConfig.balanced(max_length: 100))
  puts "Chat response: #{response}"
end

# Test different generation configs
puts "\n=== Generation Configs ==="
prompt = "Ruby is"

puts "\nDeterministic (temperature=0):"
response = llm.generate(prompt, config: Candle::GenerationConfig.deterministic(max_length: 30))
puts response

puts "\nCreative (temperature=1.0):"
response = llm.generate(prompt, config: Candle::GenerationConfig.creative(max_length: 30))
puts response

puts "\nDone!"