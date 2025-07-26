#!/usr/bin/env ruby

require 'candle'

device = Candle::Device.cpu

puts "Phi Model Examples"
puts "=================="
puts ""
puts "Note: The first run will download model files, which may take a few minutes."
puts ""

# Example 1: Using a GGUF Phi model (if available)
puts "1. Loading a Phi GGUF model (smaller, faster):"
puts "   You can use models like:"
puts "   - TheBloke/phi-2-GGUF with gguf_file: 'phi-2.Q4_K_M.gguf'"
puts "   - Look for Phi-3 GGUF models on HuggingFace"
puts ""

# Example 2: Phi-2 basic usage
puts "2. Basic Phi-2 generation:"
puts <<-RUBY
llm = Candle::LLM.from_pretrained("microsoft/phi-2", device: device)
prompt = "Write a Python function to calculate the factorial of a number:"
result = llm.generate(prompt, config: Candle::GenerationConfig.new(max_length: 100))
puts result
RUBY

# Example 3: Phi-3 chat
puts "\n3. Phi-3 chat interface:"
puts <<-RUBY
llm = Candle::LLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device: device)
messages = [
  { role: "system", content: "You are a helpful coding assistant." },
  { role: "user", content: "How do I reverse a string in Ruby?" }
]
response = llm.chat(messages, config: Candle::GenerationConfig.balanced)
puts response
RUBY

# Example 4: Streaming
puts "\n4. Streaming generation:"
puts <<-RUBY
llm.generate_stream("The key principles of object-oriented programming are") do |token|
  print token
end
RUBY

puts "\n\nNote: microsoft/phi-2 uses sharded weights (multiple files), so the initial download"
puts "may take longer than single-file models. The model files are cached after first download."