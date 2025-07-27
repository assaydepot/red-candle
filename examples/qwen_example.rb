require "candle"

# Example of using Qwen models with Red Candle

# Device selection (use Metal on macOS, CUDA on Linux with GPU, or CPU)
device = begin
           Candle::Device.metal
         rescue
           begin
             Candle::Device.cuda
           rescue
             Candle::Device.cpu
           end
         end

puts "Using device: #{device.inspect}"

# Generation configuration
config = Candle::GenerationConfig.balanced(
  max_length: 100,
  temperature: 0.7,
  seed: 42
)

# Example 1: Using Qwen GGUF model (quantized)
puts "\n=== Qwen GGUF Model Example ==="
begin
  # This would download the model from HuggingFace
  # llm = Candle::LLM.from_pretrained("Qwen/Qwen3-4B-GGUF", 
  #                                   device: device,
  #                                   gguf_file: "qwen3-4b-q4_k_m.gguf")
  
  # For demo purposes, let's show the expected usage:
  puts "To use Qwen GGUF models:"
  puts "llm = Candle::LLM.from_pretrained('Qwen/Qwen3-4B-GGUF', device: device, gguf_file: 'qwen3-4b-q4_k_m.gguf')"
  puts "response = llm.generate('What is Ruby?', config: config)"
rescue => e
  puts "Note: Model download would be required: #{e.message}"
end

# Example 2: Using Qwen with chat interface
puts "\n=== Qwen Chat Interface Example ==="
puts "Chat interface usage:"
puts <<-RUBY
messages = [
  { role: "system", content: "You are a helpful AI assistant." },
  { role: "user", content: "Explain Ruby in one sentence." }
]

# Using GGUF model
# llm = Candle::LLM.from_pretrained("Qwen/Qwen3-8B-GGUF", 
#                                   device: device,
#                                   gguf_file: "qwen3-8b-q4_k_m.gguf")

# Using non-quantized model
# llm = Candle::LLM.from_pretrained("Qwen/Qwen3-1.8B", device: device)

# Generate response
# response = llm.chat(messages, config: config)
# puts response

# Or stream the response
# llm.chat_stream(messages, config: config) do |token|
#   print token
# end
RUBY

# Example 3: Tokenizer registry
puts "\n=== Tokenizer Auto-Detection ==="
test_models = [
  "Qwen/Qwen3-8B-GGUF",
  "bartowski/Qwen3-14B-GGUF",
  "someone/qwen-3-4b-gguf"
]

test_models.each do |model|
  tokenizer = Candle::LLM.guess_tokenizer(model)
  puts "Model: #{model}"
  puts "  -> Auto-detected tokenizer: #{tokenizer}"
end

# Example 4: Custom tokenizer registration
puts "\n=== Custom Tokenizer Registration ==="
# Register a custom Qwen model variant
Candle::LLM.register_tokenizer("MyOrg/CustomQwen3-GGUF", "Qwen/Qwen3-8B")
puts "Registered custom tokenizer mapping"
puts "  MyOrg/CustomQwen3-GGUF -> #{Candle::LLM.guess_tokenizer('MyOrg/CustomQwen3-GGUF')}"

puts "\n=== Available Qwen Models ==="
puts "Quantized (GGUF) models - Recommended (llama.cpp compatible):"
puts "  - Qwen/Qwen2-7B-Instruct-GGUF (use files: qwen2-7b-instruct-q4_k_m.gguf, etc.)"
puts "  - Qwen/Qwen2.5-7B-Instruct-GGUF"
puts "  - Qwen/Qwen2.5-32B-Instruct-GGUF"
puts "  - Qwen/Qwen2.5-Coder-32B-Instruct-GGUF"
puts "\nQuantized (GGUF) models - Qwen3 (may have compatibility issues):"
puts "  - Qwen/Qwen3-4B-GGUF"
puts "  - Qwen/Qwen3-8B-GGUF"
puts "  - Qwen/Qwen3-32B-GGUF"
puts "\nNon-quantized models (safetensors):"
puts "  - Qwen/Qwen2-1.5B"
puts "  - Qwen/Qwen2-7B"
puts "  - Qwen/Qwen2.5-0.5B"
puts "  - Qwen/Qwen2.5-7B"
puts "\nNote: Some Qwen3 GGUF files use a newer format not yet supported."
puts "      Try Qwen2 or Qwen2.5 GGUF models for better compatibility."