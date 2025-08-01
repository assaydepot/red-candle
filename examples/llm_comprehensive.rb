require "candle"
device = Candle::Device.metal
config = Candle::GenerationConfig.balanced(debug_tokens: false, max_length: 25)
puts "Running with seed: #{config.seed}"

models = [
  {
    model: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    options: [
      { gguf_file: "mistral-7b-instruct-v0.2.Q4_K_M.gguf" }
      # { gguf_file: "mistral-7b-instruct-v0.2.Q5_K_M.gguf" },
      # { gguf_file: "mistral-7b-instruct-v0.2.Q2_K.gguf" }
    ]
  },
  # {
  #   model: "TheBloke/Llama-2-7B-Chat-GGUF",
  #   gguf_files: [
  #     "llama-2-7b-chat.Q4_K_M.gguf",
  #     # "llama-2-7b-chat.Q8_0.gguf"
  #   ]
  # },
  {
    model: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    options: [
      # { gguf_file: "tinyllama-1.1b-chat-v1.0.Q2_K.gguf" },
      { gguf_file: "tinyllama-1.1b-chat-v1.0.Q3_K_L.gguf" },
      { gguf_file: "tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf" },
      { gguf_file: "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf" },
      { gguf_file: "tinyllama-1.1b-chat-v1.0.Q4_0.gguf" },
      # { gguf_file:"tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" },
      # { gguf_file:"tinyllama-1.1b-chat-v1.0.Q4_K_S.gguf" },
      { gguf_file:"tinyllama-1.1b-chat-v1.0.Q5_0.gguf" },
      # { gguf_file:"tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf" },
      # { gguf_file:"tinyllama-1.1b-chat-v1.0.Q5_K_S.gguf" },
      # { gguf_file:"tinyllama-1.1b-chat-v1.0.Q6_K.gguf" },
    ]
  },
  {
    model: "google/gemma-3-4b-it-qat-q4_0-gguf",
    options: [
      { gguf_file: "gemma-3-4b-it-q4_0.gguf", tokenizer: "google/gemma-3-4b-it" }
    ]
  },
  {
    model: "Qwen/Qwen3-4B-GGUF",
    options: [
      { gguf_file: "qwen3-4b-q4_k_m.gguf" },
      { gguf_file: "qwen3-4b-q5_k_m.gguf" }
    ]
  }
]
puts "QUANTIZED MODELS"
models.each do |entry|
  model = entry[:model]
  options = entry[:options]
  options.each do |option|
    puts "=== Loading #{model} - #{option} ==="
    llm = Candle::LLM.from_pretrained(model, device: device, **option)
    puts "\n=== Basic Generation ==="
    puts llm.generate("What is Ruby?", config: config)
    puts "\n=== Streaming Generation ==="
    llm.generate_stream("What is Ruby?", config: config) { |t| print t }
    puts "\n"
  end
end

puts "NON QUANTIZED MODELS"
models = [
  "mistralai/Mistral-7B-Instruct-v0.1",
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "google/gemma-2b-it"
]
models.each do |model|
  puts "=== Loading #{model} ==="
  llm = Candle::LLM.from_pretrained(model, device: device)
  puts "\n=== Basic Generation ==="
  puts llm.generate("What is Ruby?", config: config)
  puts "\n=== Streaming Generation ==="
  llm.generate_stream("What is Ruby?", config: config) { |t| print t }
  puts "\n"
end
