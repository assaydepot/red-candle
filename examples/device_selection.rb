#!/usr/bin/env ruby

require 'bundler/setup'
require 'candle'

puts "=== Device Selection for LLMs ==="
puts "=" * 60

# Helper method to detect available devices
def detect_devices
  devices = { cpu: true, metal: false, cuda: false }
  
  # Check for Metal (Apple GPU)
  begin
    _device = Candle::Device.metal
    devices[:metal] = true
    puts "✓ Metal (Apple GPU) is available"
  rescue => e
    puts "✗ Metal not available: #{e.message}"
  end
  
  # Check for CUDA (NVIDIA GPU)
  begin
    _device = Candle::Device.cuda
    devices[:cuda] = true
    puts "✓ CUDA (NVIDIA GPU) is available"
  rescue => e
    puts "✗ CUDA not available: #{e.message}"
  end
  
  puts "✓ CPU is always available"
  
  devices
end

# Helper method to select best available device
def select_best_device
  devices = detect_devices
  
  if devices[:metal]
    puts "\n→ Selected Metal (Apple GPU) for best performance"
    Candle::Device.metal
  elsif devices[:cuda]
    puts "\n→ Selected CUDA (NVIDIA GPU) for best performance"
    Candle::Device.cuda
  else
    puts "\n→ Selected CPU (no GPU acceleration available)"
    Candle::Device.cpu
  end
end

# Detect available devices
puts "Detecting available devices..."
available_devices = detect_devices

# Select the best available device
puts "\nSelecting best device..."
device = select_best_device

# Load model on selected device
puts "\nLoading model on #{device.inspect}..."
begin
  llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device)
  puts "✓ Model loaded successfully!"
  puts "Model is running on: #{llm.device.inspect}"
  
  # Quick performance test
  config = Candle::GenerationConfig.new(
    temperature: 0.7,
    max_length: 50
  )
  
  puts "\n--- Performance Test ---"
  prompts = [
    "What is machine learning?",
    "Explain quantum computing",
    "How does a CPU work?"
  ]
  
  prompts.each do |prompt|
    llm.clear_cache
    
    start_time = Time.now
    response = llm.generate(prompt, config)
    elapsed = Time.now - start_time
    
    puts "\nPrompt: #{prompt}"
    puts "Response: #{response[0..100]}..."
    puts "Time: #{elapsed.round(3)} seconds"
  end
  
rescue => e
  puts "✗ Error: #{e.message}"
end

# Device-specific optimizations
puts "\n" + "=" * 60
puts "Device-Specific Tips:"

if available_devices[:metal]
  puts "\nMetal (Apple Silicon) Optimizations:"
  puts "- Unified memory architecture means no CPU↔GPU transfer overhead"
  puts "- Best performance on M1/M2/M3 chips"
  puts "- Automatically uses Apple's ML accelerators"
  puts "- Power efficient for laptop use"
end

if available_devices[:cuda]
  puts "\nCUDA (NVIDIA) Optimizations:"
  puts "- Ensure you have sufficient VRAM for the model"
  puts "- Larger batch sizes can improve throughput"
  puts "- Consider using half-precision (FP16) for larger models"
end

puts "\nCPU Optimizations:"
puts "- Uses all available cores automatically"
puts "- Consider quantized models for better CPU performance"
puts "- May use AVX/AVX2 instructions if available"

# Example: Explicitly choosing a device
puts "\n--- Manual Device Selection ---"
puts <<~RUBY
  # Always use CPU
  cpu_llm = Candle::LLM.from_pretrained("model_name", Candle::Device.cpu)
  
  # Try Metal, fallback to CPU
  device = begin
    Candle::Device.metal
  rescue
    Candle::Device.cpu
  end
  llm = Candle::LLM.from_pretrained("model_name", device)
  
  # Check device of loaded model
  puts llm.device  # => #<Candle::Device:Metal> or #<Candle::Device:Cpu>
RUBY