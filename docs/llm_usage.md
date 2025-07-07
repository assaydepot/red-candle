# Red Candle LLM Usage Guide

## Overview

Red Candle now supports Large Language Model (LLM) text generation through a unified Ruby interface. This allows you to use various LLMs for text generation, chat completion, and more.

## Basic Usage

### Loading a Model

```ruby
require 'candle'

# Load a model from HuggingFace
llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# Specify a device (optional)
llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device: Candle::Device.cuda)
```

### Simple Text Generation

```ruby
# Generate text with default settings
response = llm.generate("What is Ruby?")
puts response

# Generate with custom configuration
config = Candle::GenerationConfig.new(
  temperature: 0.7,
  max_length: 200,
  top_p: 0.9
)
response = llm.generate("Tell me about Ruby", config)
```

### Streaming Generation

```ruby
# Stream tokens as they're generated
llm.generate_stream("Write a story about") do |token|
  print token
  $stdout.flush
end
```

### Chat Interface

```ruby
# Use the chat interface for conversation
messages = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "What is Ruby?" }
]

response = llm.chat(messages)
puts response

# Stream chat responses
llm.chat_stream(messages) do |token|
  print token
end
```

## Generation Configuration

### Pre-defined Configurations

```ruby
# Deterministic (temperature = 0)
config = Candle::GenerationConfig.deterministic

# Creative (higher temperature, more randomness)
config = Candle::GenerationConfig.creative

# Balanced (moderate settings)
config = Candle::GenerationConfig.balanced

# Default configuration
config = Candle::GenerationConfig.default
```

### Custom Configuration

```ruby
config = Candle::GenerationConfig.new(
  max_length: 512,           # Maximum tokens to generate
  temperature: 0.7,          # Controls randomness (0.0 = deterministic, 1.0+ = more random)
  top_p: 0.9,               # Nucleus sampling threshold
  top_k: 50,                # Top-k sampling
  repetition_penalty: 1.1,   # Penalty for repeating tokens
  seed: 42,                 # Random seed for reproducibility
  stop_sequences: ["\n\n"], # Sequences that stop generation
  include_prompt: false     # Whether to include the prompt in the output
)
```

### Modifying Configurations

```ruby
# Create a new config based on an existing one
creative_config = Candle::GenerationConfig.creative
custom_config = creative_config.with(temperature: 0.8, max_length: 1000)
```

## Model Registry

Check supported models and their capabilities:

```ruby
# Check if a model is supported
if Candle::LLM::ModelRegistry.supported?("mistralai/Mistral-7B-Instruct-v0.2")
  puts "Model is supported!"
end

# Get model information
info = Candle::LLM::ModelRegistry.model_info("mistralai/Mistral-7B-Instruct-v0.2")
puts "Model: #{info[:name]}"
puts "Context length: #{info[:context_length]}"
puts "Supports chat: #{info[:supports_chat]}"

# List all registered models
Candle::LLM::ModelRegistry.registered_models.each do |model|
  puts "#{model[:name]} - #{model[:size]}"
end
```

## Supported Models

Currently supported:
- **Mistral 7B** - Base and Instruct variants
- **Mistral Nemo** - 12B parameter model with 128K context

Planned support:
- **Gemma** - Google's 2B-9B models
- **DeepSeek** - Various sizes

## Performance Tips

1. **Device Selection**: Use GPU when available for faster inference
   ```ruby
   llm = Candle::LLM.from_pretrained(model_id, device: Candle::Device.cuda)
   ```

2. **Batch Processing**: Process multiple prompts together when possible

3. **Configuration Tuning**:
   - Lower temperature for more consistent outputs
   - Adjust max_length based on your needs
   - Use repetition_penalty to avoid repetitive text

4. **Memory Management**: Models are cached, so loading the same model multiple times is efficient

## Error Handling

```ruby
begin
  llm = Candle::LLM.from_pretrained("some-model")
  response = llm.generate("Hello")
rescue => e
  puts "Error: #{e.message}"
  # Handle model loading errors, generation errors, etc.
end
```

## Advanced Features (Coming Soon)

- **Tool Calling**: Allow models to call Ruby methods
- **Structured Generation**: Generate JSON/structured data with schemas
- **Fine-tuning**: Adapt models to specific tasks
- **Quantization**: Use quantized models for reduced memory usage

## Example: Building a Simple Chatbot

```ruby
require 'candle'

class SimpleChatbot
  def initialize(model_id)
    @llm = Candle::LLM.from_pretrained(model_id)
    @messages = []
  end

  def chat(user_input)
    @messages << { role: "user", content: user_input }
    
    response = @llm.chat(@messages)
    @messages << { role: "assistant", content: response }
    
    response
  end

  def reset
    @messages = []
  end
end

# Usage
bot = SimpleChatbot.new("mistralai/Mistral-7B-Instruct-v0.2")
puts bot.chat("Hello! What's your name?")
puts bot.chat("Can you help me with Ruby programming?")
```