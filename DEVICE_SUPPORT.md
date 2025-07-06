# Device Support in red-candle

## Overview

red-candle supports multiple device types for running models:
- **CPU**: Default, works everywhere
- **Metal**: Apple GPU acceleration (Apple Silicon Macs)
- **CUDA**: NVIDIA GPU acceleration (Linux/Windows)

## Device Creation

```ruby
# Create devices
cpu_device = Candle::Device.cpu
metal_device = Candle::Device.metal
cuda_device = Candle::Device.cuda
```

## Model Support by Device

### EmbeddingModel

```ruby
# CPU (always works)
model = Candle::EmbeddingModel.new(device: Candle::Device.cpu)

# Metal
model = Candle::EmbeddingModel.new(device: Candle::Device.metal)

# CUDA (if available)
model = Candle::EmbeddingModel.new(device: Candle::Device.cuda)
```

### Reranker

```ruby
# CPU (always works)
reranker = Candle::Reranker.new(device: Candle::Device.cpu)

# Metal
reranker = Candle::Reranker.new(device: Candle::Device.metal)

# CUDA (if available)
reranker = Candle::Reranker.new(device: Candle::Device.cuda)
```

### LLM

```ruby
# CPU (always works)
llm = Candle::LLM.from_pretrained("model-name", Candle::Device.cpu)

# Metal
llm = Candle::LLM.from_pretrained("model-name", Candle::Device.metal)

# CUDA (if available)
llm = Candle::LLM.from_pretrained("model-name", Candle::Device.cuda)
```

## Checking Device Availability

```ruby
begin
  device = Candle::Device.metal
  puts "Metal is available"
rescue => e
  puts "Metal not available: #{e.message}"
end

begin
  device = Candle::Device.cuda
  puts "CUDA is available"
rescue => e
  puts "CUDA not available: #{e.message}"
end
```

## Future Improvements

CUDA support.