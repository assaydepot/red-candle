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

# Metal (limited by operation support)
model = Candle::EmbeddingModel.new(device: Candle::Device.metal)
# Note: May fail with "no metal implementation for layer-norm" for some models

# CUDA (if available)
model = Candle::EmbeddingModel.new(device: Candle::Device.cuda)
```

### Reranker

```ruby
# CPU (always works)
reranker = Candle::Reranker.new(device: Candle::Device.cpu)

# Metal (limited by operation support)
reranker = Candle::Reranker.new(device: Candle::Device.metal)
# Note: May fail with "no metal implementation for layer-norm"

# CUDA (if available)
reranker = Candle::Reranker.new(device: Candle::Device.cuda)

# Legacy CUDA parameter (deprecated, use device instead)
reranker = Candle::Reranker.new(cuda: true)
```

### LLM

```ruby
# CPU (always works)
llm = Candle::LLM.from_pretrained("model-name", Candle::Device.cpu)

# Metal (very limited - missing RMS norm for most models)
llm = Candle::LLM.from_pretrained("model-name", Candle::Device.metal)
# Note: Mistral models fail with "no metal implementation for rms-norm"

# CUDA (if available)
llm = Candle::LLM.from_pretrained("model-name", Candle::Device.cuda)
```

## Metal Limitations

The Metal backend in Candle is still under development. Common missing operations:
- **Layer Normalization**: Required by BERT-based models (EmbeddingModel, Reranker)
- **RMS Normalization**: Required by modern LLMs (Mistral, Llama, etc.)
- Various dtype conversions

## Recommendations

1. **For production**: Use CPU for maximum compatibility
2. **For performance**: Use CUDA if available on Linux/Windows
3. **For Mac users**: Metal support is experimental, expect failures

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

## Example: Device Fallback Pattern

```ruby
def create_model_with_best_device(model_path)
  # Try devices in order of preference
  devices_to_try = [
    Candle::Device.cuda,   # Fastest if available
    Candle::Device.metal,  # Mac GPU
    Candle::Device.cpu     # Always works
  ]
  
  devices_to_try.each do |device|
    begin
      model = Candle::EmbeddingModel.new(
        model_path: model_path,
        device: device
      )
      # Test if it actually works
      model.embedding("test")
      puts "Using device: #{device.inspect}"
      return model
    rescue => e
      puts "Device #{device.inspect} failed: #{e.message}"
      next
    end
  end
  
  raise "No suitable device found"
end
```

## Future Improvements

As Candle development progresses, more operations will be implemented for Metal, improving GPU support on macOS. Check the [Candle repository](https://github.com/huggingface/candle) for updates.