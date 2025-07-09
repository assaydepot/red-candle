# Device Support in red-candle

## Overview

red-candle supports multiple device types for running models:
- **CPU**: Default, works everywhere
- **Metal**: Apple GPU acceleration (Apple Silicon Macs)
- **CUDA**: NVIDIA GPU acceleration (requires explicit enablement, see CUDA_SUPPORT.md)

## Checking Device Availability

The recommended way to check device availability is using the BuildInfo methods:

```ruby
# Check what devices were compiled in
if Candle::BuildInfo.metal_available?
  puts "Metal support is available"
end

if Candle::BuildInfo.cuda_available?
  puts "CUDA support is available"
end

# List all available devices
devices = Candle::Device.available_devices
puts "Available devices: #{devices}"
# => ["cpu", "metal"]  # Example output on Apple Silicon Mac
```

## Device Creation

```ruby
# CPU is always available
cpu_device = Candle::Device.cpu

# Create Metal device (if available)
if Candle::BuildInfo.metal_available?
  metal_device = Candle::Device.metal
end

# Create CUDA device (if available)
if Candle::BuildInfo.cuda_available?
  cuda_device = Candle::Device.cuda
end
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
llm = Candle::LLM.from_pretrained("model-name", device: Candle::Device.cpu)

# Metal
llm = Candle::LLM.from_pretrained("model-name", device: Candle::Device.metal)

# CUDA (if available)
llm = Candle::LLM.from_pretrained("model-name", device: Candle::Device.cuda)
```

## Automatic Device Selection

Use the DeviceUtils helper for automatic device selection:

```ruby
# Get the best available device (Metal > CUDA > CPU)
device = Candle::DeviceUtils.best_device

# Create a model with automatic device selection
model = Candle::DeviceUtils.create_with_best_device(
  Candle::EmbeddingModel,
  model_type: :snowflake_arctic_embed_m_v15
)
```

## Edge Cases: Runtime Device Availability

In rare cases, runtime availability might differ from build-time configuration:
- Binary gems installed on different systems
- CUDA driver issues or removal
- Running in containers without GPU passthrough

For these edge cases, you can use defensive programming:

```ruby
begin
  device = Candle::Device.cuda
  # Successfully created CUDA device
rescue => e
  puts "CUDA runtime error: #{e.message}"
  # Fall back to CPU
  device = Candle::Device.cpu
end
```

## Performance Considerations

- **Metal**: Best performance on Apple Silicon Macs
- **CUDA**: Best performance on NVIDIA GPUs (when enabled)
- **CPU**: Works everywhere but slower for large models

## See Also

- [CUDA_SUPPORT.md](CUDA_SUPPORT.md) - How to enable CUDA support
- [examples/check_devices.rb](examples/check_devices.rb) - Example script to check device configuration