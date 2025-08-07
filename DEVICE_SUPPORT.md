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
model = Candle::EmbeddingModel.from_pretrained(device: Candle::Device.cpu)

# Metal
model = Candle::EmbeddingModel.from_pretrained(device: Candle::Device.metal)

# CUDA (if available)
model = Candle::EmbeddingModel.from_pretrained(device: Candle::Device.cuda)
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

## LLM Performance

### CUDA

We see a nearly a 18x speedup when running under CUDA vs CPU on the same machine.

Tests were run on an NVIDIA A10G GPU with 24GB of VRAM and 64GB of system ram.

#### NVIDIA-SMI
```
> nvidia-smi
Tue Jul 15 18:38:16 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.158.01             Driver Version: 570.158.01     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A10G                    On  |   00000000:00:1E.0 Off |                    0 |
|  0%   30C    P0             58W /  300W |       0MiB /  23028MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

#### CPU
```
$ bundle exec irb
> require 'candle'
> require 'benchmark'
> cpu_llm = Candle::LLM.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", gguf_file: "tinyllama-1.1b-chat-v1.0.Q3_K_L.gguf", device: Candle::Device.cpu)
* puts Benchmark.measure {
*   cpu_llm.generate_stream("Tell me about running Large Language Models in Ruby. Answer: ", config: Candle::GenerationConfig.deterministic(max_length: 50)) { |t| print t };nil
*   puts
*   puts
> }
> 

- Large Language Models (LLMs) are a type of pre-trained language model that can be used for natural language processing tasks such as text

 36.319670  12.974286  49.293956 (  4.802834)
=> nil
```

#### CUDA
```
$ bundle exec irb
> require 'candle'
> require 'benchmark'
> cuda_llm = Candle::LLM.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", gguf_file: "tinyllama-1.1b-chat-v1.0.Q3_K_L.gguf", device: Candle::Device.cuda)
* puts Benchmark.measure {
*   cuda_llm.generate_stream("Tell me about running Large Language Models in Ruby. Answer: ", config: Candle::GenerationConfig.deterministic(max_length: 50)) { |t| print t }
*   puts
*   puts
> }
> 

- Large Language Models (LLMs) are a type of machine learning model that can process large amounts of text data. They are typically used for tasks

  0.243573   0.020900   0.264473 (  0.264503)
=> nil
```

### METAL

We see a >3x speedup when running under Metal vs CPU on the same machine.

Tests were run on an Apple M4 Max with 64GB of RAM and 10-core CPU.

#### METAL

```
$ bundle exec irb
> require 'candle'
> require 'benchmark'
> cuda_llm = Candle::LLM.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", gguf_file: "tinyllama-1.1b-chat-v1.0.Q3_K_L.gguf", device: Candle::Device.metal)
> puts Benchmark.measure {
*   cuda_llm.generate_stream("Tell me about running Large Language Models in Ruby. Answer: ", config: Candle::GenerationConfig.deterministic(max_length: 50)) { |t| print t }
*   puts
*   puts
> }

- Large Language Models (LLMs) are a type of pre-trained language model that has been trained on large amounts of text data. They

  0.025938   0.047080   0.073018 (  0.221948)
=> nil
```

#### CPU

```
$ bundle exec irb
> require 'candle'
> require 'benchmark'
> cpu_llm = Candle::LLM.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", gguf_file: "tinyllama-1.1b-chat-v1.0.Q3_K_L.gguf", device: Candle::Device.cpu)
> puts Benchmark.measure {
*   cpu_llm.generate_stream("Tell me about running Large Language Models in Ruby. Answer: ", config: Candle::GenerationConfig.deterministic(max_length: 50)) { |t| print t };nil
*   puts
*   puts
> }

- Large Language Models (LLMs) are a type of pre-trained language model that has been trained on a large corpus of text data

  3.669953   7.103764  10.773717 (  0.827887)
=> nil
```

## See Also

- [CUDA_SUPPORT.md](CUDA_SUPPORT.md) - How to enable CUDA support
- [examples/check_devices.rb](examples/check_devices.rb) - Example script to check device configuration