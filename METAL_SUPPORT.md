# Metal Support Status for red-candle

## Current Status (Candle 0.9.1)

Metal (Apple GPU) support in red-candle continues to improve! Both EmbeddingModel and Reranker now work on Metal.

### ✅ What Works
- **EmbeddingModel**: Fully functional on Metal! 🎉
  - Layer normalization is now working with candle 0.9.1
  - Significant performance improvements over CPU
- **Reranker**: Now works on Metal! 🎉
  - Fixed tensor indexing issues with a workaround
  - All pooling methods (pooler, cls, mean) are supported
- Basic tensor operations: mean, sqrt, broadcast operations, etc.
- Device creation and tensor manipulation

### ❌ What Doesn't Work Yet
- **LLMs (Mistral, Llama, etc.)**: Missing RMS normalization for Metal
- Various dtype conversions and specialized operations

## Error Messages

When trying to use LLMs (Mistral/Llama) on Metal:
```
Generation failed: Metal error no metal implementation for rms-norm
```

## How to Enable Metal Support

To use Metal acceleration, ensure all candle crates have the `metal` feature enabled in your `Cargo.toml`:

```toml
[dependencies]
candle-core = { version = "0.9.1", features = ["metal"] }
candle-nn = { version = "0.9.1", features = ["metal"] }
candle-transformers = { version = "0.9.1", features = ["metal"] }
```

## Usage Examples

### EmbeddingModel (Works on Metal!)
```ruby
device = Candle::Device.metal
model = Candle::EmbeddingModel.new(
  model_path: "jinaai/jina-embeddings-v2-base-en",
  device: device
)
embedding = model.embedding("Hello world!")  # Works great!
```

### Reranker (Works on Metal!)
```ruby
device = Candle::Device.metal
reranker = Candle::Reranker.new(device: device)
results = reranker.rerank("query", ["doc1", "doc2"])  # Works on Metal!

# All pooling methods work
results = reranker.rerank_with_pooling("query", ["doc1", "doc2"], "mean")
results = reranker.rerank_with_pooling("query", ["doc1", "doc2"], "cls")
```

### LLMs (Use CPU for now)
```ruby
# Use CPU due to missing RMS norm
device = Candle::Device.cpu
llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device)
```

## Future Updates

The Candle project is actively adding Metal implementations for missing operations. Once RMS normalization and other required operations are implemented in candle-metal-kernels, LLMs will work on Metal with significant performance improvements.

## Checking Metal Support

To check if a model will work on Metal, you can:

1. Try loading and generating with a small prompt
2. If you get "no metal implementation" errors, fall back to CPU
3. Monitor Candle updates for improved Metal support

## Smart Device Selection

Red-candle now includes utilities for automatic device selection based on model compatibility:

```ruby
require 'candle'

# Automatically selects the best device for your model
model = Candle::DeviceUtils.create_with_best_device(
  Candle::EmbeddingModel,
  :embedding_model,
  model_path: "jinaai/jina-embeddings-v2-base-en",
  device: Candle::Device.metal  # Will fallback to CPU automatically
)

# Check device support
Candle::DeviceUtils.supports_model?(Candle::Device.metal, :embedding_model)  # => true
Candle::DeviceUtils.supports_model?(Candle::Device.metal, :reranker)         # => true
Candle::DeviceUtils.supports_model?(Candle::Device.metal, :llm)              # => false
```

## Example Code

- `examples/smart_device_selection.rb` - Automatic device selection
- `examples/test_metal_091.rb` - Testing Metal support
- `examples/test_layer_norm_direct.rb` - Manual layer norm implementation that works on Metal

## Future Updates

The Candle team is actively working on Metal implementations. Once layer_norm and rms_norm get Metal support, all models will benefit from GPU acceleration automatically.