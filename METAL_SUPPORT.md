# Metal Support Status for red-candle

## Current Status (Candle 0.9.1)

Metal (Apple GPU) support in red-candle is now complete! All models (EmbeddingModel, Reranker, and LLMs) now work on Metal! 🎉

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
results = reranker.rerank("query", ["doc1", "doc2"], pooling_method: "mean")
results = reranker.rerank("query", ["doc1", "doc2"], pooling_method: "cls")
```

### LLMs (Works on Metal!)
```ruby
device = Candle::Device.metal
llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device: device)
config = Candle::GenerationConfig.balanced(max_length: 50)
response = llm.generate("Hello world", config: config)  # Works on Metal!

# Streaming also works
llm.generate_stream("Once upon a time", config: config) do |token|
  print token
end
```

## Performance Benefits

Using Metal acceleration provides significant performance improvements for all models:
- Faster inference times
- Lower CPU usage
- Better energy efficiency on MacBooks
- Ability to run larger models efficiently

## Checking Metal Support

To check if a model will work on Metal, you can:

1. Try loading and generating with a small prompt
2. If you get "no metal implementation" errors, fall back to CPU
3. Monitor Candle updates for improved Metal support

## Automatic Device Selection

Red-candle includes a simple utility for automatic device selection:

```ruby
require 'candle'

# Get the best available device (Metal > CUDA > CPU)
best_device = Candle::DeviceUtils.best_device
```

## Example Code

- `examples/smart_device_selection.rb` - Automatic device selection
- `examples/test_metal_091.rb` - Testing Metal support
- `examples/test_layer_norm_direct.rb` - Manual layer norm implementation that works on Metal

## Future Updates

The Candle team is actively working on Metal implementations. Once layer_norm and rms_norm get Metal support, all models will benefit from GPU acceleration automatically.