# Metal Support Status for red-candle

## Current Status

As of the current version, Metal (Apple GPU) support in red-candle has the following limitations:

### ✅ What Works
- Device creation: `Candle::Device.metal` creates a Metal device successfully
- Model loading: Models can be loaded onto Metal device
- Basic tensor operations (limited)

### ❌ What Doesn't Work Yet
- **RMS Normalization**: Required by Mistral, Llama, and many other modern LLMs
- **Various dtype conversions**: Some tensor type conversions are not implemented
- **Other operations**: Various other operations may not have Metal implementations

## Error Messages

When trying to use Mistral models on Metal, you'll see:
```
Generation failed: Metal error no metal implementation for rms-norm
```

## Workaround

For now, use CPU for LLM inference:

```ruby
# Use CPU instead of Metal
device = Candle::Device.cpu
llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device)

# Generation works normally on CPU
config = Candle::GenerationConfig.new(
  temperature: 0.7,
  max_length: 200,
  top_p: 0.9
)
response = llm.generate("Tell me about Ruby", config)
```

## Future Updates

The Candle project is actively adding Metal implementations for missing operations. Once RMS normalization and other required operations are implemented in candle-metal-kernels, LLMs will work on Metal with significant performance improvements.

## Checking Metal Support

To check if a model will work on Metal, you can:

1. Try loading and generating with a small prompt
2. If you get "no metal implementation" errors, fall back to CPU
3. Monitor Candle updates for improved Metal support

## Example Code

See `examples/llm_simple_cpu.rb` for a working example using CPU.
See `examples/llm_device_fallback.rb` for an example with automatic fallback.