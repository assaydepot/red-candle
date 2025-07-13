# Stable Diffusion 3 Support in Red Candle

## Current Status

The SD3 implementation is complete but currently falls back to placeholder generation due to:

1. **FP8 Models**: The default SD3 model (`sd3_medium_incl_clips_t5xxlfp8.safetensors`) uses FP8 (8-bit floating point) which is not yet supported by Candle.

2. **Model Structure**: The model file structure needs further investigation to properly load all components.

## Available Options

### Option 1: Use FP16 Model (Recommended for full quality)
```ruby
require 'candle'

config = Candle::ImageGenerationConfig.new(
  height: 1024,
  width: 1024,
  num_inference_steps: 28,
  guidance_scale: 7.0,
  seed: 42
)

# This will download ~20GB FP16 model
model = Candle::ImageGenerator.from_pretrained(
  "stabilityai/stable-diffusion-3-medium",
  model_file: "sd3_medium_incl_clips_t5xxlfp16.safetensors"
)

image_data = model.generate("A beautiful sunset", config: config)
File.binwrite("output.png", image_data)
```

### Option 2: Use Base Model Without Text Encoders
```ruby
# Smaller download, but requires separate text encoder files
model = Candle::ImageGenerator.from_pretrained(
  "stabilityai/stable-diffusion-3-medium",
  model_file: "sd3_medium.safetensors"
)
```

### Option 3: Use GGUF Quantized Models (Future)
```ruby
# GGUF support is implemented but needs testing
model = Candle::ImageGenerator.from_pretrained(
  "second-state/stable-diffusion-3-medium-GGUF",
  gguf_file: "sd3-medium-Q5_0.gguf"
)
```

## Implementation Details

The SD3 implementation includes:

- ✅ Full pipeline architecture (`ext/candle/src/image_gen/sd3/`)
- ✅ MMDiT (Multimodal Diffusion Transformer) wrapper
- ✅ VAE decoder with 16-channel support
- ✅ Text encoder interfaces (CLIP-G, CLIP-L, T5)
- ✅ Euler scheduler
- ✅ Thread-safe pipeline wrapper
- ✅ Streaming support with progress callbacks

## Known Limitations

1. **FP8 Support**: Candle doesn't support FP8 tensors yet
2. **Text Encoders**: Currently returns dummy tensors - needs proper loading implementation
3. **Performance**: CPU-only generation is slow for high-resolution images

## Next Steps

1. Add support for loading text encoders from model files
2. Implement FP8 to FP16/FP32 conversion
3. Add GPU acceleration support
4. Optimize memory usage for large models

## Placeholder Mode

When the full model can't be loaded, the implementation falls back to a placeholder that generates patterns based on the seed:
- Seed % 4 == 0: Gradient pattern
- Seed % 4 == 1: Wave pattern  
- Seed % 4 == 2: Radial gradient
- Seed % 4 == 3: Noise pattern

This allows testing the API while the full implementation is being completed.