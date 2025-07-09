# CUDA Support in Red Candle

## Overview

Red Candle supports CUDA acceleration for NVIDIA GPUs, but it is **disabled by default** to ensure broad compatibility. CUDA support is currently experimental and will be improved in future releases.

## Enabling CUDA Support

To build Red Candle with CUDA support, you need to:

1. Have CUDA installed on your system
2. Set the `CANDLE_ENABLE_CUDA` environment variable during installation

### Installation with CUDA

```bash
# Install from RubyGems with CUDA support
CANDLE_ENABLE_CUDA=1 gem install red-candle

# Or build from source with CUDA support
CANDLE_ENABLE_CUDA=1 bundle install
CANDLE_ENABLE_CUDA=1 bundle exec rake compile
```

### Verifying CUDA Support

You can check if your installation has CUDA support enabled:

```ruby
require 'candle'

# Get build information
info = Candle.build_info
puts "CUDA available: #{info['cuda_available']}"
puts "Default device: #{info['default_device']}"

# List all available devices
devices = Candle::Device.available_devices
puts "Available devices: #{devices}"
```

## Environment Variables

- `CANDLE_ENABLE_CUDA`: Set to `1` to enable CUDA support during build
- `CANDLE_FORCE_CPU`: Set to force CPU-only execution, even if accelerators are available
- `CANDLE_VERBOSE`: Set to see detailed build configuration on gem load
- `CUDA_ROOT` / `CUDA_PATH`: Standard CUDA installation path variables

## Troubleshooting

If you see the message "CUDA detected but not enabled", it means:
- CUDA is installed on your system
- But Red Candle was built without CUDA support
- Reinstall with `CANDLE_ENABLE_CUDA=1` to enable it

## Future Improvements

CUDA support is experimental and will be enhanced in future releases with:
- Better automatic device selection
- Improved error messages
- Performance optimizations
- Multi-GPU support