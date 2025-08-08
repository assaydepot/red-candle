# Red Candle Build Configuration

Red Candle supports conditional compilation to enable hardware acceleration when available. The gem will automatically detect and enable supported acceleration methods during installation.

## Supported Acceleration Methods

- **CPU**: Always available (fallback)
- **CUDA**: NVIDIA GPU acceleration (requires CUDA toolkit)
- **Metal**: Apple Silicon/GPU acceleration (macOS only)
- **MKL**: Intel Math Kernel Library acceleration
- **Accelerate**: Apple's Accelerate framework (macOS only)

## Automatic Detection

During gem installation, the build system automatically detects:

1. **CUDA**: Checks for CUDA installation via:
   - Environment variables: `CUDA_ROOT`, `CUDA_PATH`
   - Common installation paths: `/usr/local/cuda`, `/opt/cuda`
   - Windows paths: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`

2. **Metal**: Automatically enabled on macOS

3. **MKL**: Checks for Intel MKL via:
   - Environment variables: `MKLROOT`, `MKL_ROOT`
   - Common paths: `/opt/intel/mkl`, `/opt/intel/oneapi/mkl/latest`

4. **Accelerate**: Automatically enabled on macOS

## Environment Variables

### Build-time Configuration

- `CANDLE_FORCE_CPU=1` - Force CPU-only build, disable all acceleration
- `CANDLE_FEATURES=cuda,metal` - Manually specify features (comma-separated)
- `CANDLE_CUDA_PATH=/path/to/cuda` - Override CUDA detection path
- `CANDLE_CUDNN=1` - Enable cuDNN support (requires CUDA)
- `CUDNN_ROOT=/path/to/cudnn` - Specify cuDNN installation path
- `CANDLE_CARGO_FLAGS="--flags"` - Pass additional flags to cargo

### Examples

```bash
# Force CPU-only build
CANDLE_FORCE_CPU=1 gem install red-candle

# Build with specific CUDA installation
CANDLE_CUDA_PATH=/opt/cuda-12.0 gem install red-candle

# Build with cuDNN support
CANDLE_CUDNN=1 CUDNN_ROOT=/usr/local/cudnn gem install red-candle

# Manually specify features
CANDLE_FEATURES=cuda,mkl gem install red-candle
```

## Runtime Usage

### Check Available Devices

```ruby
require 'candle'

# List available devices
puts Candle::Device.available_devices
# => ["cpu", "cuda", "metal"]

# Get default device
device = Candle::Device.default
# => #<Candle::Device:metal>

# Check build information
puts Candle.build_info
# => {
#   "default_device" => "metal",
#   "cuda_available" => false,
#   "metal_available" => true,
#   "mkl_available" => false,
#   "accelerate_available" => true,
#   "cudnn_available" => false
# }
```

### Device Selection

```ruby
# Use specific device
cpu_device = Candle::Device.cpu
cuda_device = Candle::Device.cuda  # Raises error if CUDA not compiled in
metal_device = Candle::Device.metal # Raises error if Metal not compiled in

# Create tensor on specific device
tensor = Candle::Tensor.new([1, 2, 3], device: cuda_device)
```

## Troubleshooting

### CUDA Not Detected

1. Ensure CUDA toolkit is installed
2. Set `CUDA_PATH` or `CUDA_ROOT` environment variable
3. Add CUDA binaries to PATH

### Metal Not Available

Metal is only available on macOS. Ensure you're building on a Mac.

### Build Fails with Acceleration

If the build fails with acceleration features:

1. Try forcing CPU-only: `CANDLE_FORCE_CPU=1 gem install red-candle`
2. Check that required libraries are installed
3. Review build output for specific error messages

### Runtime Errors

If you get "support not compiled in" errors:

1. Check `Candle.build_info` to see what was compiled
2. Rebuild the gem with the required features available
3. Use `Candle::Device.available_devices` to see runtime-available devices