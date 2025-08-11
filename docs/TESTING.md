# Testing Guide

This guide covers how to test Red Candle across different devices (CPU, Metal, CUDA).

## Overview

Red Candle supports three types of compute devices:
- **CPU**: Always available, works on all platforms
- **Metal**: Apple GPU acceleration (macOS only)
- **CUDA**: NVIDIA GPU acceleration (Linux/Windows with NVIDIA GPU)

All three model types work on all available devices:
- **EmbeddingModel**: For generating text embeddings
- **Reranker**: For reranking documents by relevance
- **LLM**: For text generation (various model architectures)

## Running Tests

### Basic Test Suite
```bash
# Run all tests
bundle exec rake test
```

### Device-Specific Tests
```bash
# Test all available devices
bundle exec rake test:device

# Test specific devices
bundle exec rake test:device:cpu
bundle exec rake test:device:metal
bundle exec rake test:device:cuda
```

### Performance Benchmarks
```bash
# Run benchmarks only
bundle exec rake test:benchmark

# Run device tests with benchmarks
bundle exec rake test:device:benchmark
```

## Environment Variables

- `CANDLE_TEST_DEVICES`: Comma-separated list of devices to test (cpu,metal,cuda)
- `CANDLE_TEST_VERBOSE`: Set to 'true' for verbose test output

