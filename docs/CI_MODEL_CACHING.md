# CI Model Caching Strategy

## Overview

To avoid HuggingFace rate limiting (HTTP 429 errors) in CI, we implement a model caching strategy that pre-downloads all required models and reuses them across test runs.

## Implementation

### 1. GitHub Actions Cache

The CI workflow uses GitHub Actions cache to store downloaded models:

```yaml
- name: Cache HuggingFace models
  uses: actions/cache@v3
  with:
    path: |
      ~/.cache/huggingface
      ~/Library/Caches/huggingface
    key: ${{ runner.os }}-huggingface-models-v1-${{ hashFiles('test/model_manifest.txt') }}
```

### 2. Model Manifest

All models required for tests are listed in `test/model_manifest.txt`:

```
# NER Models
Babelscape/wikineural-multilingual-ner
# Note: dslim/bert-base-NER uses vocab.txt format which may not be fully supported
# It's kept as a fallback but the primary model above should work

# Embedding Models
sentence-transformers/all-MiniLM-L6-v2
jinaai/jina-embeddings-v2-base-en

# Reranker Models
cross-encoder/ms-marco-MiniLM-L-12-v2

# LLM Models (GGUF)
TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF gguf:tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf

# Tokenizers
bert-base-uncased
gpt2
TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Small test models
microsoft/phi-2
```

### 3. Pre-download Scripts

Two download scripts are available:

#### `script/download_test_models.rb` (Primary)
- Uses the Candle library to download models
- Requires the native extension to be compiled first
- Downloads models in the exact format needed by tests
- Handles compatibility issues (e.g., BERT tokenizer format)
- Implements retry logic with exponential backoff
- Gracefully handles expected failures (e.g., dslim/bert-base-NER)

#### `script/download_models_http.rb` (Fallback)
- Uses direct HTTP requests (no Candle required)
- Can run before compilation
- Downloads model files to HuggingFace cache structure
- Handles both modern (tokenizer.json) and legacy (vocab.txt) formats
- Used as fallback if primary script fails or compilation hasn't happened

### 4. Smart Model Loader

`test/support/model_loader.rb` provides fallback strategies:

1. Try loading from local cache
2. Try downloading from HuggingFace
3. Retry with exponential backoff on rate limits
4. Skip test gracefully if all strategies fail

## Usage

### In CI

The workflow automatically:
1. Restores cached models if available (GitHub Actions cache)
2. Compiles the native extension (`rake compile`)
3. Pre-downloads missing models (with fallback to HTTP downloader)
4. Runs tests in offline mode with cached models (`HF_OFFLINE=true`)

### Locally

```bash
# First, compile the native extension
bundle exec rake compile

# Pre-download all test models
bundle exec ruby script/download_test_models.rb

# Or use HTTP downloader (doesn't require compilation)
ruby script/download_models_http.rb

# Run tests with offline mode
HF_OFFLINE=true bundle exec rake test

# Run tests with specific cache directory
HF_HOME=/path/to/cache bundle exec rake test
```

### With HuggingFace Token

For better rate limits, add your HuggingFace token:

```bash
# In CI: Add as GitHub secret HF_TOKEN

# Locally:
export HF_TOKEN=your_token_here
bundle exec ruby script/download_test_models.rb
```

## Environment Variables

- `HF_HOME`: HuggingFace cache directory (default: ~/.cache/huggingface)
- `HF_TOKEN`: HuggingFace API token for authenticated requests
- `HF_OFFLINE`: Set to "true" to use only cached models
- `CI_OFFLINE_MODE`: Alternative to HF_OFFLINE for CI environments
- `LOCAL_MODEL_PATH`: Override path for local model storage

## Cache Management

### Cache Size

Typical cache sizes:
- Tokenizers: ~5MB each
- Embedding models: 100-500MB each
- NER models: 200-400MB each
- Rerankers: 200-300MB each
- GGUF models: 500MB-2GB each

Total cache for test suite: ~3-4GB

### Cache Invalidation

The cache key includes a hash of `test/model_manifest.txt`, so:
- Adding/removing models invalidates the cache
- Models are re-downloaded when the manifest changes
- Increment the version suffix (v1, v2) to force cache refresh

### Manual Cache Clear

```bash
# Clear local cache
rm -rf ~/.cache/huggingface

# In CI: Increment cache key version
# Change: huggingface-models-v1-
# To: huggingface-models-v2-
```

## Troubleshooting

### Rate Limiting Still Occurs

1. Check if `HF_TOKEN` is set correctly as a GitHub secret
2. Verify models are in the manifest
3. Check cache is being restored (look for "Cache restored" in CI logs)
4. Try increasing retry delays in the download script

### Model Compatibility Issues

Some models may fail to download due to tokenizer format incompatibilities:

- **dslim/bert-base-NER**: Uses older BERT tokenizer format (vocab.txt instead of tokenizer.json)
- This is expected and handled - tests use fallback models
- The download scripts will report these as non-fatal errors

### Models Not Found in Cache

1. Verify `HF_HOME` points to correct directory
2. Check manifest file is complete
3. Ensure compilation succeeded before running download script
4. Try the HTTP downloader as a fallback
5. Check disk space for cache storage (~3-4GB needed)

### Compilation Errors

If the native extension fails to compile:

```bash
# Use the HTTP downloader instead
ruby script/download_models_http.rb

# This downloads files directly without needing Candle
```

### Tests Skip When Models Unavailable

This is expected behavior to prevent CI failures. To force downloads:

```bash
# Disable offline mode
unset HF_OFFLINE
unset CI_OFFLINE_MODE

# Enable heavy tests
export ENABLE_HEAVY_TESTS=true
```

## Benefits

1. **No rate limiting**: Models are downloaded once and cached
2. **Faster CI runs**: No download time during tests
3. **Reliable tests**: No failures due to network issues
4. **Cost efficient**: Uses free GitHub Actions cache
5. **Offline testing**: Can run tests without internet

## Future Improvements

1. **S3 Mirror**: Set up S3 bucket with models for faster downloads
2. **Docker Image**: Create image with pre-loaded models
3. **Git LFS**: Store small models directly in repo
4. **Model Pruning**: Remove unused models from manifest
5. **Parallel Downloads**: Speed up initial cache population

## Related Files

- `.github/workflows/build.yml` - CI workflow configuration with caching and compilation
- `test/model_manifest.txt` - List of required models (with compatibility notes)
- `script/download_test_models.rb` - Primary pre-download script (requires compilation)
- `script/download_models_http.rb` - Fallback HTTP downloader (no compilation needed)
- `test/support/model_loader.rb` - Smart loading with fallbacks
- `test/test_helper.rb` - Test environment configuration