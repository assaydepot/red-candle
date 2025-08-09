# CI Model Caching Strategy

## Overview

This document describes the model caching strategy for GitHub Actions CI to avoid HuggingFace rate limiting (HTTP 429 errors).

## Problem

- HuggingFace rate limits API requests (429 errors)
- Tests download multiple models during execution
- Rate limiting causes test failures in CI
- Even with HF_TOKEN, rate limits can still be hit with many parallel requests

## Solution

We use a two-pronged approach:
1. **HF_TOKEN** for authentication (provides higher rate limits)
2. **HF_HUB_OFFLINE** mode when cache is populated (avoids API calls entirely)

### 1. GitHub Actions Cache

The workflow caches the HuggingFace model directory:

```yaml
- name: Cache HuggingFace models
  uses: actions/cache@v3
  with:
    path: |
      ~/.cache/huggingface
      /home/runner/.cache/huggingface
    key: ${{ runner.os }}-huggingface-models-v2-${{ hashFiles('test/model_manifest.txt') }}
```

### 2. Model Manifest

All required models are listed in `test/model_manifest.txt`:
- Used as cache key for invalidation
- Documents all model dependencies
- Format: `model_id [optional: gguf:filename]`

### 3. Pre-download Script

`script/download_test_models.rb`:
- Reads model manifest
- Downloads models using Candle library (which uses hf_hub internally)
- Creates proper cache structure that Rust code can reuse
- Handles rate limiting with retries

**Important**: This script requires the native extension to be compiled first, which is why the CI workflow compiles before downloading.

### 4. Important Notes

#### Cache Structure

The hf_hub Rust crate uses a specific cache structure:
```
~/.cache/huggingface/hub/
├── models--org--model/
│   ├── blobs/          # Content-addressed storage
│   ├── refs/           # Branch references
│   └── snapshots/      # Commit snapshots
│       └── <sha>/      # Files for specific commit
```

This structure can only be created by the hf_hub crate itself. Any attempt to manually create this structure will fail because it requires:
1. Git commit SHAs for snapshots
2. Content-addressed blob storage with symlinks
3. Proper metadata files

Therefore, models must be downloaded using the actual hf_hub crate (via Candle).

#### Cache Location

Default locations by platform:
- **Linux**: `~/.cache/huggingface/hub`
- **macOS**: `~/.cache/huggingface/hub` (not `~/Library/Caches`)
- **Windows**: `%USERPROFILE%\.cache\huggingface\hub`

The hf_hub crate respects these environment variables:
- `HF_HOME`: Override entire HuggingFace home
- `HUGGINGFACE_HUB_CACHE`: Override hub cache specifically
- `XDG_CACHE_HOME`: Standard XDG cache directory

**Important**: We do NOT set these variables in CI to ensure the Rust code uses its default locations.

## Workflow

1. **Cache Restore**: GitHub Actions restores cached models if available
2. **Compilation**: Native extension compiled with `rake compile`
3. **Cache Check**: Determine if cache has sufficient models
4. **Conditional Download**:
   - If cache is incomplete: Download missing models with HF_TOKEN
   - If cache is complete: Skip downloads
5. **Test Execution**:
   - If cache is complete: Run with `HF_HUB_OFFLINE=1` (no API calls)
   - If cache is incomplete: Run normally with HF_TOKEN
6. **Cache Save**: GitHub Actions saves any newly downloaded models

## Debugging

To debug cache issues:

1. Check cache location:
```bash
echo "HOME: $HOME"
echo "HF_HOME: ${HF_HOME:-not set}"
ls -la ~/.cache/huggingface/hub/
```

2. Test if model loads from cache:
```ruby
start = Time.now
Candle::Tokenizer.from_pretrained("bert-base-uncased")
puts "Loaded in #{Time.now - start}s"
# < 1s = cached, > 3s = downloaded
```

3. Check for rate limiting:
- Look for HTTP 429 errors in logs
- Add HF_TOKEN to environment for higher limits

## Known Issues

### dslim/bert-base-NER

This model returns 404 errors because it uses an older tokenizer format (`vocab.txt` instead of `tokenizer.json`). It's kept in tests as a fallback example but the primary NER model is `Babelscape/wikineural-multilingual-ner`.

## Environment Variables

For CI, we use:
- `HF_TOKEN`: Authentication token for higher rate limits (stored as GitHub secret)
- `HF_HUB_OFFLINE=1`: Set dynamically when cache is complete to avoid all API calls

We do NOT set:
- `HF_HOME`: Let hf_hub use default location
- `HUGGINGFACE_HUB_CACHE`: Let hf_hub use default location

## Testing Locally

To test the caching strategy locally:

```bash
# Clear cache
rm -rf ~/.cache/huggingface

# Compile extension
bundle exec rake compile

# Download models
bundle exec ruby script/download_test_models.rb

# Run tests (should use cache)
bundle exec rake test
```

## Cache Invalidation

The cache key includes `hashFiles('test/model_manifest.txt')`, so the cache is invalidated when:
- New models are added to the manifest
- Model versions are changed
- The manifest file is modified in any way

To force cache refresh, increment the version in the cache key (e.g., `v2` to `v3`).