# Candle vs Informers Comparison Tests

These tests compare the outputs of Candle (Rust-based) and Informers (ONNX-based) implementations to ensure compatibility.

## Setup

First, install the informers gem:

```bash
bundle install
```

## Running the Tests

### Automated Tests

Run the comparison test suite:

```bash
CANDLE_RUN_COMPARISON_TESTS=true bundle exec ruby test/candle_informers_comparison_test.rb
```

With verbose output:

```bash
CANDLE_RUN_COMPARISON_TESTS=true CANDLE_TEST_VERBOSE=true bundle exec ruby test/candle_informers_comparison_test.rb
```

### Manual Debugging

Run the debug script to see detailed output comparisons:

```bash
bundle exec ruby test/debug_comparison.rb
```

## What's Being Tested

### Reranker Comparison

The test compares the cross-encoder/ms-marco-MiniLM-L-12-v2 model outputs between:
- Informers' default reranking pipeline
- Candle's reranker with various configurations:
  - Pooling methods: pooler (default), cls, mean
  - Sigmoid activation: on/off

The test finds which Candle configuration best matches Informers' output.

### Embedding Model Comparison

**Note**: The embedding model comparison is currently skipped because Informers has limited support for sentence-transformers models in ONNX format.

To verify Candle's embedding functionality, run the standalone verification test:

```bash
bundle exec ruby test/candle_embedding_verification_test.rb
```

This test verifies:
- Embeddings are properly normalized (L2 norm = 1.0)
- Similar texts have high cosine similarity
- Different texts have lower cosine similarity

## Expected Results

Due to slight differences in floating-point operations between Rust and ONNX implementations, we expect:
- Differences < 1e-4 (0.0001) for individual values
- Cosine similarity > 0.9999 for embedding vectors
- The same relative ranking for reranker scores

## Troubleshooting

If tests fail:
1. Check that both libraries are using the same model version
2. Verify the pooling and normalization methods match
3. Run the debug script to see detailed differences
4. Consider adjusting FLOAT_TOLERANCE if differences are consistently small