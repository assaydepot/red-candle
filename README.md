# red-candle

[![build](https://github.com/assaydepot/red-candle/actions/workflows/build.yml/badge.svg)](https://github.com/assaydepot/red-candle/actions/workflows/build.yml)
[![Gem Version](https://badge.fury.io/rb/red-candle.svg)](https://badge.fury.io/rb/red-candle)

 [candle](https://github.com/huggingface/candle) - Minimalist ML framework - for Ruby

## Usage

```ruby
require "candle"

x = Candle::Tensor.new([1, 2, 3, 4, 5, 6], :i64)
x = x.reshape([3, 2])
# [[1., 2.],
#  [3., 4.],
#  [5., 6.]]
# Tensor[[3, 2], f32]
```

```ruby
require 'candle'

# Default model (JinaBERT) on CPU
model = Candle::EmbeddingModel.new
embedding = model.embedding("Hi there!")

# Specify device (CPU, Metal, or CUDA)
device = Candle::Device.cpu     # or Candle::Device.metal, Candle::Device.cuda
model = Candle::EmbeddingModel.new(
  model_path: "jinaai/jina-embeddings-v2-base-en",
  device: device
)
embedding = model.embedding("Hi there!")

# Reranker also supports device selection
reranker = Candle::Reranker.new(
  model_path: "cross-encoder/ms-marco-MiniLM-L-12-v2",
  device: device
)
results = reranker.rerank("query", ["doc1", "doc2", "doc3"])
```

## LLM Support

Red-Candle now supports Large Language Models (LLMs) with GPU acceleration!

### Supported Models

- **Gemma**: Google's Gemma models (e.g., `google/gemma-2b`, `google/gemma-7b`, `google/gemma-2b-it`)
- **Llama**: Llama 2 and Llama 3 models (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `meta-llama/Llama-2-7b-hf`, `NousResearch/Llama-2-7b-hf`)
- **Mistral**: All Mistral models (e.g., `mistralai/Mistral-7B-Instruct-v0.1`)

### Quantized Model Support (GGUF)

Red-Candle supports quantized models in GGUF format, offering 4-8x memory reduction:

> **Note on GGUF Support**: Red-Candle now uses a unified GGUF loader that automatically detects the model architecture from the GGUF file. This means all GGUF models (including Mistral models from TheBloke) should now work correctly! The loader automatically selects the appropriate tokenizer based on the model type to ensure proper text generation.

```ruby
# Load quantized models - always specify the GGUF filename
llm = Candle::LLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGUF", 
                                  device: device, 
                                  gguf_file: "llama-2-7b-chat.Q4_K_M.gguf")

# Register custom tokenizer mappings for your models
Candle::LLM.register_tokenizer("my-org/my-model-GGUF", "my-org/my-tokenizer")

# Popular quantized model sources:
# - TheBloke: Extensive collection of GGUF models
# - Search HuggingFace for "GGUF" models
```

**Memory usage comparison (7B models):**
- Full precision: ~28 GB
- Q8_0 (8-bit): ~7 GB - Best quality, larger size
- Q5_K_M (5-bit): ~4.5 GB - Very good quality  
- Q4_K_M (4-bit): ~4 GB - Recommended default, best balance
- Q3_K_M (3-bit): ~3 GB - Good for memory-constrained systems

**Quantization levels explained:**
- **Q8_0**: Almost identical to full model, use when quality is paramount
- **Q5_K_M**: Excellent quality with good compression
- **Q4_K_M**: Best balance of quality/size/speed (recommended default)
- **Q3_K_M**: Noticeable quality reduction but very compact
- **Q2_K**: ⚠️ **Not recommended** - Can cause inference errors due to extreme quantization

> **Warning**: Q2_K quantization can lead to "weight is negative, too large or not a valid number" errors during inference. Use Q3_K_M or higher for stable operation.

> ### ⚠️ Huggingface login warning
> 
> Many models, including the one below, require you to agree to the terms. You'll need to:
> 1. Login to [Huggingface](https://huggingface.co)
> 2. Agree to the terms. For example: [here](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
> 3. Authenticate your session. Simplest way is with `huggingface-cli login`. Detail here: [Huggingface CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
>
> More details here: [Huggingface Authentication](HUGGINGFACE.md)

```ruby
require 'candle'

# Choose your device
device = Candle::Device.cpu     # CPU (default)
device = Candle::Device.metal   # Apple GPU (Metal)
device = Candle::Device.cuda    # NVIDIA GPU (CUDA)

# Load a model
llm = Candle::LLM.from_pretrained("google/gemma-2b-it", device: device)  # Gemma
# llm = Candle::LLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: device)  # Llama
# llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device: device)  # Mistral

# Generate text
response = llm.generate("What is Ruby?", config: Candle::GenerationConfig.balanced)

# Stream generation
llm.generate_stream("Tell me a story", config: Candle::GenerationConfig.balanced) do |token|
  print token
end

# Chat interface
messages = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "Explain Ruby in one sentence." }
]
response = llm.chat(messages)
```

### GPU Acceleration

```ruby
# CPU works for all models
device = Candle::Device.cpu
llm = Candle::LLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: device)

# Metal
device = Candle::Device.metal 

# CUDA support (for NVIDIA GPUs COMING SOON)
device = Candle::Device.cuda   # Linux/Windows with NVIDIA GPU
```

### Debugging Token Generation

For debugging purposes, you can enable raw token output to see both token IDs and their raw representations:

```ruby
# Enable debug mode to see raw tokens during generation
config = Candle::GenerationConfig.balanced(debug_tokens: true)

# Non-streaming generation with debug tokens
result = llm.generate("Hello, world!", config: config)
puts result
# Output: [15043:Hello][11:,][1917:world][0:!]

# Streaming generation with debug tokens
llm.generate_stream("Hello, world!", config: config) do |text|
  print text  # Will show each token as it's generated: [15043:Hello][11:,][1917:world][0:!]
end

# Works with all models (Llama, Mistral, Gemma, and quantized GGUF models)
```

This is particularly useful for:
- Debugging tokenization issues
- Understanding how the model processes text
- Troubleshooting generation problems
- Analyzing model behavior

## ⚠️ Model Format Requirements

### EmbeddingModels and Rerankers: Safetensors Only

Red-Candle **only supports embedding models and rerankers that provide their weights in the [safetensors](https://github.com/huggingface/safetensors) format** (i.e., the model repo must contain a `model.safetensors` file). If the model repo does not provide the required file, loading will fail with a clear error. Most official BERT and DistilBERT models do **not** provide safetensors; many Sentence Transformers and JinaBERT models do.

**If you encounter an error like:**

```
RuntimeError: model.safetensors not found after download. Only safetensors models are supported. Please ensure your model repo contains model.safetensors.
```

this means the selected model is not compatible. Please choose a model repo that provides the required file.

### LLMs: Safetensors and GGUF Support

LLM models support two formats:
1. **Safetensors format** - Standard HuggingFace models (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
2. **GGUF quantized format** - Memory-efficient quantized models (e.g., `TheBloke/Llama-2-7B-Chat-GGUF`)

See the [Quantized Model Support](#quantized-model-support-gguf) section for details on using GGUF models.

## Supported Embedding Models

Red-Candle supports the following embedding model types from Hugging Face:

1. `Candle::EmbeddingModelType::JINA_BERT` - Jina BERT models (e.g., `jinaai/jina-embeddings-v2-base-en`) (**safetensors required**)
2. `Candle::EmbeddingModelType::MINILM` - MINILM models (e.g., `sentence-transformers/all-MiniLM-L6-v2`) (**safetensors required**)
3. `Candle::EmbeddingModelType::DISTILBERT` - DistilBERT models (e.g., `distilbert-base-uncased-finetuned-sst-2-english`) (**safetensors required**)
4. `Candle::EmbeddingModelType::STANDARD_BERT` - Standard BERT models (e.g., `scientistcom/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`) (**safetensors required**)

> **Note:** Most official BERT and DistilBERT models do _not_ provide safetensors. Please check the model repo before use.

You can get a list of all supported model types and suggested models paths:

```ruby
Candle::EmbeddingModelType.all  # Returns all supported model types
Candle::EmbeddingModelType.suggested_model_paths  # Returns hash of suggested models for each type
```

## A note on memory usage
The default model (`jinaai/jina-embeddings-v2-base-en` with the `sentence-transformers/all-MiniLM-L6-v2` tokenizer, both from [HuggingFace](https://huggingface.co)) takes a little more than 3GB of memory running on a Mac. The memory stays with the instantiated `Candle::EmbeddingModel` class, if you instantiate more than one, you'll use more memory. Likewise, if you let it go out of scope and call the garbage collector, you'll free the memory. For example:

```ruby
> require 'candle'
# Ruby memory = 25.9 MB
> model = Candle::EmbeddingModel.new
# Ruby memory = 3.50 GB
> model2 = Candle::EmbeddingModel.new
# Ruby memory = 7.04 GB
> model2 = nil
> GC.start
# Ruby memory = 3.56 GB
> model = nil
> GC.start
# Ruby memory = 55.2 MB
```

## A note on returned embeddings

The code should match the same embeddings when generated from the python `transformers` library. For instance, locally I was able to generate the same embedding for the text "Hi there!" using the python code:

```python
from transformers import AutoModel
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
sentence = ['Hi there!']
embedding = model.encode(sentence)
print(embedding)
```

And the following ruby:

```ruby
require 'candle'
model = Candle::EmbeddingModel.new
embedding = model.embedding("Hi there!")
```

## Document Reranking

Red-Candle includes support for cross-encoder reranking models, which can be used to reorder documents by relevance to a query. This is particularly useful for improving search results or implementing retrieval-augmented generation (RAG) systems.

### Basic Usage

```ruby
require 'candle'

# Initialize the reranker with a cross-encoder model
reranker = Candle::Reranker.new(model_path: "cross-encoder/ms-marco-MiniLM-L-12-v2")

# Define your query and candidate documents
query = "How many people live in London?"
documents = [
  "London is known for its financial district",
  "Around 9 Million people live in London", 
  "The weather in London is often rainy",
  "London is the capital of England"
]

# Rerank documents by relevance to the query (raw logits)
ranked_results = reranker.rerank(query, documents, pooling_method: "pooler", apply_sigmoid: false)

# Or apply sigmoid activation to get scores between 0 and 1
sigmoid_results = reranker.rerank(query, documents, pooling_method: "pooler", apply_sigmoid: true)

# The pooler method is the default and is recommended for cross-encoders, as is apply_sigmod, so the above is the same as:
ranked_results = reranker.rerank(query, documents)

# Results are returned as an array of hashes, sorted by relevance
e.g.
ranked_results.each do |result|
  puts "Score: #{result[:score].round(4)} - Doc ##{result[:doc_id]}: #{result[:text]}"
end
# Output:
# Score: 1.0 - Doc #1: Around 9 Million people live in London
# Score: 0.0438 - Doc #3: London is the capital of England
# Score: 0.0085 - Doc #0: London is known for its financial district
# Score: 0.0005 - Doc #2: The weather in London is often rainy
```

### Arguments & Activation Functions

By default, `apply_sigmoid` is `true` (scores between 0 and 1). Set it to `false` to get raw logits. You can also select the pooling method:

- `pooling_method: "pooler"` (default)
- `pooling_method: "cls"`
- `pooling_method: "mean"`

Example without sigmoid activation:

```ruby
# Get raw logits
ranked_results = reranker.rerank(query, documents, apply_sigmoid: false)

ranked_results.each do |result|
  puts "Score: #{result[:score].round(4)} - Doc ##{result[:doc_id]}: #{result[:text]}"
end
# Output:
# Score: 10.3918 - Doc #1: Around 9 Million people live in London
# Score: -3.0829 - Doc #3: London is the capital of England
# Score: -4.7619 - Doc #0: London is known for its financial district
# Score: -7.5251 - Doc #2: The weather in London is often rainy
```

### Output Format

The reranker returns an array of hashes, each with the following keys:
- `:text` – The original document text
- `:score` – The relevance score (raw logit or sigmoid-activated)
- `:doc_id` – The original 0-based index of the document in the input array

This format is compatible with the Informers gem, which returns results as hashes with `:doc_id` and `:score` keys. The `doc_id` allows you to map results back to your original data structure.

### Pooling Methods

The reranker supports different pooling strategies for aggregating BERT embeddings:

```ruby
# Use alternative pooling methods
# "pooler" (default) - Uses the pooler layer with tanh activation (most accurate for cross-encoders)
# "cls" - Uses raw [CLS] token embeddings without the pooler layer
# "mean" - Mean pooling across all tokens (not recommended for cross-encoders)

# With raw logits
results = reranker.rerank_with_pooling(query, documents, "cls")

# With sigmoid activation
results = reranker.rerank_sigmoid_with_pooling(query, documents, "cls")
```

Note: The default "pooler" method is recommended as it matches how cross-encoder models are trained. Other pooling methods may produce different ranking results.

### CUDA Support

For faster inference on NVIDIA GPUs:

```ruby
# Initialize with CUDA if available (falls back to CPU if not)
reranker = Candle::Reranker.new(model_path: "cross-encoder/ms-marco-MiniLM-L-12-v2", cuda: true)
```

### How It Works

Cross-encoder reranking models differ from bi-encoder embedding models:

- **Bi-encoders** (like the embedding models above) encode queries and documents separately into dense vectors
- **Cross-encoders** process the query and document together, allowing for more nuanced relevance scoring

The reranker uses a BERT-based architecture that:
1. Concatenates the query and document with special tokens: `[CLS] query [SEP] document [SEP]`
2. Processes them jointly through BERT layers
3. Applies a pooler layer (dense + tanh) to the [CLS] token
4. Uses a classifier layer to produce a single relevance score

This joint processing allows cross-encoders to capture subtle semantic relationships between queries and documents, making them more accurate for reranking tasks, though at the cost of higher computational requirements.

## Common Runtime Errors

### 1. Weight is negative, too large or not a valid number

**Error:**
```
/Users/cpetersen/src/scientist/red-candle/lib/candle/llm.rb:25:in `_generate_stream': Generation failed: A weight is negative, too large or not a valid number (RuntimeError)
    from /Users/cpetersen/src/scientist/red-candle/lib/candle/llm.rb:25:in `generate_stream'
    ...
```

**Cause:** This error occurs when using overly aggressive quantization levels (particularly Q2_K) that result in numerical instability during inference. The 2-bit quantization can cause weights to become corrupted or produce NaN/Inf values.

**Solution:** Use a higher quantization level. Recommended options:
- Q4_K_M (4-bit) - Best balance of quality and size
- Q5_K_M (5-bit) - Higher quality with slightly larger size
- Q3_K_M (3-bit) - Minimum recommended quantization

```ruby
# Instead of Q2_K:
llm = Candle::LLM.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", 
                                  device: device, 
                                  gguf_file: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
```

### 2. Cannot find tensor model.embed_tokens.weight

**Error:**
```
Failed to load quantized model: cannot find tensor model.embed_tokens.weight (RuntimeError)
```

**Cause:** This error was common in earlier versions when loading GGUF files with incompatible tensor naming conventions. The unified GGUF loader in version 1.0.0+ should handle most GGUF files correctly.

**If you still encounter this error:**
1. Ensure you're using the latest version of red-candle (1.0.0 or higher)
2. Make sure to specify the exact GGUF filename:
   ```ruby
   llm = Candle::LLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", 
                                     device: device,
                                     gguf_file: "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
   ```
3. If the error persists, the GGUF file may use an unsupported architecture or format

### 3. No GGUF file found in repository

**Error:**
```
Failed to load quantized model: No GGUF file found in repository TheBloke/model-name-GGUF. Try specifying a quantization level like Q4_K_M, Q5_K_M, or Q8_0. (RuntimeError)
```

**Cause:** The automatic GGUF file detection couldn't find a matching file, often due to naming variations.

**Solution:** Specify the exact GGUF filename:
```ruby
# Visit the HuggingFace repository to find the exact filename
llm = Candle::LLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGUF", 
                                  device: device, 
                                  gguf_file: "llama-2-7b-chat.Q4_K_M.gguf")
```

### 4. Failed to download tokenizer

**Error:**
```
Failed to load quantized model: Failed to download tokenizer: request error: HTTP status client error (404 Not Found)
```

**Cause:** GGUF repositories often don't include separate tokenizer files since they're embedded in the GGUF format.

**Solution:** The code now includes fallback tokenizer loading. If you still encounter this error, ensure you're using the latest version of red-candle.

### 5. Missing metadata in GGUF file

**Error:**
```
Failed to load GGUF model: cannot find gemma3.attention.head_count in metadata (RuntimeError)
```
or
```
Failed to load GGUF model: cannot find llama.attention.head_count in metadata (RuntimeError)
```

**Cause:** Some GGUF files may have been created with older conversion tools that don't include all required metadata fields.

**Solution:** 
- Try a different GGUF file from the same model
- Look for GGUF files from TheBloke or other reputable sources
- Check if a newer version of the GGUF file is available
- Some Gemma GGUF files may not be compatible with the current loader

**Known compatibility issues:**
- `lmstudio-ai/gemma-2b-it-GGUF` - Missing required metadata fields
- Gemma 3 GGUF files may require specific tokenizers that are not publicly available
- For best compatibility, use Llama or Mistral GGUF files from TheBloke

## Development

FORK IT!

```
git clone https://github.com/your_name/red-candle
cd red-candle
bundle
bundle exec rake compile
```

Implemented with [Magnus](https://github.com/matsadler/magnus), with reference to [Polars Ruby](https://github.com/ankane/polars-ruby)

Pull requests are welcome.

### See Also

- [Numo::NArray](https://github.com/ruby-numo/numo-narray)
- [Cumo](https://github.com/sonots/cumo)
