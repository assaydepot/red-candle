# `red-candle` Native LLMs for Ruby 🚀

[![build](https://github.com/assaydepot/red-candle/actions/workflows/build.yml/badge.svg)](https://github.com/assaydepot/red-candle/actions/workflows/build.yml)
[![Gem Version](https://badge.fury.io/rb/red-candle.svg)](https://badge.fury.io/rb/red-candle)

Run state-of-the-art **language models directly from Ruby**. No Python, no APIs, no external services - just Ruby with blazing-fast Rust under the hood. Hardware accelerated with **Metal (Mac)** and **CUDA (NVIDIA).**

## Install & Chat in 30 Seconds

[![red-candle quickstart](https://img.youtube.com/vi/hbyFCyh8esk/0.jpg)](https://www.youtube.com/watch?v=hbyFCyh8esk)

```bash
# Install the gem
gem install red-candle
```

```ruby
require 'candle'

# Download a model (one-time, ~650MB) - Mistral, Llama3, Gemma all work!
llm = Candle::LLM.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", 
                                  gguf_file: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# Chat with it - no API calls, running locally in your Ruby process!
messages = [
  { role: "user", content: "Explain Ruby in one sentence" }
]

puts llm.chat(messages)
# => "Ruby is a dynamic, object-oriented programming language known for its 
#     simplicity, elegance, and productivity, often used for web development 
#     with frameworks like Rails."
```

## What Just Happened?

You just ran a 1.1-billion parameter AI model inside Ruby. The model lives in your process memory, runs on your hardware (CPU/GPU), and responds instantly without network latency.

## Stream Responses Like a Pro

```ruby
# Watch the AI think in real-time
llm.chat_stream(messages) do |token|
  print token
end
```

## Why This Matters

- **Privacy**: Your data never leaves your machine
- **Speed**: No network overhead, direct memory access
- **Control**: Fine-tune generation parameters, access raw tokens
- **Integration**: It's just Ruby objects - use it anywhere Ruby runs

## Supports

- **Tokenizers**: Access the tokenizer directly
- **EmbeddingModel**: Generate embeddings for text
- **Reranker**: Rerank documents based on relevance
- **NER**: Named Entity Recognition directly from Ruby
- **LLM**: Chat with Large Language Models (e.g., Llama, Mistral, Gemma)

## Model Storage

Models are automatically downloaded and cached when you first use them. They are stored in:
- **Location**: `~/.cache/huggingface/hub/`
- **Size**: Models range from ~100MB (embeddings) to several GB (LLMs)
- **Reuse**: Models are downloaded once and reused across sessions

To check your cache or manage storage:
```bash
# View cache contents
ls -la ~/.cache/huggingface/hub/

# Check total cache size
du -sh ~/.cache/huggingface/

# Clear cache if needed (removes all downloaded models)
rm -rf ~/.cache/huggingface/hub/
```

----

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

We see an 18x speed up running LLMs under CUDA vs CPU and a >3x speed up running under Metal vs CPU. Details [here](DEVICE_SUPPORT.md#performance-considerations).

```ruby
# CPU works for all models
device = Candle::Device.cpu
llm = Candle::LLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: device)

# Metal
device = Candle::Device.metal 

# CUDA support (for NVIDIA GPUs)
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

## Structured Generation

Red Candle supports structured generation to constrain LLM outputs to follow specific patterns like JSON schemas or regular expressions:

```ruby
# Define a JSON schema
schema = {
  type: "object",
  properties: {
    answer: { type: "string", enum: ["yes", "no"] },
    confidence: { type: "number", minimum: 0, maximum: 1 }
  },
  required: ["answer"]
}

# Create constraint from schema
constraint = llm.constraint_from_schema(schema)

# Generate with constraint
config = Candle::GenerationConfig.balanced(constraint: constraint)
result = llm.generate("Is Ruby easy to learn?", config: config)
# Output: {"answer": "yes", "confidence": 0.9}

# Or use regex patterns
phone_constraint = llm.constraint_from_regex('\d{3}-\d{3}-\d{4}')
```

See [STRUCTURED_GENERATION.md](docs/STRUCTURED_GENERATION.md) for detailed documentation.

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

# The pooler method is the default and is recommended for cross-encoders, as is apply_sigmoid, so the above is the same as:
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

## Tokenizer

Red-Candle provides direct access to tokenizers for text preprocessing and analysis. This is useful for understanding how models process text, debugging issues, and building custom NLP pipelines.

### Basic Usage

```ruby
require 'candle'

# Load a tokenizer from HuggingFace
tokenizer = Candle::Tokenizer.from_pretrained("bert-base-uncased")

# Encode text to token IDs
token_ids = tokenizer.encode("Hello, world!")
# => [101, 7592, 1010, 2088, 999, 102]

# Decode token IDs back to text
text = tokenizer.decode(token_ids)
# => "hello, world!"

# Get token strings (subwords) - useful for visualization
tokens = tokenizer.encode_to_tokens("Hello, world!")
# => ["[CLS]", "hello", ",", "world", "!", "[SEP]"]

# Get both IDs and tokens together
result = tokenizer.encode_with_tokens("preprocessing")
# => {"ids" => [101, 3653, 22618, 2527, 102], 
#     "tokens" => ["[CLS]", "prep", "##ro", "##ces", "##sing", "[SEP]"]}
```

### Batch Processing

```ruby
# Encode multiple texts at once
texts = ["Hello world", "How are you?", "Tokenizers are cool"]
batch_ids = tokenizer.encode_batch(texts)

# Get token strings for multiple texts
batch_tokens = tokenizer.encode_batch_to_tokens(texts)
```

### Vocabulary Access

```ruby
# Get vocabulary size
vocab_size = tokenizer.vocab_size
# => 30522

# Get full vocabulary as a hash
vocab = tokenizer.get_vocab
# vocab["hello"] => 7592

# Convert a specific token ID to its string
token_str = tokenizer.id_to_token(7592)
# => "hello"

# Get special tokens
special = tokenizer.get_special_tokens
# => {"cls_token" => 101, "sep_token" => 102, "pad_token" => 0, ...}
```

### Configuration

```ruby
# Create a tokenizer with padding enabled
padded_tokenizer = tokenizer.with_padding(length: 128)

# Create a tokenizer with truncation
truncated_tokenizer = tokenizer.with_truncation(512)

# Configure padding with more options
padded_tokenizer = tokenizer.with_padding(
  length: 128,          # Fixed length padding
  direction: "right",   # Pad on the right (default)
  pad_token: "[PAD]"    # Padding token
)
```

### Model Integration

All models expose their tokenizers:

```ruby
# From LLM
llm = Candle::LLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
llm_tokenizer = llm.tokenizer

# From EmbeddingModel
embedding_model = Candle::EmbeddingModel.new
emb_tokenizer = embedding_model.tokenizer

# From Reranker
reranker = Candle::Reranker.new(model_path: "cross-encoder/ms-marco-MiniLM-L-12-v2")
rank_tokenizer = reranker.tokenizer
```

### Understanding Subword Tokenization

Modern tokenizers split unknown or rare words into subword pieces:

```ruby
# See how words are split into subwords
result = tokenizer.encode_with_tokens("unbelievable")
# => {"ids" => [101, 4895, 6499, 102], 
#     "tokens" => ["[CLS]", "un", "##believable", "[SEP]"]}

# The ## prefix indicates a continuation of the previous token
complex = tokenizer.encode_to_tokens("preprocessing tokenization")
# => ["[CLS]", "prep", "##ro", "##ces", "##sing", "token", "##ization", "[SEP]"]
```

### Use Cases

- **Token Analysis**: Understand how your text is being processed by models
- **Debugging**: See why certain inputs might cause unexpected model behavior  
- **Custom Preprocessing**: Build your own text processing pipelines
- **Educational**: Teach how modern NLP models handle text
- **NER Preparation**: Get aligned tokens for named entity recognition tasks

## Named Entity Recognition (NER)

Red-Candle includes comprehensive Named Entity Recognition capabilities for extracting entities like people, organizations, locations, and custom entity types from text.

### Model-based NER

Load pre-trained NER models from HuggingFace:

```ruby
require 'candle'

# Load a pre-trained NER model
ner = Candle::NER.from_pretrained("Babelscape/wikineural-multilingual-ner")

# Or load a model with a specific tokenizer (for models without tokenizer.json)
ner = Candle::NER.from_pretrained("dslim/bert-base-NER", tokenizer: "bert-base-cased")

# Extract entities from text
text = "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California."
entities = ner.extract_entities(text)

entities.each do |entity|
  puts "#{entity['text']} (#{entity['label']}) - confidence: #{entity['confidence'].round(2)}"
end
# Output:
# Apple Inc. (ORG) - confidence: 0.99
# Steve Jobs (PER) - confidence: 0.99
# Steve Wozniak (PER) - confidence: 0.98
# Cupertino (LOC) - confidence: 0.97
# California (LOC) - confidence: 0.98

# Adjust confidence threshold (default: 0.9)
entities = ner.extract_entities(text, confidence_threshold: 0.95)

# Get token-level predictions for detailed analysis
tokens = ner.predict_tokens(text)
```

### Pattern-based Recognition

For domain-specific entities, use regex patterns:

```ruby
# Create pattern-based recognizers
email_recognizer = Candle::PatternEntityRecognizer.new("EMAIL", [
  /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/
])

phone_recognizer = Candle::PatternEntityRecognizer.new("PHONE", [
  /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/,         # 555-123-4567
  /\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b/,      # (555) 123-4567
  /\b\+1\s*\d{3}[-.]?\d{3}[-.]?\d{4}\b/   # +1 555-123-4567
])

# Extract entities
text = "Contact us at info@example.com or call 555-123-4567"
email_entities = email_recognizer.recognize(text)
phone_entities = phone_recognizer.recognize(text)
```

### Gazetteer-based Recognition

Use dictionaries for known entities:

```ruby
# Create gazetteer recognizers
companies = ["Apple", "Google", "Microsoft", "Amazon", "Tesla"]
company_recognizer = Candle::GazetteerEntityRecognizer.new("COMPANY", companies)

# Load from file
drug_recognizer = Candle::GazetteerEntityRecognizer.new("DRUG")
drug_recognizer.load_from_file("drug_names.txt")

# Case-sensitive matching
product_recognizer = Candle::GazetteerEntityRecognizer.new("PRODUCT", 
  ["iPhone", "iPad", "MacBook"], 
  case_sensitive: true
)
```

### Hybrid NER

Combine ML models with rule-based approaches for best results:

```ruby
# Create hybrid NER system
hybrid = Candle::HybridNER.new("Babelscape/wikineural-multilingual-ner")

# Add pattern recognizers
hybrid.add_pattern_recognizer("EMAIL", [/\b[\w._%+-]+@[\w.-]+\.[A-Z|a-z]{2,}\b/])
hybrid.add_pattern_recognizer("PHONE", [/\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/])

# Add gazetteer recognizers  
hybrid.add_gazetteer_recognizer("COMPANY", ["Apple", "Google", "Microsoft"])
hybrid.add_gazetteer_recognizer("PRODUCT", ["iPhone", "Android", "Windows"])

# Extract all entities
text = "John Smith (john@apple.com) from Apple called about the new iPhone. Reach him at 555-0123."
entities = hybrid.extract_entities(text)

# Results include entities from all recognizers
# Overlapping entities are automatically resolved (highest confidence wins)
```

### Custom Entity Types

Perfect for specialized domains:

```ruby
# Biomedical entities
gene_patterns = [
  /\b[A-Z][A-Z0-9]{2,}\b/,      # TP53, BRCA1, EGFR
  /\bCD\d+\b/,                  # CD4, CD8, CD34
  /\b[A-Z]+\d[A-Z]\d*\b/        # RAD51C, PALB2
]
gene_recognizer = Candle::PatternEntityRecognizer.new("GENE", gene_patterns)

# Financial entities
ticker_patterns = [
  /\$[A-Z]{1,5}\b/,             # $AAPL, $GOOGL
  /\b[A-Z]{1,5}\.NYSE\b/,       # AAPL.NYSE
  /\b[A-Z]{1,5}\.NASDAQ\b/      # GOOGL.NASDAQ
]
ticker_recognizer = Candle::PatternEntityRecognizer.new("TICKER", ticker_patterns)

# Legal entities
case_patterns = [
  /\b\d+\s+F\.\d+\s+\d+\b/,     # 123 F.3d 456
  /\b\d+\s+U\.S\.\s+\d+\b/,     # 123 U.S. 456
  /\bNo\.\s+\d+-\d+\b/          # No. 20-1234
]
case_recognizer = Candle::PatternEntityRecognizer.new("CASE", case_patterns)
```

### Available Pre-trained Models

Popular NER models on HuggingFace:

```ruby
# General multilingual NER (4 entity types: PER, ORG, LOC, MISC)
ner = Candle::NER.from_pretrained("Babelscape/wikineural-multilingual-ner")

# English NER (requires separate tokenizer)
ner = Candle::NER.from_pretrained("dslim/bert-base-NER", tokenizer: "bert-base-cased")

# Multilingual NER  
ner = Candle::NER.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl")

# OntoNotes 5 (18 entity types including DATE, TIME, MONEY, etc.)
ner = Candle::NER.from_pretrained("flair/ner-english-ontonotes-large")

# Biomedical NER
ner = Candle::NER.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
ner = Candle::NER.from_pretrained("allenai/scibert_scivocab_uncased")
```

### Performance Tips

1. **Device Selection**: Use GPU for faster inference
   ```ruby
   ner = Candle::NER.from_pretrained("Babelscape/wikineural-multilingual-ner", device: Candle::Device.metal)
   ```

2. **Batch Processing**: Process multiple texts together when possible
   
3. **Confidence Threshold**: Balance precision/recall with appropriate thresholds
   
4. **Entity Resolution**: The hybrid NER automatically handles overlapping entities

### Output Format

All NER methods return entities in a consistent format:

```ruby
{
  "text" => "Apple Inc.",          # The entity text
  "label" => "ORG",               # Entity type
  "start" => 0,                   # Character start position
  "end" => 10,                    # Character end position  
  "confidence" => 0.99,           # Confidence score (0-1)
  "token_start" => 0,             # Token start index (model-based only)
  "token_end" => 2,               # Token end index (model-based only)
  "source" => "model"             # Source: "model", "pattern", or "gazetteer"
}
```

## Common Runtime Errors

### Weight is negative, too large or not a valid number

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
llm = Candle::LLM.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", 
                                  device: device, 
                                  gguf_file: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
```

### Cannot find tensor model.embed_tokens.weight

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

### No GGUF file found in repository

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

### Failed to download tokenizer

**Error:**
```
Failed to load quantized model: Failed to download tokenizer: request error: HTTP status client error (404 Not Found)
```

**Cause:** GGUF repositories often don't include separate tokenizer files since they're embedded in the GGUF format.

**Solution:** The code now includes fallback tokenizer loading. If you still encounter this error, ensure you're using the latest version of red-candle.

### Missing metadata in GGUF file

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
git clone https://github.com/assaydepot/red-candle
cd red-candle
bundle
bundle exec rake compile
```

Pull requests are welcome.

## Release

1. Update version number in `lib/candle/version.rb` and commit.
2. `bundle exec rake build`
3. `git tag VERSION_NUMBER`
4. `git push --follow-tags`
5. `gem push pkg/red-candle-VERSION_NUMBER.gem`

## See Also

- [Candle](https://github.com/huggingface/candle)
- [Magnus](https://github.com/matsadler/magnus)
- [Outlines-core](https://github.com/dottxt-ai/outlines-core)
