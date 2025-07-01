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

# Default model (JinaBERT)
model = Candle::Model.new
embedding = model.embedding("Hi there!")

# Specify a different model type
model = Candle::Model.new(
  model_path: "sentence-transformers/all-MiniLM-L6-v2",
  tokenizer_path: "sentence-transformers/all-MiniLM-L6-v2",
  device: nil,  # nil = CPU
  model_type: Candle::ModelType::STANDARD_BERT
)
embedding = model.embedding("Hi there!")
```

## ⚠️ Model Format Requirement: Safetensors Only

Red-Candle **only supports embedding models that provide their weights in the [safetensors](https://github.com/huggingface/safetensors) format** (i.e., the model repo must contain a `model.safetensors` file). If the model repo does not provide the required file, loading will fail with a clear error. Most official BERT and DistilBERT models do **not** provide safetensors; many Sentence Transformers and JinaBERT models do.

**If you encounter an error like:**

```
RuntimeError: model.safetensors not found after download. Only safetensors models are supported. Please ensure your model repo contains model.safetensors.
```

this means the selected model is not compatible. Please choose a model repo that provides the required file.

## Supported Embedding Models

Red-Candle supports the following embedding model types from Hugging Face:

1. `Candle::ModelType::JINA_BERT` - Jina BERT models (e.g., `jinaai/jina-embeddings-v2-base-en`) (**safetensors required**)
2. `Candle::ModelType::STANDARD_BERT` - Standard BERT models (e.g., `sentence-transformers/all-MiniLM-L6-v2`) (**safetensors required**)
3. `Candle::ModelType::DISTILBERT` - DistilBERT models (e.g., `distilbert-base-uncased-finetuned-sst-2-english`) (**safetensors required**)

> **Note:** Most official BERT and DistilBERT models do _not_ provide safetensors. Please check the model repo before use.

You can get a list of all supported model types and suggested models paths:

```ruby
Candle::ModelType.all  # Returns all supported model types
Candle::ModelType.suggested_model_paths  # Returns hash of suggested models for each type
```

## A note on memory usage
The default model (`jinaai/jina-embeddings-v2-base-en` with the `sentence-transformers/all-MiniLM-L6-v2` tokenizer, both from [HuggingFace](https://huggingface.co)) takes a little more than 3GB of memory running on a Mac. The memory stays with the instantiated `Candle::Model` class, if you instantiate more than one, you'll use more memory. Likewise, if you let it go out of scope and call the garbage collector, you'll free the memory. For example:

```ruby
> require 'candle'
# Ruby memory = 25.9 MB
> model = Candle::Model.new
# Ruby memory = 3.50 GB
> model2 = Candle::Model.new
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
model = Candle::Model.new
embedding = model.embedding("Hi there!")
```

## Document Reranking

Red-Candle includes support for cross-encoder reranking models, which can be used to reorder documents by relevance to a query. This is particularly useful for improving search results or implementing retrieval-augmented generation (RAG) systems.

### Basic Usage

```ruby
require 'candle'

# Initialize the reranker with a cross-encoder model
reranker = Candle::Reranker.new("cross-encoder/ms-marco-MiniLM-L-12-v2")

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
# [{:doc_id=>1, :score=>0.9999693632125854, :text=>"Around 9 Million people live in London"},
#  {:doc_id=>3, :score=>0.04381871595978737, :text=>"London is the capital of England"},
#  {:doc_id=>0, :score=>0.008476530201733112, :text=>"London is known for its financial district"},
#  {:doc_id=>2, :score=>0.0005390917649492621, :text=>"The weather in London is often rainy"}]
```

### Arguments & Activation Functions

By default, `apply_sigmoid` is `true` (scores between 0 and 1). Set it to `false` to get raw logits. You can also select the pooling method:

- `pooling_method: "pooler"` (default)
- `pooling_method: "cls"`
- `pooling_method: "mean"`

Example with sigmoid activation:

```ruby
# Get sigmoid-activated scores (0 to 1 range)
ranked_results = reranker.rerank(query, documents, apply_sigmoid: true)

ranked_results.each do |result|
  puts "Score: #{result[:score].round(4)} - Doc ##{result[:doc_id]}: #{result[:text]}"
end
# Output:
# Score: 1.0000 - Doc #1: Around 9 Million people live in London
# Score: 0.0438 - Doc #3: London is the capital of England
# Score: 0.0085 - Doc #0: London is known for its financial district
# Score: 0.0005 - Doc #2: The weather in London is often rainy
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
reranker = Candle::Reranker.new_cuda("cross-encoder/ms-marco-MiniLM-L-12-v2")
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
