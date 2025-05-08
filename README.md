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

## ⚠️ Model Format Requirement: Safetensors Only (GGML for Llama)

Red-Candle **only supports embedding models that provide their weights in the [safetensors](https://github.com/huggingface/safetensors) format** (i.e., the model repo must contain a `model.safetensors` file), **except for Llama models, which must provide a `model.ggml` file**. If the model repo does not provide the required file, loading will fail with a clear error. Most official BERT and DistilBERT models do **not** provide safetensors; many Sentence Transformers and JinaBERT models do. Llama models are only supported in GGML format (not safetensors or bin).

**If you encounter an error like:**

```
RuntimeError: model.safetensors not found after download. Only safetensors models are supported. Please ensure your model repo contains model.safetensors.
```

or

```
RuntimeError: model.ggml not found after download. Only GGML format is supported for Llama models. Please ensure your model repo contains model.ggml.
```

this means the selected model is not compatible. Please choose a model repo that provides the required file.

## Supported Embedding Models

Red-Candle supports the following embedding model types from Hugging Face:

1. `Candle::ModelType::JINA_BERT` - Jina BERT models (e.g., `jinaai/jina-embeddings-v2-base-en`) (**safetensors required**)
2. `Candle::ModelType::STANDARD_BERT` - Standard BERT models (e.g., `sentence-transformers/all-MiniLM-L6-v2`) (**safetensors required**)
3. `Candle::ModelType::DISTILBERT` - DistilBERT models (e.g., `distilbert-base-uncased-finetuned-sst-2-english`) (**safetensors required**)
4. `Candle::ModelType::LLAMA` - Llama models (e.g., `meta-llama/Llama-2-7b` - requires Hugging Face token, **GGML required**)

> **Note:** Most official BERT and DistilBERT models do _not_ provide safetensors. Llama models must be in GGML format. Please check the model repo before use.

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
