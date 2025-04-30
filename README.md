# red-candle

[![build](https://github.com/assaydepot/red-candle/actions/workflows/build.yml/badge.svg)](https://github.com/assaydepot/red-candle/actions/workflows/build.yml)
[![Gem Version](https://badge.fury.io/rb/red-candle.svg)](https://badge.fury.io/rb/red-candle)

🕯️ [candle](https://github.com/huggingface/candle) - Minimalist ML framework - for Ruby

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
model = Candle::Model.new2(
  "sentence-transformers/all-MiniLM-L6-v2",  # model path
  "sentence-transformers/all-MiniLM-L6-v2",  # tokenizer path
  nil,                                       # device (nil = CPU)
  Candle::ModelType::STANDARD_BERT           # model type
)
embedding = model.embedding("Hi there!")
```

## Supported Embedding Models

Red-Candle supports the following embedding model types from Hugging Face:

1. `Candle::ModelType::JINA_BERT` - Jina BERT models (e.g., `jinaai/jina-embeddings-v2-base-en`)
2. `Candle::ModelType::STANDARD_BERT` - Standard BERT models (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
3. `Candle::ModelType::SENTIMENT` - Sentiment models (e.g., `distilbert-base-uncased-finetuned-sst-2-english`)
4. `Candle::ModelType::LLAMA` - Llama models (e.g., `meta-llama/Llama-2-7b` - requires Hugging Face token)

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
