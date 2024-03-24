# red-candle

[![build](https://github.com/kojix2/red-candle/actions/workflows/build.yml/badge.svg)](https://github.com/kojix2/red-candle/actions/workflows/build.yml)
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
model = Candle::Model.new
embedding = model.embedding("Hi there!")
```

## A note on memory usage
The `Candle::Model` defaults to the `jinaai/jina-embeddings-v2-base-en` model with the `sentence-transformers/all-MiniLM-L6-v2` tokenizer (both from [HuggingFace](https://huggingface.co)). With this configuration the model takes a little more than 3GB of memory running on my Mac. The memory stays with the instantiated `Candle::Model` class, if you instantiate more than one, you'll use more memory. Likewise, if you let it go out of scope and call the garbage collector, you'll free the memory. For example:

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

Policies
- The less code, the better.
- Ideally, the PyPO3 code should work as is.
- Error handling is minimal.

Pull requests are welcome.

kojix2 started this project to learn Rust, but does not necessarily have enough time to maintain this library. If you are interested in becoming a project owner or committer, please send me a pull request.

### See Also

- [Numo::NArray](https://github.com/ruby-numo/numo-narray)
- [Cumo](https://github.com/sonots/cumo)
