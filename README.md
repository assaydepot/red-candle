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
model.embedding("Hi there!")
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
