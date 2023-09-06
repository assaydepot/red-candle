# ruby-candle

[![build](https://github.com/kojix2/ruby-candle/actions/workflows/build.yml/badge.svg)](https://github.com/kojix2/ruby-candle/actions/workflows/build.yml)

[candle](https://github.com/huggingface/candle) - Minimalist ML framework - for Ruby

## Usage

```ruby
require "candle"

x = Candle::Tensor.new([1, 2, 3, 4, 5, 6])
x = x.reshape([3, 2])
# [[1., 2.],
#  [3., 4.],
#  [5., 6.]]
# Tensor[[3, 2], f32]
```

## Development

Fork it.

Implemented with [Magnus](https://github.com/matsadler/magnus), with reference to [Polars Ruby](https://github.com/ankane/polars-ruby)

Policies
- The less code, the better.
- Ideally, the PyPO3 code should work as is.
- Error handling is minimal.

### See Also

- [Numo::NArray](https://github.com/ruby-numo/numo-narray)
- [Cumo](https://github.com/sonots/cumo)
