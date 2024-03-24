module Candle
  class Tensor
    include Enumerable

    def each
      if self.rank == 1
        self.values.each do |value|
          yield value
        end
      else
        shape.first.times do |i|
          yield self[i]
        end
      end
    end
  end
end
