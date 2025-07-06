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
    
    # Override class methods to support keyword arguments for device
    class << self
      alias_method :_original_new, :new
      alias_method :_original_ones, :ones
      alias_method :_original_zeros, :zeros
      alias_method :_original_rand, :rand
      alias_method :_original_randn, :randn
      
      def new(data, dtype = nil, device: nil)
        _original_new(data, dtype, device)
      end
      
      def ones(shape, device: nil)
        _original_ones(shape, device)
      end
      
      def zeros(shape, device: nil)
        _original_zeros(shape, device)
      end
      
      def rand(shape, device: nil)
        _original_rand(shape, device)
      end
      
      def randn(shape, device: nil)
        _original_randn(shape, device)
      end
    end
  end
end
