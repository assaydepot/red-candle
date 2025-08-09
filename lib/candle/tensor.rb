module Candle
  class Tensor
    include Enumerable

    def each
      case self.rank
      when 0
        # Scalar tensor - yield the single value
        yield self.item
      when 1
        # 1D tensor - yield each value
        # Check if we can use f32 values to avoid conversion
        if dtype.to_s.downcase == "f32"
          begin
            values_f32.each { |value| yield value }
          rescue NoMethodError
            # :nocov:
            # If values_f32 isn't available yet (not recompiled), fall back
            if device.to_s != "cpu"
              # Move to CPU to avoid Metal F32->F64 conversion issue
              to_device(Candle::Device.cpu).values.each { |value| yield value }
            else
              values.each { |value| yield value }
            end
            # :nocov:
          end
        else
          # For non-F32 dtypes, use regular values
          values.each { |value| yield value }
        end
      else
        # Multi-dimensional tensor - yield each sub-tensor
        shape.first.times do |i|
          yield self[i]
        end
      end
    end
    
    # Convert scalar tensor to float
    def to_f
      if rank == 0
        # Use item method which handles dtype conversion properly
        item
      else
        raise ArgumentError, "to_f can only be called on scalar tensors (rank 0), but this tensor has rank #{rank}"
      end
    end
    
    # Convert scalar tensor to integer
    def to_i
      to_f.to_i
    end
    
    # Improved inspect method showing shape, dtype, and device
    def inspect
      shape_str = shape.join("x")
      
      parts = ["#<Candle::Tensor"]
      parts << "shape=#{shape_str}"
      parts << "dtype=#{dtype}"
      parts << "device=#{device}"
      
      # Add element count for clarity
      parts << "elements=#{elem_count}"
      
      parts.join(" ") + ">"
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
