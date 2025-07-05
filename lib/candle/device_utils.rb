module Candle
  module DeviceUtils
    # Get the best available device (Metal > CUDA > CPU)
    def self.best_device
      # Try devices in order of preference
      begin
        # Try Metal first (for Mac users)
        Device.metal
      rescue
        begin
          # Try CUDA next (for NVIDIA GPU users)
          Device.cuda
        rescue
          # Fall back to CPU
          Device.cpu
        end
      end
    end
    
    # Create a model with automatic device selection
    # This is a convenience method that automatically picks the best device
    def self.create_with_best_device(model_class, **options)
      device = options.delete(:device) || best_device
      
      case model_class.name
      when "Candle::EmbeddingModel"
        model_class.new(**options, device: device)
      when "Candle::Reranker"
        model_class.new(**options, device: device)
      when "Candle::LLM"
        model_id = options.delete(:model_id) || options.delete(:model_path)
        model_class.from_pretrained(model_id, device, **options)
      else
        raise "Unknown model class: #{model_class}"
      end
    end
  end
end