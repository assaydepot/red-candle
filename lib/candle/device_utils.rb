module Candle
  module DeviceUtils
    # Check if a device supports all required operations for a model type
    def self.supports_model?(device, model_type)
      # All models now work on Metal with candle 0.9.1!
      true
    end
    
    # Get the best available device for a model type
    def self.best_device_for(model_type, preferred_device = nil)
      if preferred_device
        if supports_model?(preferred_device, model_type)
          preferred_device
        else
          warn "Device #{preferred_device.inspect} doesn't support #{model_type}, falling back to CPU"
          Device.cpu
        end
      else
        # Try devices in order of preference
        begin
          if supports_model?(Device.metal, model_type)
            Device.metal
          elsif (Device.cuda rescue nil) && supports_model?(Device.cuda, model_type)
            Device.cuda
          else
            Device.cpu
          end
        rescue
          Device.cpu
        end
      end
    end
    
    # Create a model with automatic device selection
    def self.create_with_best_device(model_class, model_type, **options)
      device = options.delete(:device)
      best_device = best_device_for(model_type, device)
      
      case model_class.name
      when "Candle::EmbeddingModel"
        model_class.new(**options, device: best_device)
      when "Candle::Reranker"
        model_class.new(**options, device: best_device)
      when "Candle::LLM"
        model_id = options.delete(:model_id) || options.delete(:model_path)
        model_class.from_pretrained(model_id, best_device, **options)
      else
        raise "Unknown model class: #{model_class}"
      end
    end
  end
end