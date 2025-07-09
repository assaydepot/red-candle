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
  end
end