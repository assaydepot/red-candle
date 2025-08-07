module Candle
  module DeviceUtils
    # @deprecated Use {Device.best} instead
    # Get the best available device (Metal > CUDA > CPU)
    def self.best_device
      warn "[DEPRECATION] `DeviceUtils.best_device` is deprecated. Please use `Device.best` instead."
      Device.best
    end
  end
end