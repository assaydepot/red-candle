# Custom RSpec matchers for ML testing

RSpec::Matchers.define :be_a_tensor do
  match do |actual|
    actual.is_a?(Candle::Tensor)
  end
  
  description do
    "be a Candle::Tensor"
  end
end

RSpec::Matchers.define :have_shape do |expected|
  match do |actual|
    actual.is_a?(Candle::Tensor) && actual.shape == expected
  end
  
  failure_message do |actual|
    if actual.is_a?(Candle::Tensor)
      "expected tensor to have shape #{expected}, got #{actual.shape}"
    else
      "expected a Candle::Tensor, got #{actual.class}"
    end
  end
  
  description do
    "have shape #{expected}"
  end
end

RSpec::Matchers.define :have_dtype do |expected|
  match do |actual|
    actual.is_a?(Candle::Tensor) && actual.dtype == expected
  end
  
  failure_message do |actual|
    if actual.is_a?(Candle::Tensor)
      "expected tensor to have dtype #{expected}, got #{actual.dtype}"
    else
      "expected a Candle::Tensor, got #{actual.class}"
    end
  end
  
  description do
    "have dtype #{expected}"
  end
end

RSpec::Matchers.define :be_on_device do |expected|
  match do |actual|
    if actual.is_a?(Candle::Tensor)
      actual.device.to_s.include?(expected.to_s)
    elsif actual.respond_to?(:device)
      actual.device.to_s.include?(expected.to_s)
    else
      false
    end
  end
  
  failure_message do |actual|
    if actual.respond_to?(:device)
      "expected to be on device #{expected}, but was on #{actual.device}"
    else
      "expected #{actual.class} to respond to :device"
    end
  end
  
  description do
    "be on device #{expected}"
  end
end

RSpec::Matchers.define :generate_text do
  match do |actual|
    begin
      result = actual.generate("Hello", max_length: 10)
      result.is_a?(String) && result.length > 5
    rescue
      false
    end
  end
  
  description do
    "be able to generate text"
  end
end

RSpec::Matchers.define :extract_entities do
  match do |actual|
    begin
      result = actual.extract_entities("Apple Inc. was founded by Steve Jobs.")
      result.is_a?(Array) && result.any? { |e| e[:label] && e[:text] }
    rescue
      false
    end
  end
  
  description do
    "be able to extract entities"
  end
end

RSpec::Matchers.define :compute_embeddings do
  match do |actual|
    begin
      result = actual.embedding("test text")
      result.is_a?(Candle::Tensor) && result.shape.length == 2
    rescue
      false
    end
  end
  
  description do
    "be able to compute embeddings"
  end
end

# Matcher for approximate equality (useful for floating point comparisons)
RSpec::Matchers.define :be_close_to_tensor do |expected, tolerance = 1e-6|
  match do |actual|
    return false unless actual.is_a?(Candle::Tensor) && expected.is_a?(Candle::Tensor)
    return false unless actual.shape == expected.shape
    
    diff = (actual - expected).abs
    max_diff = diff.max.to_f
    max_diff <= tolerance
  end
  
  failure_message do |actual|
    if actual.is_a?(Candle::Tensor) && expected.is_a?(Candle::Tensor)
      diff = (actual - expected).abs
      max_diff = diff.max.to_f
      "expected tensors to be within #{tolerance}, but maximum difference was #{max_diff}"
    else
      "expected both values to be Candle::Tensor"
    end
  end
  
  description do
    "be close to the expected tensor within tolerance #{tolerance}"
  end
end