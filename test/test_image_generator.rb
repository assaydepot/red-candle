require "test_helper"

class TestImageGenerator < Minitest::Test
  def test_image_generation_config_creation
    # Test default config
    config = Candle::ImageGenerationConfig.default
    assert_equal 1024, config.height
    assert_equal 1024, config.width
    assert_equal 28, config.num_inference_steps
    assert_equal 7.0, config.guidance_scale
    assert_nil config.negative_prompt
    assert_nil config.seed
    assert_equal "euler", config.scheduler
    assert config.use_t5
    assert_equal 0, config.clip_skip
  end
  
  def test_image_generation_config_with_custom_values
    config = Candle::ImageGenerationConfig.new(
      height: 512,
      width: 768,
      num_inference_steps: 50,
      guidance_scale: 8.5,
      negative_prompt: "blurry, low quality",
      seed: 42,
      preview_interval: 10,
      scheduler: "ddim",
      use_t5: false,
      clip_skip: 2
    )
    
    assert_equal 512, config.height
    assert_equal 768, config.width
    assert_equal 50, config.num_inference_steps
    assert_equal 8.5, config.guidance_scale
    assert_equal "blurry, low quality", config.negative_prompt
    assert_equal 42, config.seed
    assert_equal 10, config.preview_interval
    assert_equal "ddim", config.scheduler
    refute config.use_t5
    assert_equal 2, config.clip_skip
  end
  
  def test_config_to_h
    config = Candle::ImageGenerationConfig.new(height: 512, width: 512)
    hash = config.to_h
    
    assert_kind_of Hash, hash
    assert_equal 512, hash[:height]
    assert_equal 512, hash[:width]
    assert_includes hash.keys, :guidance_scale
    assert_includes hash.keys, :scheduler
  end
  
  # Skip actual model loading tests for now
  # These will be added once we have proper model loading implemented
  
  def test_placeholder_generation
    skip "Model loading not yet implemented"
    
    # This test demonstrates the intended API
    model = Candle::ImageGenerator.from_pretrained(
      "stabilityai/stable-diffusion-3-medium",
      model_file: "sd3_medium_incl_clips_t5xxlfp8.safetensors"
    )
    
    config = Candle::ImageGenerationConfig.new(
      height: 256,
      width: 256,
      num_inference_steps: 10
    )
    
    image_data = model.generate("A cute robot", config: config)
    assert_kind_of String, image_data
    assert image_data.encoding == Encoding::ASCII_8BIT # Binary data
    
    # PNG signature check
    assert_equal "\x89PNG", image_data[0..3]
  end
  
  def test_streaming_generation
    skip "Model loading not yet implemented"
    
    model = Candle::ImageGenerator.from_pretrained(
      "stabilityai/stable-diffusion-3-medium",
      model_file: "sd3_medium_incl_clips_t5xxlfp8.safetensors"
    )
    
    config = Candle::ImageGenerationConfig.new(
      height: 256,
      width: 256,
      num_inference_steps: 10,
      preview_interval: 2
    )
    
    progress_updates = []
    
    model.generate_stream("A cute robot", config: config) do |progress|
      progress_updates << progress
    end
    
    assert progress_updates.any?
    assert_equal 10, progress_updates.last[:total_steps]
  end
  
  def test_model_preferences
    # Test that model preferences are defined
    assert Candle::ImageGenerator::MODEL_PREFERENCES.key?("stabilityai/stable-diffusion-3-medium")
    assert Candle::ImageGenerator::GGUF_PREFERENCES.key?("second-state/stable-diffusion-3-medium-GGUF")
  end
  
  def test_config_registry
    # Test config registry
    assert Candle::ImageGenerator::CONFIG_REGISTRY.key?("second-state/stable-diffusion-3-medium-GGUF")
    
    # Test registering custom config
    Candle::ImageGenerator.register_config_source("my-model", "my-config")
    assert_equal "my-config", Candle::ImageGenerator::CONFIG_REGISTRY["my-model"]
  end
end