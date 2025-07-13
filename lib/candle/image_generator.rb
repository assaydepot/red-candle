# frozen_string_literal: true

module Candle
  class ImageGenerator
    # Model file preferences for automatic selection
    MODEL_PREFERENCES = {
      "stabilityai/stable-diffusion-3-medium" => [
        "sd3_medium_incl_clips_t5xxlfp8.safetensors",    # Default: smallest complete
        "sd3_medium_incl_clips_t5xxlfp16.safetensors",   # Higher precision
        "sd3_medium_incl_clips.safetensors",             # Needs T5
        "sd3_medium.safetensors"                         # Needs CLIP + T5
      ],
      "stabilityai/stable-diffusion-3.5-large" => [
        "sd3.5_large_incl_clips_t5xxlfp8.safetensors",
        "sd3.5_large_incl_clips_t5xxlfp16.safetensors",
        "sd3.5_large_incl_clips.safetensors",
        "sd3.5_large.safetensors"
      ]
    }.freeze

    GGUF_PREFERENCES = {
      "second-state/stable-diffusion-3-medium-GGUF" => [
        "sd3-medium-Q5_0.gguf",    # 5-bit, ~5.53 GB (good balance)
        "sd3-medium-Q4_0.gguf",    # 4-bit, ~4.55 GB (smallest)
        "sd3-medium-Q8_0.gguf",    # 8-bit, ~8.45 GB (high quality)
        "sd3-medium-f16.gguf"      # fp16, ~15.8 GB (best quality)
      ]
    }.freeze

    # Registry for config sources (for GGUF models that need external configs)
    CONFIG_REGISTRY = {
      "second-state/stable-diffusion-3-medium-GGUF" => "stabilityai/stable-diffusion-3-medium"
    }

    class << self
      # Load a pretrained image generation model
      def from_pretrained(model_id, **kwargs)
        # Handle different argument patterns
        device = kwargs[:device]
        model_file = kwargs[:model_file]
        gguf_file = kwargs[:gguf_file]
        clip_model = kwargs[:clip_model]
        t5_model = kwargs[:t5_model]
        config_source = kwargs[:config_source]
        tokenizer_source = kwargs[:tokenizer_source]

        # Auto-detect config source for known GGUF models
        if gguf_file || model_id.include?("GGUF")
          config_source ||= CONFIG_REGISTRY[model_id]
        end

        # Call the Rust implementation
        _from_pretrained(
          model_id,
          device: device,
          model_file: model_file,
          gguf_file: gguf_file,
          clip_model: clip_model,
          t5_model: t5_model,
          config_source: config_source,
          tokenizer_source: tokenizer_source
        )
      end

      # Register a config source for a custom model
      def register_config_source(model_id, config_source)
        CONFIG_REGISTRY[model_id] = config_source
      end
    end

    # Generate an image from a prompt
    def generate(prompt, config: nil)
      config ||= ImageGenerationConfig.default
      image_data = _generate(prompt, config)
      
      # Return raw PNG bytes
      # Users can do: File.binwrite("output.png", image_data)
      # Or use ChunkyPNG: ChunkyPNG::Image.from_blob(image_data)
      image_data
    end

    # Generate an image with streaming progress updates
    def generate_stream(prompt, config: nil, &block)
      raise ArgumentError, "Block required for streaming generation" unless block_given?
      
      config ||= ImageGenerationConfig.default
      _generate_stream(prompt, config, &block)
    end

    # Convenience method to save generated image
    def generate_and_save(prompt, filename, config: nil)
      image_data = generate(prompt, config: config)
      File.binwrite(filename, image_data)
      filename
    end

    # Convenience method for streaming with automatic file saves
    def generate_stream_with_saves(prompt, base_filename, config: nil, &block)
      config ||= ImageGenerationConfig.default
      
      generate_stream(prompt, config: config) do |progress|
        # Save intermediate images if provided
        if progress[:image_data]
          step = progress[:step]
          filename = base_filename.sub(/\.png$/, "_step_#{step}.png")
          File.binwrite(filename, progress[:image_data])
          progress[:saved_to] = filename
        end
        
        # Call user's block with enhanced progress
        block.call(progress) if block
      end
    end
  end

  class ImageGenerationConfig
    # Provide Ruby-friendly accessors
    alias_method :negative_prompt?, :negative_prompt
    alias_method :use_t5?, :use_t5
    
    def to_h
      {
        height: height,
        width: width,
        num_inference_steps: num_inference_steps,
        guidance_scale: guidance_scale,
        negative_prompt: negative_prompt,
        seed: seed,
        preview_interval: preview_interval,
        scheduler: scheduler,
        use_t5: use_t5,
        clip_skip: clip_skip
      }
    end
    
    def inspect
      "#<Candle::ImageGenerationConfig:0x#{object_id.to_s(16)} #{to_h}>"
    end
  end
end