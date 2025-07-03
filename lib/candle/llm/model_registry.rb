module Candle
  class LLM
    class ModelRegistry
      @models = {}
      @patterns = []

      class << self
        # Register a model pattern
        def register(pattern, model_info)
          @patterns << [pattern, model_info]
          @patterns.sort_by! { |p, _| -p.to_s.length } # Sort by specificity
        end

        # Get model info for a given model ID
        def model_info(model_id)
          @patterns.each do |pattern, info|
            return info if model_id.match?(pattern)
          end
          nil
        end

        # List all registered model patterns
        def registered_models
          @patterns.map { |pattern, info| { pattern: pattern, **info } }
        end

        # Check if a model is supported
        def supported?(model_id)
          !model_info(model_id).nil?
        end
      end
    end

    # Pre-register known models
    ModelRegistry.register(
      /mistral.*7b.*instruct.*v0\.[123]/i,
      {
        name: "Mistral 7B Instruct",
        type: :mistral,
        size: "7B",
        context_length: 32768,
        supports_chat: true,
        notes: "Uses sharded safetensors format"
      }
    )
    
    ModelRegistry.register(
      /mistral.*7b.*instruct/i,
      {
        name: "Mistral 7B Instruct",
        type: :mistral,
        size: "7B",
        context_length: 32768,
        supports_chat: true
      }
    )

    ModelRegistry.register(
      /mistral.*7b/i,
      {
        name: "Mistral 7B",
        type: :mistral,
        size: "7B",
        context_length: 32768,
        supports_chat: false
      }
    )

    ModelRegistry.register(
      /mistral.*nemo/i,
      {
        name: "Mistral Nemo",
        type: :mistral,
        size: "12B",
        context_length: 128000,
        supports_chat: true
      }
    )

    # Future model registrations (not yet implemented)
    ModelRegistry.register(
      /gemma.*2b/i,
      {
        name: "Gemma 2B",
        type: :gemma,
        size: "2B",
        context_length: 8192,
        supports_chat: true,
        status: :planned
      }
    )

    ModelRegistry.register(
      /deepseek/i,
      {
        name: "DeepSeek",
        type: :deepseek,
        size: "Various",
        context_length: 16384,
        supports_chat: true,
        status: :planned
      }
    )
  end
end