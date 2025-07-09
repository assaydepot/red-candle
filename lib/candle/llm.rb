module Candle
  class LLM
    # Simple chat interface for instruction models
    def chat(messages, **options)
      prompt = apply_chat_template(messages)
      generate(prompt, **options)
    end

    # Streaming chat interface
    def chat_stream(messages, **options, &block)
      prompt = apply_chat_template(messages)
      generate_stream(prompt, **options, &block)
    end

    def generate(prompt, config: GenerationConfig.balanced, reset_cache: true)
      begin
        _generate(prompt, config)
      ensure
        clear_cache if reset_cache
      end
    end

    def generate_stream(prompt, config: GenerationConfig.balanced, reset_cache: true, &block)
      begin
        _generate_stream(prompt, config, &block)
      ensure
        clear_cache if reset_cache
      end
    end

    def self.from_pretrained(model_id, device: Candle::Device.cpu)
      _from_pretrained(model_id, device)
    end

    private

    # Legacy format messages method - kept for backward compatibility
    # Use apply_chat_template for proper model-specific formatting
    def format_messages(messages)
      formatted = messages.map do |msg|
        case msg[:role]
        when "system"
          "System: #{msg[:content]}"
        when "user"
          "User: #{msg[:content]}"
        when "assistant"
          "Assistant: #{msg[:content]}"
        else
          msg[:content]
        end
      end.join("\n\n")
      
      # Add a prompt for the assistant to respond
      formatted + "\n\nAssistant:"
    end
  end

  class GenerationConfig
    # Convenience method to create config with overrides
    def with(**overrides)
      current_config = {
        max_length: max_length,
        temperature: temperature,
        top_p: top_p,
        top_k: top_k,
        repetition_penalty: repetition_penalty,
        seed: seed,
        stop_sequences: stop_sequences,
        include_prompt: include_prompt
      }.compact
      
      self.class.new(current_config.merge(overrides))
    end

    # Create a deterministic configuration (temperature = 0, fixed seed)
    def self.deterministic(**opts)
      defaults = {
        temperature: 0.0,
        top_p: nil,
        top_k: 1,
        seed: 42
      }
      new(defaults.merge(opts))
    end

    # Create a creative configuration (higher temperature, random seed)
    def self.creative(**opts)
      defaults = {
        temperature: 1.0,
        top_p: 0.95,
        top_k: 50,
        repetition_penalty: 1.2
      }
      new(defaults.merge(opts))
    end

    # Create a balanced configuration (moderate temperature, random seed)
    def self.balanced(**opts)
      defaults = {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40
      }
      new(defaults.merge(opts))
    end
  end
end