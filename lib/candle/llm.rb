module Candle
  class LLM
    # Simple chat interface for instruction models
    def chat(messages, **options)
      prompt = format_messages(messages)
      generate(prompt, **options)
    end

    # Streaming chat interface
    def chat_stream(messages, **options, &block)
      prompt = format_messages(messages)
      generate_stream(prompt, **options, &block)
    end

    def generate(prompt, config: GenerationConfig.balanced)
      _generate(prompt, config)
    end

    def generate_stream(prompt, config: GenerationConfig.balanced, &block)
      _generate_stream(prompt, config, &block)
    end

    def self.from_pretrained(model_id, device: Candle::Device.cpu)
      _from_pretrained(model_id, device)
    end

    private

    # Format messages into a prompt string
    # This is a simple implementation - model-specific formatting should be added
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
  end
end