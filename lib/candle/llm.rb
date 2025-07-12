module Candle
  class LLM
    # Tokenizer registry for automatic detection
    TOKENIZER_REGISTRY = {
      # Exact model matches
      "TheBloke/Mistral-7B-Instruct-v0.2-GGUF" => "mistralai/Mistral-7B-Instruct-v0.2",
      "TheBloke/Mistral-7B-v0.1-GGUF" => "mistralai/Mistral-7B-v0.1",
      "TheBloke/Llama-2-7B-Chat-GGUF" => "meta-llama/Llama-2-7b-chat-hf",
      "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" => "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
      
      # Pattern-based fallbacks (evaluated in order)
      :patterns => [
        # Mistral models
        [/mistral.*?7b.*?instruct.*?v0\.2/i, "mistralai/Mistral-7B-Instruct-v0.2"],
        [/mistral.*?7b.*?instruct.*?v0\.1/i, "mistralai/Mistral-7B-Instruct-v0.1"],
        [/mistral.*?7b/i, "mistralai/Mistral-7B-v0.1"],
        
        # Llama models
        [/llama.*?3.*?8b/i, "meta-llama/Meta-Llama-3-8B"],
        [/llama.*?3.*?70b/i, "meta-llama/Meta-Llama-3-70B"],
        [/llama.*?2.*?7b.*?chat/i, "meta-llama/Llama-2-7b-chat-hf"],
        [/llama.*?2.*?13b.*?chat/i, "meta-llama/Llama-2-13b-chat-hf"],
        [/llama.*?2.*?70b.*?chat/i, "meta-llama/Llama-2-70b-chat-hf"],
        [/tinyllama/i, "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
        
        # Gemma models
        [/gemma.*?2.*?9b/i, "google/gemma-2-9b"],
        [/gemma.*?2.*?2b/i, "google/gemma-2-2b"],
        [/gemma.*?7b/i, "google/gemma-7b"],
        [/gemma.*?2b/i, "google/gemma-2b"],
        
        # Qwen models
        [/qwen.*?2\.5.*?7b/i, "Qwen/Qwen2.5-7B"],
        [/qwen.*?2\.5.*?0\.5b/i, "Qwen/Qwen2.5-0.5B"],
        [/qwen.*?2.*?7b/i, "Qwen/Qwen2-7B"],
        [/qwen3/i, "Qwen/Qwen2.5-7B"], # Qwen3 uses Qwen2.5 tokenizer
      ]
    }
    
    # Allow users to register custom tokenizer mappings
    def self.register_tokenizer(model_pattern, tokenizer_id)
      if model_pattern.is_a?(String)
        TOKENIZER_REGISTRY[model_pattern] = tokenizer_id
      elsif model_pattern.is_a?(Regexp)
        TOKENIZER_REGISTRY[:patterns] ||= []
        TOKENIZER_REGISTRY[:patterns].unshift([model_pattern, tokenizer_id])
      else
        raise ArgumentError, "model_pattern must be a String or Regexp"
      end
    end
    
    # Guess the tokenizer for a model
    def self.guess_tokenizer(model_id)
      # Check exact matches first
      return TOKENIZER_REGISTRY[model_id] if TOKENIZER_REGISTRY[model_id]
      
      # Check patterns
      if patterns = TOKENIZER_REGISTRY[:patterns]
        patterns.each do |pattern, tokenizer|
          return tokenizer if model_id.match?(pattern)
        end
      end
      
      # Default: try removing common GGUF suffixes
      base_model = model_id.gsub(/-gguf|-q\d+_\w+$/i, "")
      base_model
    end
    
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

    def self.from_pretrained(model_id, device: Candle::Device.cpu, gguf_file: nil, tokenizer: nil)
      model_str = if gguf_file
        "#{model_id}@#{gguf_file}"
      else
        model_id
      end
      
      # Handle GGUF models that need tokenizer
      if model_str.downcase.include?("gguf") && tokenizer.nil?
        # Try to load without tokenizer first
        begin
          _from_pretrained(model_str, device)
        rescue => e
          if e.message.include?("No tokenizer found")
            # Auto-detect tokenizer
            detected_tokenizer = guess_tokenizer(model_id)
            warn "No tokenizer found in GGUF repo. Using tokenizer from: #{detected_tokenizer}"
            model_str = "#{model_str}@@#{detected_tokenizer}"
            _from_pretrained(model_str, device)
          else
            raise e
          end
        end
      elsif tokenizer
        # User specified tokenizer
        model_str = "#{model_str}@@#{tokenizer}"
        _from_pretrained(model_str, device)
      else
        # Non-GGUF model or GGUF with embedded tokenizer
        _from_pretrained(model_str, device)
      end
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