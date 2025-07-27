require 'json'

module Candle
  class LLM
    # Create a structured constraint from a JSON schema
    def constraint_from_schema(schema)
      schema_str = schema.is_a?(String) ? schema : JSON.generate(schema)
      StructuredConstraint.from_schema(schema_str, tokenizer)
    end
    
    # Create a structured constraint from a regex pattern
    def constraint_from_regex(pattern)
      pattern_str = pattern.is_a?(Regexp) ? pattern.source : pattern.to_s
      StructuredConstraint.from_regex(pattern_str, tokenizer)
    end
    
    # Generate and parse structured output from a JSON schema
    def generate_structured(prompt, schema:, **options)
      constraint = constraint_from_schema(schema)
      config_opts = options.merge(constraint: constraint)
      config = options[:config] || GenerationConfig.balanced(**config_opts)
      
      result = generate(prompt, config: config, reset_cache: options.fetch(:reset_cache, true))
      
      # Try to parse as JSON
      begin
        JSON.parse(result)
      rescue JSON::ParserError => e
        # Return the raw string if parsing fails
        warn "Warning: Generated output is not valid JSON: #{e.message}" if options[:warn_on_parse_error]
        result
      end
    end
    # Tokenizer registry for automatic detection
    TOKENIZER_REGISTRY = {
      # Exact model matches
      "TheBloke/Mistral-7B-Instruct-v0.2-GGUF" => "mistralai/Mistral-7B-Instruct-v0.2",
      "TheBloke/Mistral-7B-v0.1-GGUF" => "mistralai/Mistral-7B-v0.1",
      "TheBloke/Llama-2-7B-Chat-GGUF" => "meta-llama/Llama-2-7b-chat-hf",
      "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" => "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
      
      # Qwen official GGUF models
      "Qwen/Qwen3-8B-GGUF" => "Qwen/Qwen3-8B",
      "Qwen/Qwen3-4B-GGUF" => "Qwen/Qwen3-4B",
      "Qwen/Qwen3-14B-GGUF" => "Qwen/Qwen3-14B",
      "Qwen/Qwen3-32B-GGUF" => "Qwen/Qwen3-32B",
      "Qwen/Qwen3-72B-GGUF" => "Qwen/Qwen3-72B",
      
      # Phi GGUF models
      "TheBloke/phi-2-GGUF" => "microsoft/phi-2",
      "microsoft/phi-4-gguf" => "microsoft/phi-4",
      "bartowski/Phi-3.5-mini-instruct-GGUF" => "microsoft/Phi-3.5-mini-instruct",
      
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
        [/qwen.*?3.*?72b/i, "Qwen/Qwen3-72B"],
        [/qwen.*?3.*?32b/i, "Qwen/Qwen3-32B"],
        [/qwen.*?3.*?14b/i, "Qwen/Qwen3-14B"],
        [/qwen.*?3.*?8b/i, "Qwen/Qwen3-8B"],
        [/qwen.*?3.*?4b/i, "Qwen/Qwen3-4B"],
        [/qwen.*?3.*?1\.8b/i, "Qwen/Qwen3-1.8B"],
        [/qwen.*?3.*?0\.5b/i, "Qwen/Qwen3-0.5B"],
        [/qwen.*?2\.5/i, "Qwen/Qwen2.5-0.5B"],
        [/qwen.*?2/i, "Qwen/Qwen2-1.5B"],
        [/qwen/i, "Qwen/Qwen-1_8B"],
        
        # Phi models (order matters - more specific patterns first)
        [/phi.*?3\.5.*?mini/i, "microsoft/Phi-3.5-mini-instruct"],
        [/phi.*?3.*?mini.*?4k/i, "microsoft/Phi-3-mini-4k-instruct"],
        [/phi.*?3.*?medium/i, "microsoft/Phi-3-medium-4k-instruct"],
        [/phi.*?3.*?small/i, "microsoft/Phi-3-small-8k-instruct"],
        [/phi.*?3.*?mini/i, "microsoft/Phi-3-mini-4k-instruct"],
        [/phi.*?3/i, "microsoft/Phi-3-mini-4k-instruct"],
        [/phi-4/i, "microsoft/phi-4"],
        [/phi.*?2/i, "microsoft/phi-2"],
        [/phi.*?1\.5/i, "microsoft/phi-1_5"],
        [/phi/i, "microsoft/phi-2"]
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
        include_prompt: include_prompt,
        constraint: defined?(@constraint) ? @constraint : nil
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