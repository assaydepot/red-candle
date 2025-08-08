module Candle
  class EmbeddingModel
    # Default model path for Jina BERT embedding model
    DEFAULT_MODEL_PATH = "jinaai/jina-embeddings-v2-base-en"
    
    # Default tokenizer path that works well with the default model
    DEFAULT_TOKENIZER_PATH = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Default embedding model type
    DEFAULT_EMBEDDING_MODEL_TYPE = "jina_bert"
    
    # Load a pre-trained embedding model from HuggingFace
    # @param model_id [String] HuggingFace model ID (defaults to jinaai/jina-embeddings-v2-base-en)
    # @param device [Candle::Device] The device to use for computation (defaults to best available)
    # @param tokenizer [String, nil] The tokenizer to use (defaults to using the model's tokenizer)
    # @param model_type [String, nil] The type of embedding model (auto-detected if nil)
    # @param embedding_size [Integer, nil] Override for the embedding size (optional)
    # @return [EmbeddingModel] A new EmbeddingModel instance
    def self.from_pretrained(model_id = DEFAULT_MODEL_PATH, device: Candle::Device.best, tokenizer: nil, model_type: nil, embedding_size: nil)
      # Auto-detect model type based on model_id if not provided
      if model_type.nil?
        model_type = case model_id.downcase
        when /jina/
          "jina_bert"
        when /distilbert/
          "distilbert"
        when /minilm/
          "minilm"
        else
          "standard_bert"
        end
      end
      
      # Use model_id as tokenizer if not specified (usually what you want)
      tokenizer_id = tokenizer || model_id
      
      _create(model_id, tokenizer_id, device, model_type, embedding_size)
    end
    
    # Constructor for creating a new EmbeddingModel with optional parameters
    # @deprecated Use {.from_pretrained} instead
    # @param model_path [String, nil] The path to the model on Hugging Face
    # @param tokenizer_path [String, nil] The path to the tokenizer on Hugging Face
    # @param device [Candle::Device, Candle::Device.cpu] The device to use for computation
    # @param model_type [String, nil] The type of embedding model to use
    # @param embedding_size [Integer, nil] Override for the embedding size (optional)
    def self.new(model_path: DEFAULT_MODEL_PATH,
      tokenizer_path: DEFAULT_TOKENIZER_PATH,
      device: Candle::Device.best,
      model_type: DEFAULT_EMBEDDING_MODEL_TYPE,
      embedding_size: nil)
      $stderr.puts "[DEPRECATION] `EmbeddingModel.new` is deprecated. Please use `EmbeddingModel.from_pretrained` instead."
      _create(model_path, tokenizer_path, device, model_type, embedding_size)
    end
    # Returns the embedding for a string using the specified pooling method.
    # @param str [String] The input text
    # @param pooling_method [String] Pooling method: "pooled", "pooled_normalized", or "cls". Default: "pooled_normalized"
    def embedding(str, pooling_method: "pooled_normalized")
      _embedding(str, pooling_method)
    end
    
    # Improved inspect method
    def inspect
      opts = options rescue {}
      
      parts = ["#<Candle::EmbeddingModel"]
      parts << "model=#{opts["model_id"] || "unknown"}"
      parts << "type=#{opts["embedding_model_type"]}" if opts["embedding_model_type"]
      parts << "device=#{opts["device"] || "unknown"}"
      parts << "size=#{opts["embedding_size"]}" if opts["embedding_size"]
      
      parts.join(" ") + ">"
    end
  end
end
