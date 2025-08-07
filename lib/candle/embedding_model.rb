module Candle
  class EmbeddingModel
    # Default model path for Jina BERT embedding model
    DEFAULT_MODEL_PATH = "jinaai/jina-embeddings-v2-base-en"
    
    # Default tokenizer path that works well with the default model
    DEFAULT_TOKENIZER_PATH = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Default embedding model type
    DEFAULT_EMBEDDING_MODEL_TYPE = "jina_bert"
    
    # Constructor for creating a new EmbeddingModel with optional parameters
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
      _create(model_path, tokenizer_path, device, model_type, embedding_size)
    end
    # Returns the embedding for a string using the specified pooling method.
    # @param str [String] The input text
    # @param pooling_method [String] Pooling method: "pooled", "pooled_normalized", or "cls". Default: "pooled_normalized"
    def embedding(str, pooling_method: "pooled_normalized")
      _embedding(str, pooling_method)
    end
  end
end
