module Candle
  class Model
    # Default model path for Jina BERT embedding model
    DEFAULT_MODEL_PATH = "jinaai/jina-embeddings-v2-base-en"
    
    # Default tokenizer path that works well with the default model
    DEFAULT_TOKENIZER_PATH = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Default model type
    DEFAULT_MODEL_TYPE = "jina_bert"
    
    # Constructor for creating a new Model with optional parameters
    # @param model_path [String, nil] The path to the model on Hugging Face
    # @param tokenizer_path [String, nil] The path to the tokenizer on Hugging Face
    # @param device [Candle::Device, nil] The device to use for computation (nil = CPU)
    # @param model_type [String, nil] The type of model to use
    # @param embedding_size [Integer, nil] Override for the embedding size (optional)
    def self.new(model_path: DEFAULT_MODEL_PATH,
      tokenizer_path: DEFAULT_TOKENIZER_PATH,
      device: nil,
      model_type: DEFAULT_MODEL_TYPE,
      embedding_size: nil)
      _create(model_path, tokenizer_path, device, model_type, embedding_size)
    end
  end
end
