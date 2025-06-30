module Candle
  class Reranker
    # Default model path for cross-encoder/ms-marco-MiniLM-L-12-v2
    DEFAULT_MODEL_PATH = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    # Constructor for creating a new Reranker with optional parameters
    # @param model_path [String, nil] The path to the model on Hugging Face
    # @param cuda [Boolean, nil] Whether to use CUDA for computation (nil = false)
    def self.new(model_path: DEFAULT_MODEL_PATH, cuda: false)
      if cuda
        _create_cuda(model_path)
      else
        _create(model_path)
      end
    end

    # Returns the embedding for a string using the specified pooling method.
    # @param query [String] The input text
    # @param documents [Array<String>] The list of documents to compare against
    # @param pooling_method [String] Pooling method: "pooler", "cls", or "mean". Default: "pooler"
    # @param apply_sigmoid [Boolean] Whether to apply sigmoid to the scores. Default: true
    def rerank(query, documents, pooling_method: "pooler", apply_sigmoid: true)
      rerank_with_options(query, documents, pooling_method, apply_sigmoid)
    end
  end
end
