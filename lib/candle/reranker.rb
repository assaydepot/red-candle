module Candle
  class Reranker
    # Default model path for cross-encoder/ms-marco-MiniLM-L-12-v2
    DEFAULT_MODEL_PATH = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    # Load a pre-trained reranker model from HuggingFace
    # @param model_id [String] HuggingFace model ID (defaults to cross-encoder/ms-marco-MiniLM-L-12-v2)
    # @param device [Candle::Device] The device to use for computation (defaults to best available)
    # @return [Reranker] A new Reranker instance
    def self.from_pretrained(model_id = DEFAULT_MODEL_PATH, device: Candle::Device.best)
      _create(model_id, device)
    end
    
    # Constructor for creating a new Reranker with optional parameters
    # @deprecated Use {.from_pretrained} instead
    # @param model_path [String, nil] The path to the model on Hugging Face
    # @param device [Candle::Device, Candle::Device.cpu] The device to use for computation
    def self.new(model_path: DEFAULT_MODEL_PATH, device: Candle::Device.best)
      $stderr.puts "[DEPRECATION] `Reranker.new` is deprecated. Please use `Reranker.from_pretrained` instead."
      _create(model_path, device)
    end

    # Returns documents ranked by relevance using the specified pooling method.
    # @param query [String] The input text
    # @param documents [Array<String>] The list of documents to compare against
    # @param pooling_method [String] Pooling method: "pooler", "cls", or "mean". Default: "pooler"
    # @param apply_sigmoid [Boolean] Whether to apply sigmoid to the scores. Default: true
    def rerank(query, documents, pooling_method: "pooler", apply_sigmoid: true)
      rerank_with_options(query, documents, pooling_method, apply_sigmoid).collect { |doc, score, doc_id|
        { doc_id: doc_id, score: score, text: doc }
      }
    end
    
    # Improved inspect method
    def inspect
      opts = options rescue {}
      parts = ["#<Candle::Reranker"]
      parts << "model=#{opts["model_id"] || "unknown"}"
      parts << "device=#{opts["device"] || "unknown"}"
      parts.join(" ") + ">"
    end
  end
end
