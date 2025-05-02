module Candle
  # Enum for the supported embedding model types
  module ModelType
    # Jina Bert embedding models (e.g., jina-embeddings-v2-base-en)
    JINA_BERT = "jina_bert"
    
    # Standard BERT embedding models (e.g., bert-base-uncased)
    STANDARD_BERT = "standard_bert"
    
    # MiniLM embedding models (e.g., all-MiniLM-L6-v2)
    MINILM = "minilm"
    
    # Sentiment models which can be used for embeddings
    SENTIMENT = "sentiment"
    
    # Llama models which can be used for embeddings
    LLAMA = "llama"
    
    # Returns a list of all supported model types
    def self.all
      [JINA_BERT, STANDARD_BERT, MINILM, SENTIMENT, LLAMA]
    end
    
    # Returns suggested model paths for each model type
    def self.suggested_model_paths
      {
        JINA_BERT => "jinaai/jina-embeddings-v2-base-en",
        STANDARD_BERT => "google-bert/bert-base-uncased",
        MINILM => "sentence-transformers/all-MiniLM-L6-v2",
        SENTIMENT => "distilbert-base-uncased-finetuned-sst-2-english",
        LLAMA => "meta-llama/Llama-2-7b" # Requires Hugging Face token
      }
    end
  end
end