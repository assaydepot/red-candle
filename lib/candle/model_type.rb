module Candle
  # Enum for the supported embedding model types
  module ModelType
    # Jina Bert embedding models (e.g., jina-embeddings-v2-base-en)
    JINA_BERT = "jina_bert"
    
    # Standard BERT embedding models (e.g., bert-base-uncased)
    STANDARD_BERT = "standard_bert"
    
    # MiniLM embedding models (e.g., all-MiniLM-L6-v2)
    MINILM = "minilm"
    
    # DistilBERT models which can be used for embeddings
    DISTILBERT = "distilbert"
    
    # Returns a list of all supported model types
    def self.all
      [JINA_BERT, STANDARD_BERT, DISTILBERT, MINILM]
    end
    
    # Returns suggested model paths for each model type
    def self.suggested_model_paths
      {
        JINA_BERT => "jinaai/jina-embeddings-v2-base-en",
        STANDARD_BERT => "scientistcom/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        MINILM => "sentence-transformers/all-MiniLM-L6-v2",
        DISTILBERT => "distilbert-base-uncased-finetuned-sst-2-english",
      }
    end
  end
end