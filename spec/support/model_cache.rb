require 'singleton'

# Singleton cache for expensive model loading
# Models are loaded once and shared across all specs
class ModelCache
  include Singleton
  
  attr_reader :models, :load_errors
  
  def initialize
    @models = {}
    @load_errors = {}
    @mutex = Mutex.new
  end
  
  # Preload commonly used models
  def preload_models
    # These will be loaded on first access
    # We could eagerly load them here if desired
  end
  
  # Get or load an LLM model
  def llm(model_id = nil, **options)
    model_id ||= "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    cache_key = "llm:#{model_id}:#{options.hash}"
    
    cached_model(cache_key) do
      Candle::LLM.from_pretrained(model_id, **options)
    end
  end
  
  # Get or load a GGUF LLM model
  def gguf_llm(model_id = "TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF", gguf_file: "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf", **options)
    cache_key = "gguf:#{model_id}:#{gguf_file}:#{options.hash}"
    
    cached_model(cache_key) do
      Candle::LLM.from_pretrained(model_id, gguf_file: gguf_file, **options)
    end
  end
  
  # Get or load an NER model
  def ner(model_id = "Babelscape/wikineural-multilingual-ner", **options)
    cache_key = "ner:#{model_id}:#{options.hash}"
    
    cached_model(cache_key) do
      Candle::NER.from_pretrained(model_id, **options)
    end
  end
  
  # Get or load an embedding model
  def embedding_model(model_id = "sentence-transformers/all-MiniLM-L6-v2", **options)
    cache_key = "embedding:#{model_id}:#{options.hash}"
    
    cached_model(cache_key) do
      Candle::EmbeddingModel.from_pretrained(model_id, **options)
    end
  end
  
  # Get or load a reranker model
  def reranker(model_id = "BAAI/bge-reranker-base", **options)
    cache_key = "reranker:#{model_id}:#{options.hash}"
    
    cached_model(cache_key) do
      Candle::Reranker.from_pretrained(model_id, **options)
    end
  end
  
  # Clear all cached models (useful for cleanup)
  def clear!
    @mutex.synchronize do
      @models.clear
      @load_errors.clear
    end
  end
  
  # Check if a model failed to load
  def failed?(cache_key)
    @load_errors.key?(cache_key)
  end
  
  # Get error for failed model
  def error_for(cache_key)
    @load_errors[cache_key]
  end
  
  private
  
  def cached_model(cache_key)
    @mutex.synchronize do
      # Return cached model if available
      return @models[cache_key] if @models.key?(cache_key)
      
      # Raise cached error if model previously failed
      if @load_errors.key?(cache_key)
        raise @load_errors[cache_key]
      end
      
      # Try to load the model
      begin
        puts "Loading model for cache key: #{cache_key}" if ENV['CANDLE_TEST_VERBOSE']
        model = yield
        @models[cache_key] = model
        model
      rescue => e
        # Cache the error so we don't retry
        @load_errors[cache_key] = e
        raise e
      end
    end
  end
end