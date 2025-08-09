# Module for caching expensive models across test runs
# This provides a single, consistent caching mechanism for all model types
# to avoid redundant loading during test execution.
#
# Usage:
#   model = ModelCache.ner
#   ModelCache.clear!  # Clear all models
#   ModelCache.clear_model(:ner)  # Clear specific model
#
module ModelCache
  extend self
  
  @cache = {}
  
  # Get the current cache status
  def cache_status
    @cache.keys
  end
  
  # Check if a model is cached
  def cached?(key)
    @cache.key?(key)
  end
  
  # GGUF LLM - Small quantized model for fast testing
  def gguf_llm
    @cache[:gguf_llm] ||= begin
      puts "Loading GGUF LLM..." if ENV['CANDLE_TEST_VERBOSE']
      Candle::LLM.from_pretrained(
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", 
        gguf_file: "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf",
        device: Candle::Device.cpu
      )
    rescue => e
      puts "Failed to load GGUF LLM: #{e.message}" if ENV['CANDLE_TEST_VERBOSE']
      nil
    end
  end
  
  # Standard LLM - Phi-2 model
  def llm
    @cache[:llm] ||= begin
      puts "Loading LLM (phi-2)..." if ENV['CANDLE_TEST_VERBOSE']
      Candle::LLM.from_pretrained(
        "microsoft/phi-2",
        device: Candle::Device.cpu
      )
    rescue => e
      puts "Failed to load LLM: #{e.message}" if ENV['CANDLE_TEST_VERBOSE']
      nil
    end
  end
  
  # NER model with fallback
  def ner
    @cache[:ner] ||= begin
      puts "Loading NER model..." if ENV['CANDLE_TEST_VERBOSE']
      # Try multiple models in order of preference
      models_to_try = [
        "Babelscape/wikineural-multilingual-ner",
        "dslim/bert-base-NER"
      ]
      
      loaded_model = nil
      models_to_try.each do |model_id|
        begin
          loaded_model = Candle::NER.from_pretrained(model_id)
          puts "Loaded NER model: #{model_id}" if ENV['CANDLE_TEST_VERBOSE']
          break
        rescue => e
          puts "Failed to load #{model_id}: #{e.message}" if ENV['CANDLE_TEST_VERBOSE']
        end
      end
      
      loaded_model
    end
  end
  
  # Alias for consistency with some specs
  def ner_model
    ner
  end
  
  # Embedding model
  def embedding_model
    @cache[:embedding_model] ||= begin
      puts "Loading Embedding model..." if ENV['CANDLE_TEST_VERBOSE']
      Candle::EmbeddingModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        tokenizer: "sentence-transformers/all-MiniLM-L6-v2",
        model_type: Candle::EmbeddingModelType::MINILM
      )
    rescue => e
      puts "Failed to load Embedding model: #{e.message}" if ENV['CANDLE_TEST_VERBOSE']
      nil
    end
  end
  
  # Reranker model
  def reranker
    @cache[:reranker] ||= begin
      puts "Loading Reranker model..." if ENV['CANDLE_TEST_VERBOSE']
      Candle::Reranker.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
    rescue => e
      puts "Failed to load Reranker: #{e.message}" if ENV['CANDLE_TEST_VERBOSE']
      nil
    end
  end
  
  # Tokenizer - useful for tokenizer-specific tests
  def tokenizer
    @cache[:tokenizer] ||= begin
      puts "Loading Tokenizer..." if ENV['CANDLE_TEST_VERBOSE']
      Candle::Tokenizer.from_pretrained("bert-base-uncased")
    rescue => e
      puts "Failed to load Tokenizer: #{e.message}" if ENV['CANDLE_TEST_VERBOSE']
      nil
    end
  end
  
  # Clear all cached models to free memory
  def clear!
    puts "Clearing all cached models..." if ENV['CANDLE_TEST_VERBOSE']
    @cache.clear
    GC.start # Force garbage collection
  end
  
  # Clear specific model
  def clear_model(key)
    puts "Clearing cached model: #{key}" if ENV['CANDLE_TEST_VERBOSE']
    @cache.delete(key)
    GC.start if @cache.empty? # GC if cache is now empty
  end
  
  # Preload commonly used models (useful for CI)
  def preload_common
    puts "Preloading common models..." if ENV['CANDLE_TEST_VERBOSE']
    gguf_llm
    ner
    embedding_model
    reranker
  end
end