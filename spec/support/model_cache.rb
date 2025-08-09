# Module for caching expensive models across test runs
module ModelCache
  extend self
  
  @cache = {}
  
  def gguf_llm
    @cache[:gguf_llm] ||= begin
      Candle::LLM.from_pretrained(
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", 
        gguf_file: "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf",
        device: Candle::Device.cpu
      )
    end
  end
  
  def llm
    @cache[:llm] ||= begin
      Candle::LLM.from_pretrained(
        "microsoft/phi-2",
        device: Candle::Device.cpu
      )
    end
  end
  
  def ner
    @cache[:ner] ||= begin
      # Try multiple models in order of preference
      models_to_try = [
        "Babelscape/wikineural-multilingual-ner",
        "dslim/bert-base-NER"
      ]
      
      loaded_model = nil
      models_to_try.each do |model_id|
        begin
          loaded_model = Candle::NER.from_pretrained(model_id)
          break
        rescue => e
          puts "Failed to load #{model_id}: #{e.message}"
        end
      end
      
      loaded_model
    end
  end
  
  # Clear all cached models to free memory
  def clear!
    @cache.clear
    GC.start # Force garbage collection
  end
  
  # Clear specific model
  def clear_model(key)
    @cache.delete(key)
  end
end