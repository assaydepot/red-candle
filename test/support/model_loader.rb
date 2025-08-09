# Smart model loader that supports multiple sources
module ModelLoader
  extend self
  
  # Check if we should use offline mode (CI with cached models)
  def offline_mode?
    ENV['HF_OFFLINE'] == 'true' || ENV['CI_OFFLINE_MODE'] == 'true'
  end
  
  # Check if we have a local model cache directory
  def local_model_path(model_id)
    # Check multiple possible locations
    paths = [
      ENV['LOCAL_MODEL_PATH'],
      File.join(ENV['HF_HOME'] || "~/.cache/huggingface", "hub", model_id.gsub('/', '--')),
      File.join("test", "fixtures", "models", model_id.gsub('/', '--'))
    ].compact
    
    paths.each do |path|
      expanded = File.expand_path(path)
      return expanded if Dir.exist?(expanded)
    end
    
    nil
  end
  
  # Load model with fallback strategies
  def load_model(model_type, model_id, **options)
    # Try strategies in order
    strategies = [
      -> { load_from_cache(model_type, model_id, **options) },
      -> { load_from_huggingface(model_type, model_id, **options) },
      -> { load_with_retry(model_type, model_id, **options) }
    ]
    
    last_error = nil
    strategies.each do |strategy|
      begin
        return strategy.call
      rescue => e
        last_error = e
        if rate_limited?(e)
          puts "Rate limited, trying next strategy..."
        end
      end
    end
    
    # If all strategies fail, return nil or raise
    if offline_mode?
      raise "Model #{model_id} not available in offline mode: #{last_error.message}"
    else
      nil  # Let tests skip gracefully
    end
  end
  
  private
  
  def rate_limited?(error)
    error.message.include?("429") || error.message.include?("Too Many Requests")
  end
  
  def load_from_cache(model_type, model_id, **options)
    local_path = local_model_path(model_id)
    return nil unless local_path
    
    puts "Loading #{model_id} from cache: #{local_path}"
    
    case model_type
    when :ner
      Candle::NER.from_pretrained(local_path, **options)
    when :embedding
      Candle::EmbeddingModel.from_pretrained(local_path, **options)
    when :reranker
      Candle::Reranker.from_pretrained(local_path, **options)
    when :llm
      Candle::LLM.from_pretrained(local_path, **options)
    when :tokenizer
      Candle::Tokenizer.from_pretrained(local_path)
    else
      raise "Unknown model type: #{model_type}"
    end
  end
  
  def load_from_huggingface(model_type, model_id, **options)
    return nil if offline_mode?
    
    puts "Loading #{model_id} from HuggingFace..."
    
    case model_type
    when :ner
      Candle::NER.from_pretrained(model_id, **options)
    when :embedding
      Candle::EmbeddingModel.from_pretrained(model_id, **options)
    when :reranker
      Candle::Reranker.from_pretrained(model_id, **options)
    when :llm
      Candle::LLM.from_pretrained(model_id, **options)
    when :tokenizer
      Candle::Tokenizer.from_pretrained(model_id)
    else
      raise "Unknown model type: #{model_type}"
    end
  end
  
  def load_with_retry(model_type, model_id, **options)
    return nil if offline_mode?
    
    retries = 3
    delay = 5
    
    retries.times do |i|
      begin
        return load_from_huggingface(model_type, model_id, **options)
      rescue => e
        if rate_limited?(e) && i < retries - 1
          puts "Rate limited, waiting #{delay} seconds before retry #{i + 1}..."
          sleep delay
          delay *= 2  # Exponential backoff
        else
          raise
        end
      end
    end
  end
end

# Monkey-patch test classes to use smart loader
class Minitest::Test
  def load_model_with_fallback(model_type, model_id, **options)
    model = ModelLoader.load_model(model_type, model_id, **options)
    
    if model.nil?
      skip "Model #{model_id} not available (rate limited or offline)"
    end
    
    model
  end
end