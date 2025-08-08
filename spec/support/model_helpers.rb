module ModelHelpers
  # Provide easy access to the model cache
  def model_cache
    ModelCache.instance
  end
  
  # Skip test if model fails to load
  def skip_if_model_unavailable(model_type, *args, **kwargs)
    begin
      case model_type
      when :llm
        model_cache.llm(*args, **kwargs)
      when :gguf_llm
        model_cache.gguf_llm(*args, **kwargs)
      when :ner
        model_cache.ner(*args, **kwargs)
      when :embedding_model
        model_cache.embedding_model(*args, **kwargs)
      when :reranker
        model_cache.reranker(*args, **kwargs)
      else
        raise ArgumentError, "Unknown model type: #{model_type}"
      end
    rescue => e
      skip "Model unavailable: #{e.message}"
    end
  end
  
  # Shared context for specs that need cached models
  RSpec.shared_context "cached models" do
    # Default models - loaded lazily and cached
    let(:llm) do
      model_cache.llm
    rescue => e
      skip "LLM model unavailable: #{e.message}"
    end
    
    let(:gguf_llm) do
      model_cache.gguf_llm
    rescue => e
      skip "GGUF model unavailable: #{e.message}"
    end
    
    let(:ner) do
      model_cache.ner
    rescue => e
      skip "NER model unavailable: #{e.message}"
    end
    
    let(:embedding_model) do
      model_cache.embedding_model
    rescue => e
      skip "Embedding model unavailable: #{e.message}"
    end
    
    let(:reranker) do
      model_cache.reranker
    rescue => e
      skip "Reranker model unavailable: #{e.message}"
    end
  end
  
  # Shared examples for model behavior
  RSpec.shared_examples "a model with device support" do
    it "can be created on CPU" do
      model = described_class.from_pretrained(model_id, device: Candle::Device.cpu)
      expect(model.device.to_s).to include("cpu")
    end
    
    context "with Metal available", if: DeviceHelpers::AVAILABLE_DEVICES[:metal] do
      it "can be created on Metal" do
        model = described_class.from_pretrained(model_id, device: Candle::Device.metal)
        expect(model.device.to_s).to include("metal")
      end
    end
    
    context "with CUDA available", if: DeviceHelpers::AVAILABLE_DEVICES[:cuda] do
      it "can be created on CUDA" do
        model = described_class.from_pretrained(model_id, device: Candle::Device.cuda)
        expect(model.device.to_s).to include("cuda")
      end
    end
  end
  
  RSpec.shared_examples "a model with inspect" do
    it "has a meaningful inspect output" do
      expect(subject.inspect).to match(/#<Candle::\w+/)
      expect(subject.inspect).to include("model")
    end
    
    it "responds to model_id" do
      expect(subject).to respond_to(:model_id)
    end
    
    it "responds to options" do
      expect(subject).to respond_to(:options)
      expect(subject.options).to be_a(Hash)
    end
  end
end