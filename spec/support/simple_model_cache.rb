# Simple shared model caching for RSpec
# Models are loaded once and shared across examples

module SimpleModelCache
  # Shared contexts for model caching
  RSpec.shared_context "cached_ner_model" do
    # Load once for all examples in this context
    let!(:ner_model) do
      @ner_model ||= begin
        puts "Loading NER model..." if ENV['CANDLE_TEST_VERBOSE']
        Candle::NER.from_pretrained("Babelscape/wikineural-multilingual-ner")
      end
    end
  end
  
  RSpec.shared_context "cached_embedding_model" do
    # Load once for all examples in this context
    let!(:embedding_model) do
      @embedding_model ||= begin
        puts "Loading Embedding model..." if ENV['CANDLE_TEST_VERBOSE']
        Candle::EmbeddingModel.from_pretrained
      end
    end
  end
  
  RSpec.shared_context "cached_llm_models" do
    # Cache GGUF model
    let!(:gguf_llm) do
      @gguf_llm ||= begin
        puts "Loading GGUF LLM..." if ENV['CANDLE_TEST_VERBOSE']
        Candle::LLM.from_pretrained(
          "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", 
          gguf_file: "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf",
          device: Candle::Device.cpu
        )
      end
    end
    
    # Cache regular LLM (only if needed)
    let!(:llm) do
      @llm ||= begin
        puts "Loading LLM..." if ENV['CANDLE_TEST_VERBOSE']
        Candle::LLM.from_pretrained(
          "microsoft/phi-2",
          device: Candle::Device.cpu
        )
      end
    end
  end
  
  RSpec.shared_context "cached_reranker_model" do
    let!(:reranker_model) do
      @reranker_model ||= begin
        puts "Loading Reranker model..." if ENV['CANDLE_TEST_VERBOSE']
        Candle::Reranker.from_pretrained
      end
    end
  end
end