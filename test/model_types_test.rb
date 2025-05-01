require_relative "test_helper"

class ModelTypesTest < Minitest::Test
  MODEL_TYPES = {
    Candle::ModelType::JINA_BERT => "jinaai/jina-embeddings-v2-base-en",
    Candle::ModelType::STANDARD_BERT => "bert-base-uncased",
    Candle::ModelType::MINILM => "sentence-transformers/all-MiniLM-L6-v2",
    Candle::ModelType::SENTIMENT => "distilbert-base-uncased-finetuned-sst-2-english",
    Candle::ModelType::LLAMA => "meta-llama/Llama-2-7b"
  }

  SAFETENSOR_MODELS = {
    Candle::ModelType::JINA_BERT => {
      model_path: "jinaai/jina-embeddings-v2-base-en",
      embedding_size: 768
    },
    Candle::ModelType::STANDARD_BERT => {
      model_path: "bert-base-uncased",
      embedding_size: 768
    },
    Candle::ModelType::MINILM => {
      model_path: "sentence-transformers/all-MiniLM-L6-v2",
      embedding_size: 384
    },
    # Add more safetensors models as needed
  }

  def test_model_types_initialize
    MODEL_TYPES.each do |model_type, model_path|
      # Llama may require a token, so we skip it by default
      next if model_type == Candle::ModelType::LLAMA

      embedding_size = case model_type
        when Candle::ModelType::JINA_BERT then 768
        when Candle::ModelType::STANDARD_BERT then 768
        when Candle::ModelType::MINILM then 384
        when Candle::ModelType::SENTIMENT then 768
        else nil
      end

      model = Candle::Model.new(
        model_path: model_path,
        tokenizer_path: model_path,
        model_type: model_type,
        device: nil,
        embedding_size: embedding_size
      )
      assert model, "Model should be initialized for #{model_type}"
    end
  end

  def test_safetensor_model_types_initialize
    SAFETENSOR_MODELS.each do |model_type, info|
      model = Candle::Model.new(
        model_path: info[:model_path],
        tokenizer_path: info[:model_path],
        model_type: model_type,
        device: nil,
        embedding_size: info[:embedding_size]
      )
      assert model, "Model should be initialized for #{model_type}"
    end
  end
end
