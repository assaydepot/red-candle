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
    # Test all model types with their default Hugging Face repos
    MODEL_TYPES.each do |model_type, model_path|
      embedding_size = case model_type
        when Candle::ModelType::JINA_BERT then 768
        when Candle::ModelType::STANDARD_BERT then 768
        when Candle::ModelType::MINILM then 384
        when Candle::ModelType::SENTIMENT then 768
        when Candle::ModelType::LLAMA then nil
        else nil
      end

      if model_type == Candle::ModelType::LLAMA
        # Llama: should succeed if GGML is present, fail with safetensors or bin
        assert_raises RuntimeError, /safetensors|not yet implemented|not found/ do
          Candle::Model.new(
            model_path: model_path,
            tokenizer_path: model_path,
            model_type: model_type,
            device: nil,
            embedding_size: embedding_size
          )
        end
      elsif model_type == Candle::ModelType::STANDARD_BERT || model_type == Candle::ModelType::SENTIMENT
        # These official models do not provide safetensors, should error helpfully
        assert_raises RuntimeError, /model\.safetensors not found|Only safetensors models are supported/ do
          Candle::Model.new(
            model_path: model_path,
            tokenizer_path: model_path,
            model_type: model_type,
            device: nil,
            embedding_size: embedding_size
          )
        end
      else
        # Should succeed for models with safetensors
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

  def test_llama_ggml_and_pytorch_bin
    llama_model = "meta-llama/Llama-2-7b"
    # Should error for safetensors (not implemented)
    assert_raises RuntimeError, /Llama safetensors loading is not yet implemented/ do
      Candle::Model.new(
        model_path: llama_model,
        tokenizer_path: llama_model,
        model_type: Candle::ModelType::LLAMA,
        device: nil
      )
    end
    # Should error for missing ggml
    assert_raises RuntimeError, /model\.ggml not found/ do
      Candle::Model.new(
        model_path: "some/llama-without-ggml",
        tokenizer_path: "some/llama-without-ggml",
        model_type: Candle::ModelType::LLAMA,
        device: nil
      )
    end
    # Should error for pytorch_model.bin
    assert_raises RuntimeError, /model\.safetensors not found|Only safetensors models are supported/ do
      Candle::Model.new(
        model_path: "bert-base-uncased", # This repo only provides pytorch_model.bin
        tokenizer_path: "bert-base-uncased",
        model_type: Candle::ModelType::STANDARD_BERT,
        device: nil
      )
    end
  end
end
