require_relative "test_helper"

class ModelTypesTest < Minitest::Test
  # Helper to initialize a model and assert correctness
  private def initialize_model_for_type(model_type, model_option)
    model_path = model_option[:model_path]
    embedding_size = model_option[:embedding_size]
    model = Candle::Model.new(
      model_path: model_path,
      tokenizer_path: model_path,
      model_type: model_type,
      device: nil,
      embedding_size: embedding_size
    )
    assert model, "Model should be initialized for #{model_type}"
  end

  # Individual tests for each model type (explicitly written)

  def test_initialize_jina_bert
    model_type = Candle::ModelType::JINA_BERT
    model_option = { model_path: "jinaai/jina-embeddings-v2-base-en", embedding_size: 768 }
    initialize_model_for_type(model_type, model_option)
  end

  def test_safetensor_initialize_jina_bert
    model_type = Candle::ModelType::JINA_BERT
    model_option = { model_path: "jinaai/jina-embeddings-v2-base-en", embedding_size: 768 }
    initialize_model_for_type(model_type, model_option)
  end

  def test_initialize_standard_bert
    model_type = Candle::ModelType::STANDARD_BERT
    model_option = { model_path: "scientistcom/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", embedding_size: 768 }
    initialize_model_for_type(model_type, model_option)
  end

  def test_safetensor_initialize_standard_bert
    model_type = Candle::ModelType::STANDARD_BERT
    model_option = { model_path: "scientistcom/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", embedding_size: 768 }
    initialize_model_for_type(model_type, model_option)
  end

  def test_initialize_minilm
    model_type = Candle::ModelType::MINILM
    model_option = { model_path: "sentence-transformers/all-MiniLM-L6-v2", embedding_size: 384 }
    initialize_model_for_type(model_type, model_option)
  end

  def test_safetensor_initialize_minilm
    model_type = Candle::ModelType::MINILM
    model_option = { model_path: "sentence-transformers/all-MiniLM-L6-v2", embedding_size: 384 }
    initialize_model_for_type(model_type, model_option)
  end

  def test_initialize_distilbert
    model_type = Candle::ModelType::DISTILBERT
    model_option = { model_path: "scientistcom/distilbert-base-uncased-finetuned-sst-2-english", embedding_size: 768 }
    initialize_model_for_type(model_type, model_option)
  end

  def test_safetensor_initialize_distilbert
    model_type = Candle::ModelType::DISTILBERT
    model_option = { model_path: "scientistcom/distilbert-base-uncased-finetuned-sst-2-english", embedding_size: 768 }
    initialize_model_for_type(model_type, model_option)
  end
end
