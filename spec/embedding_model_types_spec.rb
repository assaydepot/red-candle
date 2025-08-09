require "spec_helper"

RSpec.describe "EmbeddingModelTypes" do
  # Helper to initialize a model and assert correctness
  def initialize_embedding_model_for_type(model_type, model_option)
    model_path = model_option[:model_path]
    embedding_size = model_option[:embedding_size]
    model = Candle::EmbeddingModel.from_pretrained(
      model_path,
      tokenizer: model_path,
      model_type: model_type,
      device: nil,
      embedding_size: embedding_size
    )
    expect(model).not_to be_nil, "EmbeddingModel should be initialized for #{model_type}"
  end
  
  # Clear any cached models after all tests
  after(:all) do
    GC.start
  end

  # Individual tests for each model type (explicitly written)

  describe "JINA_BERT" do
    it "initializes correctly" do
      model_type = Candle::EmbeddingModelType::JINA_BERT
      model_option = { model_path: "jinaai/jina-embeddings-v2-base-en", embedding_size: 768 }
      initialize_embedding_model_for_type(model_type, model_option)
    end
  end

  describe "STANDARD_BERT" do
    it "initializes correctly" do
      model_type = Candle::EmbeddingModelType::STANDARD_BERT
      model_option = { model_path: "scientistcom/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", embedding_size: 768 }
      initialize_embedding_model_for_type(model_type, model_option)
    end
  end

  describe "MINILM" do
    it "initializes correctly" do
      model_type = Candle::EmbeddingModelType::MINILM
      model_option = { model_path: "sentence-transformers/all-MiniLM-L6-v2", embedding_size: 384 }
      initialize_embedding_model_for_type(model_type, model_option)
    end
  end

  describe "DISTILBERT" do
    it "initializes correctly" do
      model_type = Candle::EmbeddingModelType::DISTILBERT
      model_option = { model_path: "scientistcom/distilbert-base-uncased-finetuned-sst-2-english", embedding_size: 768 }
      initialize_embedding_model_for_type(model_type, model_option)
    end
  end
end