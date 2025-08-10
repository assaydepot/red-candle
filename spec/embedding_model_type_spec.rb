# frozen_string_literal: true

require "spec_helper"

RSpec.describe "EmbeddingModelType" do
  describe "constants" do
    it "defines all expected constants" do
      expect(Candle::EmbeddingModelType::JINA_BERT).to eq("jina_bert")
      expect(Candle::EmbeddingModelType::STANDARD_BERT).to eq("standard_bert")
      expect(Candle::EmbeddingModelType::MINILM).to eq("minilm")
      expect(Candle::EmbeddingModelType::DISTILBERT).to eq("distilbert")
    end
  end
  
  describe ".all" do
    it "returns all model types" do
      all_types = Candle::EmbeddingModelType.all
      
      expect(all_types).to be_an(Array)
      expect(all_types.length).to eq(4)
      
      # Check all constants are included
      expect(all_types).to include(Candle::EmbeddingModelType::JINA_BERT)
      expect(all_types).to include(Candle::EmbeddingModelType::STANDARD_BERT)
      expect(all_types).to include(Candle::EmbeddingModelType::MINILM)
      expect(all_types).to include(Candle::EmbeddingModelType::DISTILBERT)
    end
  end
  
  describe ".suggested_model_paths" do
    it "provides suggested model paths for each type" do
      paths = Candle::EmbeddingModelType.suggested_model_paths
      
      expect(paths).to be_a(Hash)
      expect(paths.length).to eq(4)
      
      # Check all types have suggested paths
      expect(paths).to have_key(Candle::EmbeddingModelType::JINA_BERT)
      expect(paths).to have_key(Candle::EmbeddingModelType::STANDARD_BERT)
      expect(paths).to have_key(Candle::EmbeddingModelType::MINILM)
      expect(paths).to have_key(Candle::EmbeddingModelType::DISTILBERT)
      
      # Check specific paths
      expect(paths[Candle::EmbeddingModelType::JINA_BERT]).to eq("jinaai/jina-embeddings-v2-base-en")
      expect(paths[Candle::EmbeddingModelType::STANDARD_BERT]).to eq("scientistcom/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
      expect(paths[Candle::EmbeddingModelType::MINILM]).to eq("sentence-transformers/all-MiniLM-L6-v2")
      expect(paths[Candle::EmbeddingModelType::DISTILBERT]).to eq("scientistcom/distilbert-base-uncased-finetuned-sst-2-english")
    end
  end
  
  describe "enum-like usage" do
    it "can be used as enum values" do
      # Test that the constants can be used as enum values
      model_type = Candle::EmbeddingModelType::JINA_BERT
      expect(model_type).to eq("jina_bert")
      
      # Test that they work in case statements
      result = case model_type
      when Candle::EmbeddingModelType::JINA_BERT
        "jina"
      when Candle::EmbeddingModelType::STANDARD_BERT
        "bert"
      else
        "other"
      end
      
      expect(result).to eq("jina")
    end
  end
end