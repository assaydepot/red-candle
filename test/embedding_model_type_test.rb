# frozen_string_literal: true

require_relative "test_helper"

class EmbeddingModelTypeTest < Minitest::Test
  def test_constants_defined
    assert_equal "jina_bert", Candle::EmbeddingModelType::JINA_BERT
    assert_equal "standard_bert", Candle::EmbeddingModelType::STANDARD_BERT
    assert_equal "minilm", Candle::EmbeddingModelType::MINILM
    assert_equal "distilbert", Candle::EmbeddingModelType::DISTILBERT
  end
  
  def test_all_method
    all_types = Candle::EmbeddingModelType.all
    
    assert_instance_of Array, all_types
    assert_equal 4, all_types.length
    
    # Check all constants are included
    assert_includes all_types, Candle::EmbeddingModelType::JINA_BERT
    assert_includes all_types, Candle::EmbeddingModelType::STANDARD_BERT
    assert_includes all_types, Candle::EmbeddingModelType::MINILM
    assert_includes all_types, Candle::EmbeddingModelType::DISTILBERT
  end
  
  def test_suggested_model_paths
    paths = Candle::EmbeddingModelType.suggested_model_paths
    
    assert_instance_of Hash, paths
    assert_equal 4, paths.length
    
    # Check all types have suggested paths
    assert paths.key?(Candle::EmbeddingModelType::JINA_BERT)
    assert paths.key?(Candle::EmbeddingModelType::STANDARD_BERT)
    assert paths.key?(Candle::EmbeddingModelType::MINILM)
    assert paths.key?(Candle::EmbeddingModelType::DISTILBERT)
    
    # Check specific paths
    assert_equal "jinaai/jina-embeddings-v2-base-en", paths[Candle::EmbeddingModelType::JINA_BERT]
    assert_equal "scientistcom/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", paths[Candle::EmbeddingModelType::STANDARD_BERT]
    assert_equal "sentence-transformers/all-MiniLM-L6-v2", paths[Candle::EmbeddingModelType::MINILM]
    assert_equal "scientistcom/distilbert-base-uncased-finetuned-sst-2-english", paths[Candle::EmbeddingModelType::DISTILBERT]
  end
  
  def test_module_usage_as_enum
    # Test that the constants can be used as enum values
    model_type = Candle::EmbeddingModelType::JINA_BERT
    assert_equal "jina_bert", model_type
    
    # Test that they work in case statements
    result = case model_type
    when Candle::EmbeddingModelType::JINA_BERT
      "jina"
    when Candle::EmbeddingModelType::STANDARD_BERT
      "bert"
    else
      "other"
    end
    
    assert_equal "jina", result
  end
end