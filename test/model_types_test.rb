require_relative "test_helper"

class ModelTypesTest < Minitest::Test
  SAFETENSOR_MODELS = {
    Candle::ModelType::JINA_BERT => [{
      model_path: "jinaai/jina-embeddings-v2-base-en",
      embedding_size: 768
    }],
    Candle::ModelType::STANDARD_BERT => [{
      model_path: "scientistcom/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
      embedding_size: 768
    }],
    Candle::ModelType::MINILM => [{
      model_path: "sentence-transformers/all-MiniLM-L6-v2",
      embedding_size: 384
    }],
    Candle::ModelType::DISTILBERT => [{
      model_path: "scientistcom/distilbert-base-uncased-finetuned-sst-2-english",
      embedding_size: 768
    }]
    # Add more safetensors models as needed
  }

  def test_model_types_initialize
    # Test all model types with their default Hugging Face repos
    SAFETENSOR_MODELS.each do |model_type, model_options|
      model_options.each do |model_option|
        model_path = model_option[:model_path]
        embedding_size = model_option[:embedding_size]
        puts ">>>>>>>>>>>>> model_path #{model_path} model_type #{model_type}"

        if model_type == Candle::ModelType::STANDARD_BERT
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
  end

  def test_safetensor_model_types_initialize
    SAFETENSOR_MODELS.each do |model_type, model_options|
      model_options.each do |model_option|
        model_path = model_option[:model_path]
        embedding_size = model_option[:embedding_size]
        begin
          model = Candle::Model.new(
            model_path: model_path,
            tokenizer_path: model_path,
            model_type: model_type,
            device: nil,
            embedding_size: embedding_size
          )
          assert model, "Model should be initialized for #{model_type}"
        rescue => e
          puts e.message
          puts e.backtrace.join("\n")
          puts "Initializing model for [#{model_type}]"
          puts model_option.inspect
          raise e
        end
      end
    end
  end
end
