require "spec_helper"
require "informers"

RSpec.describe "CandleInformersComparison" do
  include DeviceHelpers

  # Tolerance for floating point comparison
  FLOAT_TOLERANCE = 1e-4
  
  describe "reranker comparison" do
    before do
      skip "Informers gem doesn't respect HF_HUB_OFFLINE mode" if ENV['HF_HUB_OFFLINE'] == '1'
    end
    it "produces similar scores to Informers" do
      model_id = "cross-encoder/ms-marco-MiniLM-L-12-v2"
      query = "How many people live in London?"
      docs = [
        "London is known for its financial district",
        "Around 9 Million people live in London"
      ]

      # Test with Informers
      informers_model = Informers.pipeline("reranking", model_id)
      informers_result = informers_model.(query, docs, return_documents: false)
      # Extract just the scores from informers result
      informers_scores = informers_result.map { |r| r[:score] }
      
      # Test with Candle (default device)
      candle_reranker = Candle::Reranker.from_pretrained(model_id)
      
      candle_result = candle_reranker.rerank(
        query, 
        docs
      )
      # Extract just the scores from candle result
      candle_scores = candle_result.map { |r| r[:score] }

      # Compare scores elementwise
      expect(candle_scores.length).to eq(informers_scores.length), "Reranker score arrays have different lengths"
      informers_scores.zip(candle_scores).each_with_index do |(a, b), i|
        expect((a - b).abs).to be <= FLOAT_TOLERANCE, "Reranker score at index #{i} differs: informers=#{a} candle=#{b}"
      end
    end
  end

  describe "embedding model comparison" do
    before do
      skip "Informers gem doesn't respect HF_HUB_OFFLINE mode" if ENV['HF_HUB_OFFLINE'] == '1'
    end
    it "produces similar embeddings to Informers" do
      sentences = ["How is the weather today?", "What is the current weather like today?"]

      # Test with Informers
      informers_model = Informers.pipeline("embedding", "jinaai/jina-embeddings-v2-base-en", model_file_name: "../model")
      informers_embeddings = informers_model.(sentences)

      # Test with Candle
      candle_model = Candle::EmbeddingModel.from_pretrained("jinaai/jina-embeddings-v2-base-en")
      candle_embeddings = sentences.collect { |sentence| candle_model.embedding(sentence).values }

      expect(candle_embeddings.length).to eq(informers_embeddings.length), "Embedding arrays have different lengths"
      informers_embeddings.zip(candle_embeddings).each_with_index do |(informer_embedding, candle_embedding), i|
        informer_embedding.zip(candle_embedding).each_with_index do |(a, b), j|
          expect((a.to_f - b.to_f).abs).to be <= FLOAT_TOLERANCE, "Embedding value at index #{i} #{j} differs: informers=#{a} candle=#{b}"
        end
      end
    end
  end
  
  # Clear cached models after spec completes
  after(:all) do
    GC.start
  end

  private

  def cosine_similarity(vec1, vec2)
    raise "Vectors have different lengths" if vec1.length != vec2.length
    
    dot_product = 0.0
    norm1 = 0.0
    norm2 = 0.0
    
    vec1.zip(vec2).each do |a, b|
      dot_product += a * b
      norm1 += a ** 2
      norm2 += b ** 2
    end
    
    dot_product / (Math.sqrt(norm1) * Math.sqrt(norm2))
  end
end