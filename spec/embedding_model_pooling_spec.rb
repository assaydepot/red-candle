require "spec_helper"

RSpec.describe "EmbeddingModelPooling" do
  # Load model once for all tests
  let(:model) do
    @model ||= Candle::EmbeddingModel.from_pretrained
  rescue => e
    skip "EmbeddingModel loading failed: #{e.message}"
  end
  
  let(:string) { "The quick brown fox jumps over the lazy dog." }
  let(:embeddings) { model.embeddings(string) }
  
  # Clear cached model after spec completes
  after(:all) do
    @model = nil
    GC.start
  end

  describe "#pool_embedding" do
    it "matches manual pooling" do
      pooled_ruby = model.pool_embedding(embeddings)
      # Manually pool by averaging over sequence dimension (1)
      pooled_manual = embeddings.mean(1)
      pooled_ruby.first.to_a.each_index do |i|
        expect(pooled_ruby.first.to_a[i]).to be_within(1e-5).of(pooled_manual.first.to_a[i])
      end
    end
  end

  describe "#embedding with pooled_normalized" do
    it "matches manual pool and normalize" do
      pooled_norm_ruby = model.embedding(string, pooling_method: "pooled_normalized")
      # Manually pool and then normalize
      pooled_manual = embeddings.mean(1)
      norm = pooled_manual / Math.sqrt(pooled_manual.sqr.sum(1).values.first)
      pooled_norm_ruby.first.to_a.each_index do |i|
        expect(pooled_norm_ruby.first.to_a[i]).to be_within(1e-5).of(norm.first.to_a[i])
      end
    end
  end

  describe "#embedding with cls pooling" do
    it "extracts CLS token correctly" do
      cls_ruby = model.embedding(string, pooling_method: "cls")
      # Manually extract CLS token (index 0)
      # For shape [batch, seq, hidden], get first token of first batch
      cls_manual = embeddings.get(0).get(0)
      cls_ruby.first.to_a.each_index do |i|
        expect(cls_ruby.first.to_a[i]).to be_within(1e-5).of(cls_manual.to_a[i])
      end
    end
  end

  describe "different pooling methods" do
    it "produces different results for each method" do
      pooled = model.embedding(string, pooling_method: "pooled")
      pooled_norm = model.embedding(string, pooling_method: "pooled_normalized")
      cls = model.embedding(string, pooling_method: "cls")
      
      expect(pooled.first.to_a).not_to eq(pooled_norm.first.to_a)
      expect(pooled.first.to_a).not_to eq(cls.first.to_a)
      expect(pooled_norm.first.to_a).not_to eq(cls.first.to_a)
    end
  end

  describe "default pooling method" do
    it "uses pooled_normalized as default" do
      pooled = model.embedding(string, pooling_method: "pooled_normalized")
      default = model.embedding(string)
      expect(pooled.first.to_a).to eq(default.first.to_a)
    end
  end
end