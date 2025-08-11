require "spec_helper"

RSpec.describe "EmbeddingModel" do
  let(:model) do
    @model ||= Candle::EmbeddingModel.from_pretrained
  end
  
  # Clear cached model after spec completes
  after(:all) do
    @model = nil
    GC.start
  end
  
  describe "pooled embeddings" do
    it "returns correct shape for pooled normalized embeddings" do
      string = "Hi there"
      pooled = model.embedding(string, pooling_method: "pooled_normalized")
      expect(pooled.shape).to eq([1, 768])
    end
    
    it "correctly computes pooled embeddings" do
      string = "Hi there"
      embeddings = model.embeddings(string) # shape: [1, n_tokens, hidden_size]
      pooled = model.embedding(string, pooling_method: "pooled") # shape: [1, hidden_size]

      # Recreate pooling logic from Rust: mean over tokens axis (axis 1)
      # Use Candle::Tensor shape and to_a for backend-agnostic pooling
      shape = embeddings.shape # [1, n_tokens, hidden_size]
      n_tokens = shape[1]
      hidden_size = shape[2]
      arr = embeddings.to_a # [ [ [token1], [token2], ... ] ]

      # arr[0] is [n_tokens, hidden_size]
      sum = Array.new(hidden_size, 0.0)
      arr[0].each do |token_vec|
        token_vec.each_with_index { |v, i| sum[i] += v }
      end
      mean = sum.map { |v| v / n_tokens.to_f }
      custom_pooled = [mean]

      # Assert each element is close within a tolerance
      pooled_arr = pooled.first.to_a
      custom_arr = custom_pooled.first.to_a
      tolerance = 1e-6
      custom_arr.zip(pooled_arr).each_with_index do |(a, b), i|
        expect(a).to be_within(tolerance).of(b), "Mismatch at index #{i}: #{a} vs #{b}"
      end
    end
  end
end