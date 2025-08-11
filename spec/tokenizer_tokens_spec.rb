# frozen_string_literal: true

require "spec_helper"

RSpec.describe "TokenizerTokens" do
  let(:tokenizer) do
    @tokenizer ||= Candle::Tokenizer.from_pretrained("bert-base-uncased")
  end
  
  after(:all) do
    @tokenizer = nil
    GC.start
  end

  describe "#encode_to_tokens" do
    it "returns token strings" do
      text = "Hello, world!"
      tokens = tokenizer.encode_to_tokens(text)
      
      expect(tokens).to be_an(Array)
      expect(tokens).to all(be_a(String))
      
      # BERT should produce something like ["[CLS]", "hello", ",", "world", "!", "[SEP]"]
      expect(tokens).to include("[CLS]")
      expect(tokens).to include("[SEP]")
      expect(tokens.any? { |t| t.include?("hello") }).to be true
      expect(tokens.any? { |t| t.include?("world") }).to be true
    end
    
    it "respects add_special_tokens parameter" do
      text = "Hello, world!"
      tokens = tokenizer.encode_to_tokens(text, add_special_tokens: false)
      
      # Without special tokens
      expect(tokens).not_to include("[CLS]")
      expect(tokens).not_to include("[SEP]")
      expect(tokens.any? { |t| t.include?("hello") }).to be true
    end
  end

  describe "#encode_with_tokens" do
    it "returns both IDs and tokens" do
      text = "Hello, world!"
      result = tokenizer.encode_with_tokens(text)
      
      expect(result).to be_a(Hash)
      expect(result).to have_key(:ids)
      expect(result).to have_key(:tokens)
      
      ids = result[:ids]
      tokens = result[:tokens]
      
      expect(ids.length).to eq(tokens.length)
      expect(ids).to all(be_an(Integer))
      expect(tokens).to all(be_a(String))
      
      # Verify alignment
      tokens.each_with_index do |token, i|
        id = ids[i]
        # Each token should decode to itself
        expect(tokenizer.id_to_token(id)).to eq(token)
      end
    end
  end

  describe "#encode_batch_to_tokens" do
    it "tokenizes multiple texts" do
      texts = ["Hello world", "How are you?", "Testing batch"]
      token_batches = tokenizer.encode_batch_to_tokens(texts)
      
      expect(token_batches).to be_an(Array)
      expect(token_batches.length).to eq(3)
      
      token_batches.each do |tokens|
        expect(tokens).to be_an(Array)
        expect(tokens).to all(be_a(String))
        expect(tokens).to include("[CLS]")
        expect(tokens).to include("[SEP]")
      end
    end
  end

  describe "subword tokenization" do
    it "handles subword splitting" do
      # Test a word that likely gets split into subwords
      text = "unbelievable"
      tokens = tokenizer.encode_to_tokens(text, add_special_tokens: false)
      
      # BERT often splits longer words
      # Could be something like ["un", "##believ", "##able"]
      expect(tokens.length).to be >= 1
      
      # Join tokens and verify they reconstruct (roughly) to original
      joined = tokens.join("").gsub("##", "")
      expect(joined.downcase).to include("unbeliev")
    end
  end

  describe "tokenization visualization" do
    it "provides ID to token mapping" do
      # This demonstrates why token strings are useful
      text = "The quick brown fox jumps over the lazy dog."
      
      ids = tokenizer.encode(text)
      tokens = tokenizer.encode_to_tokens(text)
      
      # Create a visual mapping
      mapping = ids.zip(tokens)
      
      expect(ids.length).to eq(tokens.length)
      
      # Each ID corresponds to a token
      mapping.each do |id, token|
        expect(tokenizer.id_to_token(id)).to eq(token)
      end
    end
  end

  describe "special characters" do
    it "tokenizes special characters" do
      text = "Email: user@example.com, Price: $99.99"
      tokens = tokenizer.encode_to_tokens(text, add_special_tokens: false)
      
      # Should tokenize special characters
      expect(tokens.any? { |t| t.include?("@") || t == "@" }).to be true
      expect(tokens.any? { |t| t.include?("$") || t == "$" }).to be true
      expect(tokens.any? { |t| t.include?(".") || t == "." }).to be true
    end
  end

  describe "unicode tokenization" do
    it "handles unicode characters" do
      text = "Hello 世界 🌍"
      tokens = tokenizer.encode_to_tokens(text, add_special_tokens: false)
      
      expect(tokens.length).to be > 0
      # Unicode might be split into multiple tokens or preserved
      # depending on the tokenizer's vocabulary
    end
  end
end