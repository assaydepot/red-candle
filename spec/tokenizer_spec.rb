# frozen_string_literal: true

require "spec_helper"

RSpec.describe "Tokenizer" do
  # Cache tokenizer for tests that can reuse it
  let(:tokenizer) do
    @tokenizer ||= Candle::Tokenizer.from_pretrained("bert-base-uncased")
  end
  
  # Clear any instance variables after spec completes
  after(:all) do
    @tokenizer = nil
    @embedding_model = nil
    @reranker = nil
    ModelCache.clear_model(:gguf_llm) # Clear the LLM used in one test
    GC.start
  end
  
  describe ".from_pretrained" do
    it "loads tokenizer from pretrained model" do
      expect(tokenizer).to be_a(Candle::Tokenizer)
    end
  end

  describe "#encode and #decode" do
    it "encodes text to token IDs and decodes back" do
      text = "Hello, world!"
      token_ids = tokenizer.encode(text)
      
      expect(token_ids).to be_an(Array)
      expect(token_ids).to all(be_an(Integer))
      
      # Decode back
      decoded = tokenizer.decode(token_ids)
      expect(decoded).to be_a(String)
      # BERT tokenizer lowercases and may add special tokens
      expect(decoded.downcase).to include("hello")
      expect(decoded.downcase).to include("world")
    end
  end

  describe "#encode without special tokens" do
    it "encodes without special tokens when specified" do
      text = "test"
      tokens_with_special = tokenizer.encode(text)
      tokens_without_special = tokenizer.encode(text, add_special_tokens: false)
      
      # Without special tokens should be shorter
      expect(tokens_without_special.length).to be < tokens_with_special.length
    end
  end

  describe "#encode_batch" do
    it "encodes multiple texts as batch" do
      texts = ["Hello", "World", "Test"]
      batch_tokens = tokenizer.encode_batch(texts)
      
      expect(batch_tokens).to be_an(Array)
      expect(batch_tokens.length).to eq(3)
      
      batch_tokens.each do |tokens|
        expect(tokens).to be_an(Array)
        expect(tokens).to all(be_an(Integer))
      end
    end
  end

  describe "vocab operations" do
    it "provides vocab size" do
      vocab_size = tokenizer.vocab_size
      expect(vocab_size).to be > 0
    end
    
    it "provides vocab as hash" do
      vocab = tokenizer.get_vocab
      expect(vocab).to be_a(Hash)
      expect(vocab.size).to be > 0
      
      # Check some known tokens
      expect(vocab).to have_key("[CLS]")
      expect(vocab).to have_key("[SEP]")
      expect(vocab).to have_key("[PAD]")
    end
  end

  describe "#id_to_token" do
    it "converts token ID back to token string" do
      # Get a token ID from vocab
      vocab = tokenizer.get_vocab
      token_id = vocab["[CLS]"]
      
      # Convert back to token
      token = tokenizer.id_to_token(token_id)
      expect(token).to eq("[CLS]")
    end
  end

  describe "#get_special_tokens" do
    it "returns special tokens hash" do
      special_tokens = tokenizer.get_special_tokens
      expect(special_tokens).to be_a(Hash)
      
      # BERT should have these special tokens
      expect(special_tokens).to have_key("cls_token")
      expect(special_tokens).to have_key("sep_token")
      expect(special_tokens).to have_key("pad_token")
    end
  end

  describe "#with_padding" do
    it "creates new tokenizer with padding configuration" do
      # Create a tokenizer with padding enabled
      padded_tokenizer = tokenizer.with_padding(length: 10)
      expect(padded_tokenizer).to be_a(Candle::Tokenizer)
      
      # The padded tokenizer should be a new instance
      expect(padded_tokenizer.object_id).not_to eq(tokenizer.object_id)
    end
  end

  describe "#with_truncation" do
    it "creates new tokenizer with truncation configuration" do
      # Create a tokenizer with truncation enabled
      truncated_tokenizer = tokenizer.with_truncation(512)
      expect(truncated_tokenizer).to be_a(Candle::Tokenizer)
      
      # The truncated tokenizer should be a new instance
      expect(truncated_tokenizer.object_id).not_to eq(tokenizer.object_id)
    end
  end

  describe "tokenizer from LLM" do
    it "gets tokenizer from LLM model" do
      llm = ModelCache.gguf_llm
      
      tokenizer = llm.tokenizer
      expect(tokenizer).to be_a(Candle::Tokenizer)
      
      # Test basic functionality
      text = "Hello"
      tokens = tokenizer.encode(text)
      expect(tokens).to be_an(Array)
      expect(tokens.length).to be > 0
    end
  end

  describe "tokenizer from EmbeddingModel" do
    let(:embedding_model) do
      @embedding_model ||= Candle::EmbeddingModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        tokenizer: "sentence-transformers/all-MiniLM-L6-v2",
        model_type: Candle::EmbeddingModelType::MINILM
      )
    end
    
    it "gets tokenizer from embedding model" do
      tokenizer = embedding_model.tokenizer
      expect(tokenizer).to be_a(Candle::Tokenizer)
      
      # Test basic functionality
      text = "Test embedding"
      tokens = tokenizer.encode(text)
      expect(tokens).to be_an(Array)
      expect(tokens.length).to be > 0
    end
  end

  describe "tokenizer from Reranker" do
    let(:reranker) do
      @reranker ||= Candle::Reranker.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    end
    
    it "gets tokenizer from reranker model" do
      tokenizer = reranker.tokenizer
      expect(tokenizer).to be_a(Candle::Tokenizer)
      
      # Test basic functionality
      text = "Query text"
      tokens = tokenizer.encode(text)
      expect(tokens).to be_an(Array)
      expect(tokens.length).to be > 0
    end
  end

  describe "#inspect" do
    it "provides meaningful inspect output" do
      inspect_str = tokenizer.inspect
      expect(inspect_str).to be_a(String)
      expect(inspect_str).to include("Candle::Tokenizer")
      expect(inspect_str).to include("vocab_size")
    end
  end

  describe "#to_s" do
    it "provides string representation" do
      str = tokenizer.to_s
      expect(str).to be_a(String)
      expect(str).to include("Candle::Tokenizer")
    end
  end
end