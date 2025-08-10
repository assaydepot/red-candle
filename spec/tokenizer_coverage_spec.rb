# frozen_string_literal: true

require "spec_helper"

RSpec.describe "TokenizerCoverage" do
  include DeviceHelpers
  
  let(:bert_tokenizer) do
    @bert_tokenizer ||= Candle::Tokenizer.from_pretrained("bert-base-uncased")
  end
  
  let(:gpt2_tokenizer) do
    @gpt2_tokenizer ||= Candle::Tokenizer.from_pretrained("gpt2")
  end
  
  after(:all) do
    @bert_tokenizer = nil
    @gpt2_tokenizer = nil
    GC.start
  end
  
  describe "edge cases and error handling" do
    it "handles empty strings correctly" do
      # Empty string
      tokens = bert_tokenizer.encode("")
      expect(tokens).to be_a(Array)
      
      # Decode empty array
      text = bert_tokenizer.decode([])
      expect(text).to eq("")
    end
    
    it "handles very long text with truncation" do
      long_text = "word " * 1000
      truncated = bert_tokenizer.with_truncation(512)
      tokens = truncated.encode(long_text)
      expect(tokens.length).to be <= 512
    end
    
    it "handles special characters and emojis" do
      special_text = "Hello 😊 World! @#$%^&*()"
      tokens = bert_tokenizer.encode(special_text)
      expect(tokens).to be_a(Array)
      
      decoded = bert_tokenizer.decode(tokens)
      expect(decoded).to be_a(String)
    end
    
    it "handles newlines and tabs" do
      text_with_whitespace = "Hello\nWorld\tTest\r\nEnd"
      tokens = bert_tokenizer.encode(text_with_whitespace)
      expect(tokens).to be_a(Array)
    end
    
    it "handles batch operations with mixed content" do
      mixed_batch = [
        "Normal text",
        "",
        "Text with 😊 emoji",
        "Very " * 100 + "long text",
        "Special chars: @#$%"
      ]
      
      batch_tokens = bert_tokenizer.encode_batch(mixed_batch)
      expect(batch_tokens).to be_a(Array)
      expect(batch_tokens.length).to eq(5)
    end
    
    it "preserves token alignment with encode_with_tokens" do
      text = "Hello, world!"
      result = bert_tokenizer.encode_with_tokens(text)
      
      expect(result).to have_key(:ids)
      expect(result).to have_key(:tokens)
      expect(result[:ids].length).to eq(result[:tokens].length)
    end
    
    it "handles padding with different strategies" do
      short_text = "Hi"
      padded = bert_tokenizer.with_padding(length: 20)
      tokens = padded.encode(short_text)
      # Padding adds tokens to reach specified length
      expect(tokens).to be_a(Array)
    end
    
    it "correctly identifies special tokens" do
      special_tokens = bert_tokenizer.get_special_tokens
      expect(special_tokens).to be_a(Hash)
      # Keys are like "cls_token", "sep_token", etc.
      expect(special_tokens).to have_key("cls_token")
      expect(special_tokens).to have_key("sep_token")
      expect(special_tokens).to have_key("pad_token")
    end
    
    it "handles vocabulary queries for unknown tokens" do
      vocab = bert_tokenizer.get_vocab
      
      # Check for UNK token
      expect(vocab).to have_key("[UNK]")
      unk_id = vocab["[UNK]"]
      
      # Verify id_to_token works for UNK
      token_str = bert_tokenizer.id_to_token(unk_id)
      expect(token_str).to eq("[UNK]")
    end
    
    it "handles different tokenizer models correctly" do
      # GPT-2 specific behavior
      gpt2_text = "Hello world"
      gpt2_tokens = gpt2_tokenizer.encode(gpt2_text)
      
      # GPT-2 uses different tokenization than BERT
      bert_tokens = bert_tokenizer.encode(gpt2_text)
      
      # Same text should produce different tokens
      expect(gpt2_tokens).not_to eq(bert_tokens)
    end
    
    it "handles token string extraction" do
      text = "The quick brown fox"
      token_strings = bert_tokenizer.encode_to_tokens(text)
      
      expect(token_strings).to be_a(Array)
      expect(token_strings.all? { |t| t.is_a?(String) }).to be true
    end
    
    it "handles batch token string extraction" do
      texts = ["Hello", "World"]
      batch_token_strings = bert_tokenizer.encode_batch_to_tokens(texts)
      
      expect(batch_token_strings).to be_a(Array)
      expect(batch_token_strings.length).to eq(2)
      batch_token_strings.each do |tokens|
        expect(tokens).to be_a(Array)
        expect(tokens.all? { |t| t.is_a?(String) }).to be true
      end
    end
    
    it "handles decoding with skip_special_tokens option" do
      text = "Hello world"
      tokens = bert_tokenizer.encode(text)
      
      # Decode with special tokens
      with_special = bert_tokenizer.decode(tokens, skip_special_tokens: false)
      expect(with_special).to include("hello")
      
      # Decode without special tokens
      without_special = bert_tokenizer.decode(tokens, skip_special_tokens: true)
      expect(without_special).to include("hello")
    end
    
    it "handles encoding without special tokens" do
      text = "Hello world"
      
      # Encode with special tokens (default)
      with_special = bert_tokenizer.encode(text, add_special_tokens: true)
      
      # Encode without special tokens
      without_special = bert_tokenizer.encode(text, add_special_tokens: false)
      
      # Without special tokens should be shorter
      expect(without_special.length).to be < with_special.length
    end
  end
  
  describe "performance characteristics" do
    it "efficiently handles large batches" do
      large_batch = Array.new(100) { |i| "Test sentence number #{i}" }
      
      start_time = Time.now
      batch_results = bert_tokenizer.encode_batch(large_batch)
      elapsed = Time.now - start_time
      
      expect(batch_results.length).to eq(100)
      # Batch processing should be reasonably fast
      expect(elapsed).to be < 5.0
    end
    
    it "maintains consistency across multiple calls" do
      text = "Consistency test"
      
      tokens1 = bert_tokenizer.encode(text)
      tokens2 = bert_tokenizer.encode(text)
      
      expect(tokens1).to eq(tokens2)
    end
  end
  
  describe "model-specific behaviors" do
    it "handles BERT-specific tokenization" do
      # BERT uses WordPiece tokenization
      text = "preprocessing"
      tokens = bert_tokenizer.encode_to_tokens(text)
      
      # Should contain subword tokens
      expect(tokens).to be_a(Array)
      # BERT breaks this into subwords with ## prefix
      expect(tokens.any? { |t| t.start_with?("##") }).to be true
    end
    
    it "handles GPT-2 specific tokenization" do
      # GPT-2 uses BPE tokenization
      text = "unbelievable"
      tokens = gpt2_tokenizer.encode_to_tokens(text)
      
      # GPT-2 handles this differently than BERT
      expect(tokens).to be_a(Array)
      # GPT-2 typically uses different subword boundaries
      expect(tokens.join).to include("un")
    end
  end
  
  describe "integration with models" do
    it "provides compatible tokens for embedding models" do
      model = Candle::EmbeddingModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        tokenizer: "sentence-transformers/all-MiniLM-L6-v2",
        model_type: Candle::EmbeddingModelType::MINILM
      )
      
      tokenizer = model.tokenizer
      expect(tokenizer).to be_a(Candle::Tokenizer)
      
      # Tokenizer should work with the model
      text = "Test embedding"
      tokens = tokenizer.encode(text)
      expect(tokens).to be_a(Array)
    end
  end
  
  describe "advanced configuration" do
    pending "chains padding and truncation configurations" do
      # NOTE: Currently padding seems to override truncation in the Rust implementation
      # When both are set, the padding length takes precedence
      configured = bert_tokenizer
        .with_padding(length: 128)
        .with_truncation(64)
      
      long_text = "word " * 100
      tokens = configured.encode(long_text)
      
      # Should be truncated to 64, but currently returns 128 (padding length)
      expect(tokens.length).to be <= 64
    end
    
    it "handles vocabulary size queries" do
      vocab_size = bert_tokenizer.vocab_size
      # NOTE: vocab_size method doesn't accept arguments in current implementation
      
      expect(vocab_size).to be_a(Integer)
      expect(vocab_size).to be > 0
      expect(vocab_size).to eq(30522)  # BERT base vocabulary size
    end
    
    it "provides detailed vocabulary information" do
      vocab = bert_tokenizer.get_vocab
      # NOTE: get_vocab method doesn't accept arguments in current implementation
      
      expect(vocab).to be_a(Hash)
      expect(vocab.size).to eq(30522)  # BERT base vocabulary size
    end
  end
end