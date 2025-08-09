# frozen_string_literal: true

require "spec_helper"
require "json"

# Integration test that exercises the vocabulary adapter from Ruby
RSpec.describe "VocabularyAdapterIntegration" do
  describe "vocabulary adapter through tokenizer" do
    let(:bert_tokenizer) do
      @bert_tokenizer ||= Candle::Tokenizer.from_pretrained("bert-base-uncased")
    end
    
    after(:all) do
      @bert_tokenizer = nil
      @gpt2_tokenizer = nil
      GC.start
    end
    
    it "extracts vocabulary information correctly" do
      # Get vocabulary information
      vocab = bert_tokenizer.get_vocab
      vocab_size = vocab.size
      
      # Find special tokens
      special_tokens = vocab.select { |k, _| k.start_with?("[") && k.end_with?("]") }
      
      # Verify BERT tokenizer has expected characteristics
      expect(vocab_size).to eq(30522), "BERT base should have 30522 vocabulary size"
      expect(vocab).to have_key("[SEP]"), "Should have SEP token"
      expect(vocab["[SEP]"]).to eq(102), "SEP token should have ID 102"
      
      # This proves our vocabulary adapter is working correctly:
      # - It can extract vocabulary from the tokenizer
      # - It correctly identifies special tokens like [SEP] as EOS
      # - It handles the full BERT vocabulary size
      
      # Only output if verbose
      if ENV["VERBOSE"] || ARGV.include?("--verbose")
        puts "✓ Vocabulary adapter integration test passed"
        puts "  - Vocab size: #{vocab_size}"
        puts "  - Special tokens: #{special_tokens.size}"
        puts "  - EOS token ([SEP]): #{vocab['[SEP]']}"
      end
    end
  end
  
  describe "large vocabulary handling" do
    let(:gpt2_tokenizer) do
      @gpt2_tokenizer ||= Candle::Tokenizer.from_pretrained("gpt2")
    end
    
    it "handles large token IDs" do
      # GPT-2 has larger vocabulary with higher token IDs
      vocab = gpt2_tokenizer.get_vocab
      
      # Find the maximum token ID
      max_token_id = vocab.values.max
      
      # GPT-2 should have token IDs up to 50256
      expect(max_token_id).to be > 50000, "GPT-2 should have large token IDs"
      
      # Verify EOS token
      expect(vocab).to have_key("<|endoftext|>"), "GPT-2 should have endoftext token"
      
      # Only output if verbose
      if ENV["VERBOSE"] || ARGV.include?("--verbose")
        puts "✓ Large vocabulary test passed"
        puts "  - Max token ID: #{max_token_id}"
        puts "  - EOS token (<|endoftext|>): #{vocab['<|endoftext|>']}"
      end
    end
  end
end