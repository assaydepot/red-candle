#!/usr/bin/env ruby
# frozen_string_literal: true

require "bundler/setup"
require "candle"

# Example: Using the Candle Tokenizer

puts "Candle Tokenizer Example"
puts "=" * 50

# Create a tokenizer from a pretrained model
puts "\n1. Loading tokenizer from HuggingFace..."
tokenizer = Candle::Tokenizer.from_pretrained("bert-base-uncased")
puts "✓ Tokenizer loaded successfully"

# Basic encoding and decoding
puts "\n2. Basic text encoding and decoding:"
text = "Hello, world! How are you today?"
puts "   Original text: #{text}"

tokens = tokenizer.encode(text)
puts "   Token IDs: #{tokens.inspect}"
puts "   Number of tokens: #{tokens.length}"

decoded = tokenizer.decode(tokens)
puts "   Decoded text: #{decoded}"

# Encoding without special tokens
puts "\n3. Encoding with and without special tokens:"
tokens_with_special = tokenizer.encode(text)
tokens_without_special = tokenizer.encode(text, add_special_tokens: false)
puts "   With special tokens: #{tokens_with_special.length} tokens"
puts "   Without special tokens: #{tokens_without_special.length} tokens"
puts "   Special tokens added: #{tokens_with_special.length - tokens_without_special.length}"

# Batch encoding
puts "\n4. Batch encoding multiple texts:"
texts = [
  "The quick brown fox",
  "jumps over the lazy dog",
  "A journey of a thousand miles"
]
batch_tokens = tokenizer.encode_batch(texts)
texts.each_with_index do |txt, i|
  puts "   Text #{i + 1}: '#{txt}' -> #{batch_tokens[i].length} tokens"
end

# Vocabulary operations
puts "\n5. Vocabulary information:"
vocab_size = tokenizer.vocab_size
puts "   Total vocabulary size: #{vocab_size}"

# Get some special tokens
vocab = tokenizer.get_vocab
special_tokens = ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"]
puts "   Special token IDs:"
special_tokens.each do |token|
  if vocab[token]
    puts "     #{token}: #{vocab[token]}"
  end
end

# Token to ID mapping
puts "\n6. Token to string conversion:"
sample_tokens = tokens[0..4]  # First 5 tokens
puts "   First 5 tokens:"
sample_tokens.each do |token_id|
  token_str = tokenizer.id_to_token(token_id)
  puts "     ID #{token_id} -> '#{token_str}'"
end

# Special tokens info
puts "\n7. Special tokens information:"
special_info = tokenizer.get_special_tokens
special_info.each do |name, id|
  puts "   #{name}: #{id}"
end

# Create tokenizers with padding and truncation
puts "\n8. Tokenizer configuration:"
puts "   Creating tokenizer with padding..."
padded_tokenizer = tokenizer.with_padding(length: 128)
puts "   ✓ Padded tokenizer created"

puts "   Creating tokenizer with truncation..."
truncated_tokenizer = tokenizer.with_truncation(512)
puts "   ✓ Truncated tokenizer created"

# Demonstrate usage with models (if available)
puts "\n9. Tokenizer access from models:"
puts "   Note: Model tokenizers can be accessed via:"
puts "   - llm.tokenizer for LLM models"
puts "   - embedding_model.tokenizer for embedding models"
puts "   - reranker.tokenizer for reranker models"

puts "\n" + "=" * 50
puts "Example completed successfully!"