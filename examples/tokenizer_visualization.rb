#!/usr/bin/env ruby
# frozen_string_literal: true

require "bundler/setup"
require "candle"

# Example: Visualizing tokenization with token strings

puts "Tokenizer Visualization Example"
puts "=" * 60

# Load tokenizer
tokenizer = Candle::Tokenizer.from_pretrained("bert-base-uncased")

# Example 1: Basic tokenization visualization
puts "\n1. Basic Tokenization Visualization:"
text = "The quick brown fox jumps over the lazy dog."
result = tokenizer.encode_with_tokens(text)

puts "Original: #{text}"
puts "\nTokenization:"
result["tokens"].each_with_index do |token, i|
  id = result["ids"][i]
  puts "  [#{id.to_s.rjust(5)}] → '#{token}'"
end

# Example 2: Subword tokenization
puts "\n2. Subword Tokenization Examples:"
words = ["unbelievable", "preprocessing", "tokenization", "artificial intelligence"]

words.each do |word|
  tokens = tokenizer.encode_to_tokens(word, add_special_tokens: false)
  puts "\n'#{word}' → [#{tokens.join(", ")}]"
  
  # Show how subwords are marked
  if tokens.any? { |t| t.include?("##") }
    puts "  (## indicates continuation of previous token)"
  end
end

# Example 3: Understanding special tokens
puts "\n3. Special Tokens in Context:"
sentences = [
  "Hello world",
  "How are you?",
  "Machine learning is amazing!"
]

sentences.each do |sentence|
  tokens = tokenizer.encode_to_tokens(sentence)
  puts "\n\"#{sentence}\""
  puts "  → #{tokens.inspect}"
end

# Example 4: Comparing with and without special tokens
puts "\n4. Effect of Special Tokens:"
text = "Compare tokenization"

with_special = tokenizer.encode_to_tokens(text, add_special_tokens: true)
without_special = tokenizer.encode_to_tokens(text, add_special_tokens: false)

puts "Text: '#{text}'"
puts "With special tokens:    #{with_special.inspect}"
puts "Without special tokens: #{without_special.inspect}"
puts "Difference: #{with_special.length - without_special.length} special tokens added"

# Example 5: Batch visualization
puts "\n5. Batch Tokenization:"
texts = [
  "The weather is nice today.",
  "I love programming in Ruby!",
  "Tokenizers are fascinating."
]

batch_tokens = tokenizer.encode_batch_to_tokens(texts)
texts.each_with_index do |text, i|
  puts "\nText #{i + 1}: \"#{text}\""
  puts "Tokens: #{batch_tokens[i].join(" | ")}"
end

# Example 6: Token analysis for NER preparation
puts "\n6. Token Analysis (NER Preparation):"
ner_text = "John Smith works at OpenAI in San Francisco."
result = tokenizer.encode_with_tokens(ner_text)

puts "Text: #{ner_text}"
puts "\nToken-level analysis:"
result["tokens"].each_with_index do |token, i|
  # Skip special tokens for NER
  next if ["[CLS]", "[SEP]", "[PAD]"].include?(token)
  
  # Simple heuristic: capitalized tokens might be entities
  if token.match?(/^[A-Z]/) || token.match?(/^##[A-Z]/)
    puts "  '#{token}' (position #{i}) - Potential entity"
  else
    puts "  '#{token}' (position #{i})"
  end
end

# Example 7: Understanding tokenizer vocabulary coverage
puts "\n7. Vocabulary Coverage:"
test_words = ["hello", "world", "supercalifragilisticexpialidocious", "Ruby", "AI", "2024"]

test_words.each do |word|
  tokens = tokenizer.encode_to_tokens(word, add_special_tokens: false)
  if tokens.length == 1 && !tokens[0].include?("##")
    puts "'#{word}' → in vocabulary as single token"
  else
    puts "'#{word}' → split into #{tokens.length} tokens: #{tokens.inspect}"
  end
end

puts "\n" + "=" * 60
puts "Visualization complete!"
puts "\nKey insights:"
puts "- Tokens show exactly how the model 'sees' text"
puts "- Subword tokenization handles unknown words"
puts "- Special tokens mark sentence boundaries"
puts "- Token strings are essential for debugging and NER tasks"