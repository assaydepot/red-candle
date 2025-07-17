#!/usr/bin/env ruby
# frozen_string_literal: true

require "bundler/setup"
require "candle"

# Example: Named Entity Recognition with Candle

puts "Named Entity Recognition Example"
puts "=" * 60

# Note: This example demonstrates the API, but requires downloading a model
# For testing without downloads, see the pattern-based examples below

puts "\n1. Model-based NER (requires model download):"

# Load a pre-trained NER model
# Note: Using a model that includes tokenizer.json file
ner = Candle::NER.from_pretrained("Babelscape/wikineural-multilingual-ner")

# Example text
text = "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California in April 1976."

puts "Text: \"#{text}\""
puts "\nExtracted entities:"

# Extract entities with default confidence threshold
entities = ner.extract_entities(text)

entities.each do |entity|
  puts "  - '#{entity['text']}' [#{entity['label']}] " \
        "at positions #{entity['start']}-#{entity['end']} " \
        "(confidence: #{entity['confidence'].round(3)})"
end

# Get token-level predictions
puts "\nToken-level analysis:"
tokens = ner.predict_tokens(text)

# Show first 10 tokens
tokens[0..9].each do |token_info|
  if token_info["label"] != "O"
    puts "  Token: '#{token_info['token']}' → #{token_info['label']} " \
          "(#{token_info['confidence'].round(3)})"
  end
end

# Check supported entity types
puts "\nSupported entity types:"
puts "  #{ner.entity_types.join(', ')}"

# Pattern-based NER (works without model downloads)
puts "\n2. Pattern-based Entity Recognition:"

# Create pattern recognizers
email_recognizer = Candle::PatternEntityRecognizer.new("EMAIL")
email_recognizer.add_pattern(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/)

phone_recognizer = Candle::PatternEntityRecognizer.new("PHONE")
phone_recognizer.add_pattern(/\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/)
phone_recognizer.add_pattern(/\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b/)

url_recognizer = Candle::PatternEntityRecognizer.new("URL")
url_recognizer.add_pattern(/https?:\/\/[^\s]+/)
url_recognizer.add_pattern(/www\.[^\s]+/)

# Test text
contact_text = "Contact us at info@example.com or call 555-123-4567. Visit https://example.com for more info."

puts "\nText: \"#{contact_text}\""
puts "Extracted entities:"

# Extract entities
[email_recognizer, phone_recognizer, url_recognizer].each do |recognizer|
  entities = recognizer.recognize(contact_text)
  entities.each do |entity|
    puts "  - '#{entity['text']}' [#{entity['label']}] at positions #{entity['start']}-#{entity['end']}"
  end
end

# Gazetteer-based NER
puts "\n3. Gazetteer-based Entity Recognition:"

# Create a company gazetteer
companies = ["Apple", "Microsoft", "Google", "Amazon", "Meta", "Tesla", "OpenAI"]
company_recognizer = Candle::GazetteerEntityRecognizer.new("COMPANY", companies)

# Create a location gazetteer
locations = ["New York", "San Francisco", "London", "Tokyo", "Paris"]
location_recognizer = Candle::GazetteerEntityRecognizer.new("LOCATION", locations)

business_text = "Apple and Google compete in the smartphone market. Microsoft has offices in New York and London."

puts "\nText: \"#{business_text}\""
puts "Extracted entities:"

[company_recognizer, location_recognizer].each do |recognizer|
  entities = recognizer.recognize(business_text)
  entities.each do |entity|
    puts "  - '#{entity['text']}' [#{entity['label']}] at positions #{entity['start']}-#{entity['end']}"
  end
end

# Hybrid NER combining patterns and gazetteers
puts "\n4. Hybrid NER (Pattern + Gazetteer):"

hybrid = Candle::HybridNER.new

# Add various recognizers
hybrid.add_pattern_recognizer("EMAIL", [/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/])
hybrid.add_pattern_recognizer("PHONE", [/\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/])
hybrid.add_gazetteer_recognizer("COMPANY", ["Apple", "Google", "Microsoft"])
hybrid.add_gazetteer_recognizer("PERSON", ["Tim Cook", "Satya Nadella", "Sundar Pichai"])

mixed_text = "Tim Cook (tim@apple.com, 555-0123) announced Apple's new product. Contact: 408-555-1234"

puts "\nText: \"#{mixed_text}\""
puts "Extracted entities:"

entities = hybrid.extract_entities(mixed_text)
entities.each do |entity|
  source = entity["source"] || "model"
  puts "  - '#{entity['text']}' [#{entity['label']}] at positions #{entity['start']}-#{entity['end']} (#{source})"
end

# Custom entity types
puts "\n5. Custom Entity Types:"

# Create a cryptocurrency recognizer
crypto_patterns = [
  /\b(BTC|ETH|USDT|BNB|XRP|USDC|SOL|ADA|DOGE)\b/,
  /\bBitcoin\b/i,
  /\bEthereum\b/i,
  /\b[A-Z]{3,5}\/USD[T]?\b/  # Trading pairs
]

crypto_recognizer = Candle::PatternEntityRecognizer.new("CRYPTO", crypto_patterns)

crypto_text = "Bitcoin (BTC) reached $50k while Ethereum trades at ETH/USDT 3000."
puts "\nText: \"#{crypto_text}\""

crypto_entities = crypto_recognizer.recognize(crypto_text)
puts "Cryptocurrency entities:"
crypto_entities.each do |entity|
  puts "  - '#{entity['text']}' at positions #{entity['start']}-#{entity['end']}"
end

puts "\n" + "=" * 60
puts "NER example completed!"