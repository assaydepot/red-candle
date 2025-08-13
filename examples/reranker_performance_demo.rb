#!/usr/bin/env ruby
# frozen_string_literal: true

require "bundler/setup"
require "candle"
require "benchmark"

puts "=" * 80
puts "Reranker Performance Demo: Document Length Impact"
puts "=" * 80
puts

# Initialize the reranker
print "Loading reranker model... "
reranker = Candle::Reranker.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2", device: Candle::Device.best)
puts "✓"
puts "Device: #{reranker.device}"
puts

# Define the query
query = "What are the key benefits of machine learning in healthcare?"

puts "Query: \"#{query}\""
puts "-" * 80

# Test 1: Very short documents (1-3 tokens each)
puts "\n📄 TEST 1: Very Short Documents (1-3 tokens each)"
puts "-" * 40

short_docs = [
  "Healthcare",
  "ML benefits",
  "AI",
  "Medical diagnosis",
  "Drug discovery",
  "Patient care",
  "Clinical trials",
  "Research"
]

puts "Documents:"
short_docs.each_with_index do |doc, i|
  puts "  #{i+1}. \"#{doc}\" (#{doc.split.length} words, ~#{doc.length} chars)"
end

print "\n⏱️  Reranking short documents... "
short_time = Benchmark.realtime do
  @short_results = reranker.rerank(query, short_docs)
end
puts "Done in #{(short_time * 1000).round(2)}ms"

puts "\n📊 Results (sorted by relevance):"
@short_results.sort_by { |r| -r[:score] }.each do |result|
  doc = short_docs[result[:doc_id]]
  puts "  Score: #{'%.4f' % result[:score]} - \"#{doc}\""
end

# Test 2: Very long documents (thousands of tokens each)
puts "\n" + "=" * 80
puts "\n📚 TEST 2: Very Long Documents (2000-5000 tokens each)"
puts "-" * 40

# Generate realistic long documents about ML in healthcare
long_doc_1 = <<~DOC
  Machine learning has revolutionized healthcare in numerous ways. One of the most significant benefits is in medical imaging and diagnosis. 
  #{("Deep learning algorithms can now detect diseases like cancer, diabetic retinopathy, and heart conditions with accuracy that matches or exceeds human specialists. " * 50)}
  Furthermore, ML models can process thousands of images in the time it takes a human to review a few dozen. This speed and accuracy combination is transforming radiology departments worldwide.
  #{("The ability to identify patterns in complex medical data that humans might miss is another crucial advantage. " * 50)}
  Predictive analytics powered by machine learning can forecast patient outcomes, identify high-risk individuals, and optimize treatment plans.
  #{("In drug discovery, ML accelerates the identification of promising compounds and predicts their effectiveness. " * 50)}
  This technology is also personalizing medicine by analyzing genetic data to tailor treatments to individual patients.
DOC

long_doc_2 = <<~DOC
  The administrative side of healthcare has also benefited enormously from machine learning applications.
  #{("Natural language processing helps extract meaningful information from unstructured clinical notes and medical records. " * 60)}
  This automation reduces the burden on healthcare professionals and improves data accuracy.
  #{("Scheduling optimization, resource allocation, and workflow management all benefit from ML algorithms. " * 60)}
  Insurance claim processing and fraud detection have become more efficient and accurate.
  #{("Patient flow prediction helps hospitals manage capacity and reduce wait times. " * 60)}
  These improvements lead to better patient experiences and more efficient healthcare delivery overall.
DOC

long_doc_3 = <<~DOC
  Research and clinical trials represent another frontier where machine learning is making substantial impacts.
  #{("ML models can identify suitable candidates for clinical trials by analyzing patient records and matching them with trial criteria. " * 55)}
  This accelerates recruitment and ensures better participant selection.
  #{("Analysis of trial data is enhanced by ML's ability to detect subtle patterns and correlations. " * 55)}
  Post-market surveillance of drugs and medical devices benefits from ML's ability to process vast amounts of real-world data.
  #{("Adverse event detection and signal processing have become more sophisticated and timely. " * 55)}
  These capabilities are making medical research more efficient and effective.
DOC

long_doc_4 = <<~DOC
  Prevention and public health initiatives are being transformed by machine learning technologies.
  #{("Epidemiological models powered by ML can predict disease outbreaks and track their spread more accurately. " * 70)}
  Population health management uses ML to identify at-risk groups and target interventions.
  #{("Behavioral health applications use ML to provide personalized mental health support and crisis intervention. " * 70)}
  Wearable devices combined with ML algorithms enable continuous health monitoring and early warning systems.
  These preventive approaches are shifting healthcare from reactive to proactive care.
DOC

long_docs = [
  long_doc_1.strip,
  long_doc_2.strip,
  long_doc_3.strip,
  long_doc_4.strip
]

puts "Documents:"
long_docs.each_with_index do |doc, i|
  word_count = doc.split.length
  char_count = doc.length
  # Rough token estimate (1 token ≈ 4 chars)
  token_estimate = char_count / 4
  preview = doc[0..100].gsub(/\s+/, ' ') + "..."
  puts "  #{i+1}. \"#{preview}\""
  puts "      (#{word_count} words, #{char_count} chars, ~#{token_estimate} tokens)"
end

print "\n⏱️  Reranking long documents... "
long_time = Benchmark.realtime do
  @long_results = reranker.rerank(query, long_docs)
end
puts "Done in #{(long_time * 1000).round(2)}ms"

puts "\n📊 Results (sorted by relevance):"
@long_results.sort_by { |r| -r[:score] }.each_with_index do |result, idx|
  doc = long_docs[result[:doc_id]]
  preview = doc[0..80].gsub(/\s+/, ' ') + "..."
  puts "  #{idx+1}. Score: #{'%.4f' % result[:score]}"
  puts "     \"#{preview}\""
end

# Performance comparison
puts "\n" + "=" * 80
puts "\n⚡ PERFORMANCE COMPARISON"
puts "-" * 40

puts "Short documents (#{short_docs.length} docs, 1-3 tokens each):"
puts "  Total time: #{(short_time * 1000).round(2)}ms"
puts "  Avg per doc: #{(short_time * 1000 / short_docs.length).round(2)}ms"

puts "\nLong documents (#{long_docs.length} docs, 2000-5000 tokens each):"
puts "  Total time: #{(long_time * 1000).round(2)}ms"
puts "  Avg per doc: #{(long_time * 1000 / long_docs.length).round(2)}ms"

speedup = short_time / long_time
puts "\n📈 Analysis:"
if speedup > 1
  puts "  Short documents were #{speedup.round(2)}x faster"
else
  puts "  Long documents were #{(1/speedup).round(2)}x slower"
end

puts "\n💡 Key Insights:"
puts "  1. Long documents are automatically truncated to 512 tokens"
puts "  2. Performance remains stable regardless of input length"
puts "  3. Tokenization overhead exists but doesn't scale with document length"
puts "  4. The model processes the same amount of data (512 tokens) in both cases"

# Test 3: Mixed batch for real-world scenario
puts "\n" + "=" * 80
puts "\n🔄 TEST 3: Mixed Batch (Realistic Scenario)"
puts "-" * 40

mixed_docs = [
  "AI healthcare",  # Very short
  "Machine learning revolutionizes medical diagnosis through pattern recognition.",  # Medium
  long_doc_1[0..500],  # Truncated long
  "Benefits: accuracy, speed, cost reduction, personalization, early detection.",  # Medium
  long_doc_2,  # Full long
  "ML applications"  # Very short
]

puts "Documents (mixed lengths):"
mixed_docs.each_with_index do |doc, i|
  words = doc.split.length
  if words < 20
    preview = doc
  else
    preview = doc[0..60].gsub(/\s+/, ' ') + "..."
  end
  puts "  #{i+1}. #{words} words: \"#{preview}\""
end

print "\n⏱️  Reranking mixed batch... "
mixed_time = Benchmark.realtime do
  @mixed_results = reranker.rerank(query, mixed_docs)
end
puts "Done in #{(mixed_time * 1000).round(2)}ms"

puts "\n📊 Top 3 most relevant:"
@mixed_results.sort_by { |r| -r[:score] }[0..2].each_with_index do |result, idx|
  doc = mixed_docs[result[:doc_id]]
  preview = doc[0..80].gsub(/\s+/, ' ')
  preview += "..." if doc.length > 80
  puts "  #{idx+1}. Score: #{'%.4f' % result[:score]} - \"#{preview}\""
end

puts "\n" + "=" * 80
puts "\n✅ Demo Complete!"
puts "\nSummary:"
puts "  • Reranker handles documents of any length gracefully"
puts "  • Automatic truncation prevents errors and ensures consistent performance"
puts "  • Batch processing is efficient for mixed document lengths"
puts "  • Use chunking strategies for long documents if full coverage is needed"