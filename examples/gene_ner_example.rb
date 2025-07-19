#!/usr/bin/env ruby
# frozen_string_literal: true

require "bundler/setup"
require "candle"

# Example: Gene Name Recognition using Hybrid NER

puts "Gene Name Recognition Example"
puts "=" * 60

# Create a hybrid NER system for biomedical text
hybrid_ner = Candle::HybridNER.new

# Add gene name patterns
gene_patterns = [
  /\b[A-Z][A-Z0-9]{2,}\b/,           # TP53, BRCA1, EGFR
  /\b[A-Z][a-z]+\d+\b/,              # Abc1, Sox2
  /\b[A-Z]{2,}-[A-Z0-9]+\b/,         # HLA-B27, IL-6
  /\bCD\d+\b/,                       # CD4, CD8, CD34
  /\b[A-Z]+\d[A-Z]\d*\b/,            # RAD51C, PALB2
]

hybrid_ner.add_pattern_recognizer("GENE", gene_patterns)

# Add a gazetteer of known cancer genes
cancer_genes = [
  "TP53", "BRCA1", "BRCA2", "EGFR", "KRAS", "BRAF",
  "PIK3CA", "PTEN", "APC", "MLH1", "MSH2", "MSH6",
  "PMS2", "VHL", "RB1", "CDKN2A", "NF1", "NF2",
  "ATM", "CHEK2", "PALB2", "RAD51C", "RAD51D"
]

hybrid_ner.add_gazetteer_recognizer("CANCER_GENE", cancer_genes)

# Add protein patterns
protein_patterns = [
  /\bp\d{2}\b/,                      # p53, p16
  /\b[A-Z][a-z]+\s+kinase\b/,        # Tyrosine kinase
  /\b[A-Z][a-z]+\s+receptor\b/,      # Estrogen receptor
]

hybrid_ner.add_pattern_recognizer("PROTEIN", protein_patterns)

# Example texts
texts = [
  "TP53 mutations are found in over 50% of human cancers.",
  "The BRCA1 and BRCA2 genes are associated with hereditary breast cancer.",
  "EGFR tyrosine kinase inhibitors are used to treat lung cancer.",
  "The p53 protein regulates cell cycle and acts as a tumor suppressor.",
  "Patients with Lynch syndrome have mutations in MLH1, MSH2, MSH6, or PMS2.",
  "HER2-positive breast cancers overexpress the HER2 receptor.",
  "The RAS/RAF/MEK/ERK pathway includes KRAS, BRAF, and other signaling proteins.",
  "CD4+ T cells are crucial for immune response against tumors."
]

puts "\nProcessing biomedical texts..."
puts

texts.each_with_index do |text, i|
  puts "Text #{i + 1}: \"#{text}\""
  
  # Extract entities
  entities = hybrid_ner.extract_entities(text)
  
  if entities.empty?
    puts "  No entities found"
  else
    puts "  Entities found:"
    entities.each do |entity|
      source = entity["source"] || "model"
      puts "    - '#{entity['text']}' [#{entity['label']}] at positions #{entity['start']}-#{entity['end']} (#{source})"
    end
  end
  puts
end

# Demonstrate pattern matching for gene nomenclature
puts "Gene Nomenclature Examples:"
puts "-" * 30

gene_examples = [
  "wild-type TP53",
  "TP53 R273H mutation",
  "BRCA1/2 deficiency",
  "HER2/neu amplification",
  "KRAS G12D",
  "BRAF V600E",
  "PIK3CA H1047R",
  "EGFR exon 19 deletion",
  "ALK-EML4 fusion",
  "BCR-ABL1 translocation"
]

gene_examples.each do |example|
  entities = hybrid_ner.extract_entities(example)
  genes = entities.select { |e| e["label"] == "GENE" || e["label"] == "CANCER_GENE" }
  gene_names = genes.map { |e| e["text"] }.join(", ")
  puts "  '#{example}' → Genes: #{gene_names.empty? ? 'none detected' : gene_names}"
end

# Create a specialized gene recognizer
puts "\n\nSpecialized Gene Recognition:"
puts "-" * 30

# Gene with mutation notation
mutation_pattern = /\b([A-Z][A-Z0-9]{2,})\s+([A-Z]\d+[A-Z])\b/

text_with_mutations = "Common mutations include KRAS G12D, BRAF V600E, and EGFR T790M in lung cancer."
puts "\nText: \"#{text_with_mutations}\""

# Find gene-mutation pairs
text_with_mutations.scan(mutation_pattern) do |gene, mutation|
  puts "  Found: Gene '#{gene}' with mutation '#{mutation}'"
end

# Example of combining with a hypothetical BioBERT model
puts "\n\nNote: For production use, consider using specialized biomedical NER models:"
puts "  - dmis-lab/biobert-base-cased-v1.2"
puts "  - allenai/scibert_scivocab_uncased"
puts "  - microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
puts "\nThese models are trained on biomedical text and recognize:"
puts "  - Genes (DNA/RNA)"
puts "  - Proteins"
puts "  - Chemicals/Drugs"
puts "  - Diseases"
puts "  - Cell types"
puts "  - Species"