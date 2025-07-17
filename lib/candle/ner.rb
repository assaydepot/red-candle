# frozen_string_literal: true

module Candle
  # Named Entity Recognition (NER) for token classification
  #
  # This class provides methods to extract named entities from text using
  # pre-trained BERT-based models. It supports standard NER labels like
  # PER (person), ORG (organization), LOC (location), and can be extended
  # with custom entity types.
  #
  # @example Load a pre-trained NER model
  #   ner = Candle::NER.from_pretrained("Babelscape/wikineural-multilingual-ner")
  #
  # @example Load a model with a specific tokenizer
  #   ner = Candle::NER.from_pretrained("dslim/bert-base-NER", tokenizer: "bert-base-cased")
  #
  # @example Extract entities from text
  #   entities = ner.extract_entities("Apple Inc. was founded by Steve Jobs in Cupertino.")
  #   # => [
  #   #   { text: "Apple Inc.", label: "ORG", start: 0, end: 10, confidence: 0.99 },
  #   #   { text: "Steve Jobs", label: "PER", start: 26, end: 36, confidence: 0.98 },
  #   #   { text: "Cupertino", label: "LOC", start: 40, end: 49, confidence: 0.97 }
  #   # ]
  #
  # @example Get token-level predictions
  #   tokens = ner.predict_tokens("John works at Google")
  #   # Returns detailed token-by-token predictions with confidence scores
  class NER
    class << self
      # Load a pre-trained NER model from HuggingFace
      #
      # @param model_id [String] HuggingFace model ID (e.g., "dslim/bert-base-NER")
      # @param device [Device, nil] Device to run on (defaults to best available)
      # @param tokenizer [String, nil] Tokenizer model ID to use (defaults to same as model_id)
      # @return [NER] NER instance
      def from_pretrained(model_id, device: nil, tokenizer: nil)
        new(model_id, device, tokenizer)
      end
      
      # Popular pre-trained models for different domains
      def suggested_models
        {
          general: {
            model: "Babelscape/wikineural-multilingual-ner",
            note: "Has tokenizer.json"
          },
          general_alt: {
            model: "dslim/bert-base-NER",
            tokenizer: "bert-base-cased",
            note: "Requires separate tokenizer"
          },
          multilingual: {
            model: "Davlan/bert-base-multilingual-cased-ner-hrl",
            note: "Check tokenizer availability"
          },
          biomedical: {
            model: "dmis-lab/biobert-base-cased-v1.2",
            note: "May require specific tokenizer"
          },
          clinical: {
            model: "emilyalsentzer/Bio_ClinicalBERT",
            note: "May require specific tokenizer"
          },
          scientific: {
            model: "allenai/scibert_scivocab_uncased",
            note: "May require specific tokenizer"
          }
        }
      end
    end
    
    # Create an alias for the native method
    alias_method :_extract_entities, :extract_entities
    
    # Extract entities from text
    #
    # @param text [String] The text to analyze
    # @param confidence_threshold [Float] Minimum confidence score (default: 0.9)
    # @return [Array<Hash>] Array of entity hashes with text, label, start, end, confidence
    def extract_entities(text, confidence_threshold: 0.9)
      # Call the native method with positional arguments
      _extract_entities(text, confidence_threshold)
    end
    
    # Get available entity types
    #
    # @return [Array<String>] List of entity types (without B-/I- prefixes)
    def entity_types
      return @entity_types if @entity_types
      
      label_config = labels
      @entity_types = label_config["label2id"].keys
        .reject { |l| l == "O" }
        .map { |l| l.sub(/^[BI]-/, "") }
        .uniq
        .sort
    end
    
    # Check if model supports a specific entity type
    #
    # @param entity_type [String] Entity type to check (e.g., "GENE", "PER")
    # @return [Boolean] Whether the model recognizes this entity type
    def supports_entity?(entity_type)
      entity_types.include?(entity_type.upcase)
    end
    
    # Extract entities of a specific type
    #
    # @param text [String] The text to analyze
    # @param entity_type [String] Entity type to extract (e.g., "PER", "ORG")
    # @param confidence_threshold [Float] Minimum confidence score
    # @return [Array<Hash>] Filtered entities of the specified type
    def extract_entity_type(text, entity_type, confidence_threshold: 0.9)
      entities = extract_entities(text, confidence_threshold: confidence_threshold)
      entities.select { |e| e["label"] == entity_type.upcase }
    end
    
    # Analyze text and return both entities and token predictions
    #
    # @param text [String] The text to analyze
    # @param confidence_threshold [Float] Minimum confidence for entities
    # @return [Hash] Hash with :entities and :tokens keys
    def analyze(text, confidence_threshold: 0.9)
      {
        entities: extract_entities(text, confidence_threshold: confidence_threshold),
        tokens: predict_tokens(text)
      }
    end
    
    # Get a formatted string representation of entities
    #
    # @param text [String] The text to analyze
    # @param confidence_threshold [Float] Minimum confidence score
    # @return [String] Formatted output with entities highlighted
    def format_entities(text, confidence_threshold: 0.9)
      entities = extract_entities(text, confidence_threshold: confidence_threshold)
      return text if entities.empty?
      
      # Sort by start position (reverse for easier insertion)
      entities.sort_by! { |e| -e["start"] }
      
      result = text.dup
      entities.each do |entity|
        label = "[#{entity['label']}:#{entity['confidence'].round(2)}]"
        result.insert(entity["end"], label)
      end
      
      result
    end
    
    # Get model information
    #
    # @return [String] Model description
    def inspect
      "#<Candle::NER #{model_info}>"
    end
    
    alias to_s inspect
  end
  
  # Pattern-based entity recognizer for custom entities
  class PatternEntityRecognizer
    attr_reader :patterns, :entity_type
    
    def initialize(entity_type, patterns = [])
      @entity_type = entity_type
      @patterns = patterns
    end
    
    # Add a pattern (String or Regexp)
    def add_pattern(pattern)
      @patterns << pattern
      self
    end
    
    # Recognize entities using patterns
    def recognize(text, tokenizer = nil)
      entities = []
      
      @patterns.each do |pattern|
        regex = pattern.is_a?(Regexp) ? pattern : Regexp.new(pattern)
        
        text.scan(regex) do |match|
          match_text = $&
          match_start = $~.offset(0)[0]
          match_end = $~.offset(0)[1]
          
          entities << {
            "text" => match_text,
            "label" => @entity_type,
            "start" => match_start,
            "end" => match_end,
            "confidence" => 1.0,
            "source" => "pattern"
          }
        end
      end
      
      entities
    end
  end
  
  # Gazetteer-based entity recognizer
  class GazetteerEntityRecognizer
    attr_reader :entity_type, :terms, :case_sensitive
    
    def initialize(entity_type, terms = [], case_sensitive: false)
      @entity_type = entity_type
      @case_sensitive = case_sensitive
      @terms = build_term_set(terms)
    end
    
    # Add terms to the gazetteer
    def add_terms(terms)
      terms = [terms] unless terms.is_a?(Array)
      terms.each { |term| @terms.add(normalize_term(term)) }
      self
    end
    
    # Load terms from file
    def load_from_file(filepath)
      File.readlines(filepath).each do |line|
        term = line.strip
        add_terms(term) unless term.empty? || term.start_with?("#")
      end
      self
    end
    
    # Recognize entities using the gazetteer
    def recognize(text, tokenizer = nil)
      entities = []
      normalized_text = @case_sensitive ? text : text.downcase
      
      @terms.each do |term|
        pattern = @case_sensitive ? term : term.downcase
        pos = 0
        
        while (idx = normalized_text.index(pattern, pos))
          # Check word boundaries
          prev_char = idx > 0 ? text[idx - 1] : " "
          next_char = idx + pattern.length < text.length ? text[idx + pattern.length] : " "
          
          if word_boundary?(prev_char) && word_boundary?(next_char)
            entities << {
              "text" => text[idx, pattern.length],
              "label" => @entity_type,
              "start" => idx,
              "end" => idx + pattern.length,
              "confidence" => 1.0,
              "source" => "gazetteer"
            }
          end
          
          pos = idx + 1
        end
      end
      
      entities
    end
    
    private
    
    def build_term_set(terms)
      Set.new(terms.map { |term| normalize_term(term) })
    end
    
    def normalize_term(term)
      @case_sensitive ? term : term.downcase
    end
    
    def word_boundary?(char)
      char.match?(/\W/)
    end
  end
  
  # Hybrid NER that combines ML model with rules
  class HybridNER
    attr_reader :model_ner, :pattern_recognizers, :gazetteer_recognizers
    
    def initialize(model_id = nil, device: nil)
      @model_ner = model_id ? NER.from_pretrained(model_id, device: device) : nil
      @pattern_recognizers = []
      @gazetteer_recognizers = []
    end
    
    # Add a pattern-based recognizer
    def add_pattern_recognizer(entity_type, patterns)
      recognizer = PatternEntityRecognizer.new(entity_type, patterns)
      @pattern_recognizers << recognizer
      self
    end
    
    # Add a gazetteer-based recognizer
    def add_gazetteer_recognizer(entity_type, terms, **options)
      recognizer = GazetteerEntityRecognizer.new(entity_type, terms, **options)
      @gazetteer_recognizers << recognizer
      self
    end
    
    # Extract entities using all recognizers
    def extract_entities(text, confidence_threshold: 0.9)
      all_entities = []
      
      # Model-based entities
      if @model_ner
        model_entities = @model_ner.extract_entities(text, confidence_threshold: confidence_threshold)
        all_entities.concat(model_entities)
      end
      
      # Pattern-based entities
      @pattern_recognizers.each do |recognizer|
        pattern_entities = recognizer.recognize(text)
        all_entities.concat(pattern_entities)
      end
      
      # Gazetteer-based entities
      @gazetteer_recognizers.each do |recognizer|
        gazetteer_entities = recognizer.recognize(text)
        all_entities.concat(gazetteer_entities)
      end
      
      # Merge overlapping entities (prefer highest confidence)
      merge_entities(all_entities)
    end
    
    private
    
    def merge_entities(entities)
      # Sort by start position and confidence (descending)
      sorted = entities.sort_by { |e| [e["start"], -e["confidence"]] }
      
      merged = []
      sorted.each do |entity|
        # Check if entity overlaps with any already merged
        overlaps = merged.any? do |existing|
          entity["start"] < existing["end"] && entity["end"] > existing["start"]
        end
        
        merged << entity unless overlaps
      end
      
      merged.sort_by { |e| e["start"] }
    end
  end
end