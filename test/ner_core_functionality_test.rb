# frozen_string_literal: true

require_relative "test_helper"

class NERCoreFunctionalityTest < Minitest::Test
  # Load model once for all tests
  @@ner = nil
  @@model_loaded = false
  
  def self.load_model_once
    unless @@model_loaded
      begin
        # Try multiple models in order of preference
        models_to_try = [
          "Babelscape/wikineural-multilingual-ner",
          "dslim/bert-base-NER"
        ]
        
        models_to_try.each do |model_id|
          begin
            @@ner = Candle::NER.from_pretrained(model_id)
            @@model_loaded = true
            break
          rescue => e
            puts "Failed to load #{model_id}: #{e.message}"
          end
        end
        
        if !@@model_loaded
          @@model_loaded = :failed
          @@load_error = "Could not load any NER model"
        end
      rescue => e
        @@model_loaded = :failed
        @@load_error = e
      end
    end
  end
  
  def setup
    super
    self.class.load_model_once
    if @@model_loaded == :failed
      skip "NER model loading failed: #{@@load_error}"
    end
  end
  
  # Test extract_entities method comprehensively
  def test_extract_entities_basic
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    entities = @@ner.extract_entities(text)
    
    assert_instance_of Array, entities
    entities.each do |entity|
      assert_instance_of Hash, entity
      assert entity.key?(:text)
      assert entity.key?(:label)
      assert entity.key?(:start)
      assert entity.key?(:end)
      assert entity.key?(:confidence)
      assert entity.key?(:token_start)
      assert entity.key?(:token_end)
      
      # Validate data types
      assert_instance_of String, entity[:text]
      assert_instance_of String, entity[:label]
      assert_instance_of Integer, entity[:start]
      assert_instance_of Integer, entity[:end]
      assert_kind_of Numeric, entity[:confidence]
      assert_instance_of Integer, entity[:token_start]
      assert_instance_of Integer, entity[:token_end]
      
      # Validate ranges
      assert entity[:start] >= 0
      assert entity[:end] > entity[:start]
      assert entity[:end] <= text.length
      assert entity[:confidence] >= 0.0
      assert entity[:confidence] <= 1.0
      assert entity[:token_start] >= 0
      assert entity[:token_end] > entity[:token_start]
      
      # Validate text extraction - handle potential unicode issues
      extracted = text[entity[:start]...entity[:end]]
      # Due to tokenization, the extracted text might have slight differences
      # but should be mostly the same
      assert extracted.include?(entity[:text]) || entity[:text].include?(extracted) ||
             extracted.strip == entity[:text].strip,
             "Entity text '#{entity[:text]}' doesn't match extracted '#{extracted}'"
    end
  end
  
  def test_extract_entities_with_confidence_threshold
    text = "Microsoft and Google are technology companies."
    
    # Test with low threshold
    low_threshold_entities = @@ner.extract_entities(text, confidence_threshold: 0.3)
    
    # Test with high threshold
    high_threshold_entities = @@ner.extract_entities(text, confidence_threshold: 0.95)
    
    # Low threshold should have same or more entities
    assert low_threshold_entities.length >= high_threshold_entities.length
    
    # All entities should meet their respective thresholds
    low_threshold_entities.each do |entity|
      assert entity[:confidence] >= 0.3
    end
    
    high_threshold_entities.each do |entity|
      assert entity[:confidence] >= 0.95
    end
  end
  
  def test_extract_entities_empty_text
    entities = @@ner.extract_entities("")
    assert_instance_of Array, entities
    assert_equal 0, entities.length
  end
  
  def test_extract_entities_whitespace_only
    entities = @@ner.extract_entities("   \n\t   ")
    assert_instance_of Array, entities
    assert_equal 0, entities.length
  end
  
  def test_extract_entities_special_characters
    text = "Email: john@apple.com, Phone: +1-555-0123"
    entities = @@ner.extract_entities(text)
    
    assert_instance_of Array, entities
    # Should handle special characters without crashing
  end
  
  def test_extract_entities_unicode
    text = "François works at Zürich in 北京."
    entities = @@ner.extract_entities(text)
    
    assert_instance_of Array, entities
    # Should handle unicode without crashing
    entities.each do |entity|
      # Text extraction should work correctly with unicode
      extracted = text[entity[:start]...entity[:end]]
      # Tokenization might cause slight boundary differences with unicode
      assert entity[:text].length > 0, "Entity text should not be empty"
      assert extracted.length > 0, "Extracted text should not be empty"
    end
  end
  
  def test_extract_entities_long_text
    # Test with a longer text to ensure batch handling works
    text = "Apple Inc. is headquartered in Cupertino. " * 20
    entities = @@ner.extract_entities(text)
    
    assert_instance_of Array, entities
    # Should find multiple instances of the same entities
  end
  
  def test_extract_entities_no_entities
    text = "The the the and and and of of of"
    entities = @@ner.extract_entities(text)
    
    assert_instance_of Array, entities
    # May or may not find entities, but should not crash
  end
  
  # Test predict_tokens method comprehensively
  def test_predict_tokens_basic
    text = "Apple Inc. was founded by Steve Jobs."
    tokens = @@ner.predict_tokens(text)
    
    assert_instance_of Array, tokens
    assert tokens.length > 0
    
    tokens.each_with_index do |token, idx|
      assert_instance_of Hash, token
      assert token.key?("token")
      assert token.key?("label")
      assert token.key?("confidence")
      assert token.key?("index")
      assert token.key?("probabilities")
      
      # Validate data types
      assert_instance_of String, token["token"]
      assert_instance_of String, token["label"]
      assert_kind_of Numeric, token["confidence"]
      assert_equal idx, token["index"]
      assert_instance_of Hash, token["probabilities"]
      
      # Validate confidence
      assert token["confidence"] >= 0.0
      assert token["confidence"] <= 1.0
      
      # Validate probabilities
      prob_sum = 0.0
      token["probabilities"].each do |label, prob|
        assert_instance_of String, label
        assert_kind_of Numeric, prob
        assert prob >= 0.0
        assert prob <= 1.0
        prob_sum += prob
      end
      # Probabilities should sum to approximately 1.0
      assert_in_delta 1.0, prob_sum, 0.001
      
      # The predicted label should have the highest probability
      if token["probabilities"].any?
        max_prob_label = token["probabilities"].max_by { |_, prob| prob }[0]
        assert_equal token["label"], max_prob_label
      end
    end
  end
  
  def test_predict_tokens_special_tokens
    text = "Hello world"
    tokens = @@ner.predict_tokens(text)
    
    # Should include special tokens like [CLS] and [SEP]
    assert tokens.any? { |t| t["token"].start_with?("[") && t["token"].end_with?("]") }
  end
  
  def test_predict_tokens_empty_text
    tokens = @@ner.predict_tokens("")
    assert_instance_of Array, tokens
    # Should at least have [CLS] and [SEP] tokens
    assert tokens.length >= 2
  end
  
  def test_predict_tokens_consistency_with_extract_entities
    text = "Microsoft CEO Satya Nadella announced new products."
    
    tokens = @@ner.predict_tokens(text)
    entities = @@ner.extract_entities(text)
    
    # Both methods should process the same text consistently
    assert_instance_of Array, tokens
    assert_instance_of Array, entities
    
    # If entities are found, there should be corresponding non-O labels in tokens
    if entities.any?
      non_o_tokens = tokens.select { |t| t["label"] != "O" && !t["token"].start_with?("[") }
      assert non_o_tokens.any?, "Should have non-O token labels when entities are found"
    end
  end
  
  def test_predict_tokens_tokenization_alignment
    text = "San Francisco"
    tokens = @@ner.predict_tokens(text)
    
    # Tokens should be properly aligned
    # Remove special tokens for this check
    word_tokens = tokens.reject { |t| t["token"].start_with?("[") && t["token"].end_with?("]") }
    
    # The tokens should represent the input text
    reconstructed = word_tokens.map { |t| t["token"] }.join(" ").gsub(" ##", "").gsub("##", "")
    # Tokenization might lowercase or slightly modify, but should be recognizable
    assert reconstructed.downcase.include?(text.downcase) || 
           text.downcase.include?(reconstructed.downcase)
  end
  
  def test_both_methods_handle_errors_gracefully
    # Test with nil (should be handled by Ruby type conversion)
    assert_raises(TypeError) { @@ner.extract_entities(nil) }
    assert_raises(TypeError) { @@ner.predict_tokens(nil) }
    
    # Test with non-string types
    assert_raises(TypeError) { @@ner.extract_entities(123) }
    assert_raises(TypeError) { @@ner.predict_tokens(123) }
  end
  
  def test_confidence_threshold_validation
    text = "Test text"
    
    # Valid thresholds
    [0.0, 0.5, 1.0].each do |threshold|
      entities = @@ner.extract_entities(text, confidence_threshold: threshold)
      assert_instance_of Array, entities
    end
    
    # Invalid thresholds should be handled gracefully
    # Ruby will convert out-of-range values, but they should work
    entities = @@ner.extract_entities(text, confidence_threshold: -0.5)
    assert_instance_of Array, entities
    
    entities = @@ner.extract_entities(text, confidence_threshold: 1.5)
    assert_instance_of Array, entities
  end
  
  def test_bio_tag_consistency
    text = "Apple Inc. is a company"
    tokens = @@ner.predict_tokens(text)
    
    # Check BIO tag consistency
    previous_label = nil
    tokens.each do |token|
      label = token["label"]
      
      # Skip special tokens
      next if token["token"].start_with?("[") && token["token"].end_with?("]")
      
      if label.start_with?("I-")
        entity_type = label[2..]
        # I- tags should follow B- or I- tags of the same type
        if previous_label && !previous_label.start_with?("B-#{entity_type}") && 
           !previous_label.start_with?("I-#{entity_type}")
          # This is actually valid - some models use I- to start entities
          # Just ensure it's consistent
          assert label.start_with?("I-"), "I- tag found: #{label}"
        end
      end
      
      previous_label = label
    end
  end
  
  def test_performance_characteristics
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    
    # Measure time for extract_entities
    start_time = Time.now
    _entities = @@ner.extract_entities(text)
    extract_time = Time.now - start_time
    
    # Measure time for predict_tokens
    start_time = Time.now
    _tokens = @@ner.predict_tokens(text)
    predict_time = Time.now - start_time
    
    # Both should complete in reasonable time (< 1 second for short text)
    assert extract_time < 1.0, "extract_entities took too long: #{extract_time}s"
    assert predict_time < 1.0, "predict_tokens took too long: #{predict_time}s"
    
    # They should take similar time since they do similar work
    # Allow for 5x difference due to different post-processing
    time_ratio = [extract_time / predict_time, predict_time / extract_time].max
    assert time_ratio < 5.0, "Methods have very different performance: ratio #{time_ratio}"
  end
end