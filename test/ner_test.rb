# frozen_string_literal: true

require_relative "test_helper"

class NERTest < Minitest::Test
  def test_pattern_recognizer
    recognizer = Candle::PatternEntityRecognizer.new("GENE")
    recognizer.add_pattern(/\b[A-Z][A-Z0-9]{2,}\b/)
    recognizer.add_pattern(/\bCD\d+\b/)
    
    text = "TP53 and BRCA1 mutations are common. CD4+ cells express CD8."
    entities = recognizer.recognize(text)
    
    assert entities.any? { |e| e["text"] == "TP53" }
    assert entities.any? { |e| e["text"] == "BRCA1" }
    assert entities.any? { |e| e["text"] == "CD4" }
    assert entities.any? { |e| e["text"] == "CD8" }
    
    entities.each do |e|
      assert_equal "GENE", e["label"]
      assert_equal 1.0, e["confidence"]
      assert_equal "pattern", e["source"]
    end
  end
  
  def test_gazetteer_recognizer
    genes = ["TP53", "BRCA1", "EGFR"]
    recognizer = Candle::GazetteerEntityRecognizer.new("CANCER_GENE", genes)
    
    text = "TP53 mutations affect BRCA1 function but not ABC1."
    entities = recognizer.recognize(text)
    
    assert_equal 2, entities.length
    assert entities.any? { |e| e["text"] == "TP53" }
    assert entities.any? { |e| e["text"] == "BRCA1" }
    # ABC1 not in gazetteer, so not recognized
    refute entities.any? { |e| e["text"] == "ABC1" }
  end
  
  def test_gazetteer_case_sensitivity
    # Case insensitive
    recognizer_ci = Candle::GazetteerEntityRecognizer.new("GENE", ["TP53"], case_sensitive: false)
    entities = recognizer_ci.recognize("tp53 and TP53 and Tp53")
    assert_equal 3, entities.length
    
    # Case sensitive  
    recognizer_cs = Candle::GazetteerEntityRecognizer.new("GENE", ["TP53"], case_sensitive: true)
    entities = recognizer_cs.recognize("tp53 and TP53 and Tp53")
    assert_equal 1, entities.length
    assert_equal "TP53", entities.first["text"]
  end
  
  def test_hybrid_ner_without_model
    # Test hybrid NER with only pattern/gazetteer recognizers
    hybrid = Candle::HybridNER.new
    
    # Add pattern recognizer
    hybrid.add_pattern_recognizer("GENE", [/\b[A-Z]{2,}\d*[A-Z]?\d*\b/])
    
    # Add gazetteer
    hybrid.add_gazetteer_recognizer("PROTEIN", ["p53", "p16", "p21"])
    
    text = "TP53 encodes the p53 protein, while CDKN2A encodes p16."
    entities = hybrid.extract_entities(text)
    
    # Should find pattern-based genes
    genes = entities.select { |e| e["label"] == "GENE" }
    assert genes.any? { |e| e["text"] == "TP53" }
    assert genes.any? { |e| e["text"] == "CDKN2A" }
    
    # Should find gazetteer-based proteins
    proteins = entities.select { |e| e["label"] == "PROTEIN" }
    assert proteins.any? { |e| e["text"] == "p53" }
    assert proteins.any? { |e| e["text"] == "p16" }
  end
  
  def test_entity_overlap_resolution
    hybrid = Candle::HybridNER.new
    
    # Add overlapping patterns
    hybrid.add_pattern_recognizer("ALPHANUM", [/[A-Z]+\d+/])
    hybrid.add_pattern_recognizer("GENE", [/TP53/])
    
    text = "TP53 is important"
    entities = hybrid.extract_entities(text)
    
    # Should only have one entity due to overlap resolution
    assert_equal 1, entities.length
    # First match wins (ALPHANUM comes first)
    assert_equal "ALPHANUM", entities.first["label"]
  end
  
  def test_word_boundary_detection
    recognizer = Candle::GazetteerEntityRecognizer.new("GENE", ["TP5"])
    
    # Should match with word boundaries
    entities = recognizer.recognize("TP5 gene")
    assert_equal 1, entities.length
    
    # Should not match within words
    entities = recognizer.recognize("TP53 gene")
    assert_equal 0, entities.length
  end
  
  def test_ner_from_pretrained
    # Using a model that has tokenizer.json file
    ner = Candle::NER.from_pretrained("Babelscape/wikineural-multilingual-ner")
    assert_instance_of Candle::NER, ner
    
    # Test entity extraction
    text = "Apple Inc. was founded by Steve Jobs in Cupertino."
    entities = ner.extract_entities(text)
    
    # This model uses different label format (without B-/I- prefixes in some cases)
    assert entities.any? { |e| e["label"] =~ /ORG|MISC/ }
    assert entities.any? { |e| e["label"] =~ /PER/ }
    assert entities.any? { |e| e["label"] =~ /LOC/ }
  end
end