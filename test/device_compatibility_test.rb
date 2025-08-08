require 'test_helper'

class DeviceCompatibilityTest < Minitest::Test
  include DeviceTestHelper
  
  # Test tensor operations on each device
  DeviceTestHelper.devices_to_test.each do |device_type|
    define_method "test_tensor_operations_on_#{device_type}" do
      skip_unless_device_available(device_type)
      
      device = create_device(device_type)
      
      # Test creating tensors directly on device
      tensor = Candle::Tensor.ones([4], device: device)
      assert_equal device.to_s, tensor.device.to_s
      
      # Create and manipulate tensors from data
      data = [1.0, 2.0, 3.0, 4.0]
      tensor2 = Candle::Tensor.new(data, :f32).to_device(device)
      
      # Test various operations
      assert_equal device.to_s, tensor2.device.to_s
      
      sum = tensor2.sum(0)
      assert_in_delta 10.0, sum.to_f, 0.001
      
      mean = tensor2.mean(0)
      assert_in_delta 2.5, mean.to_f, 0.001
      
      reshaped = tensor2.reshape([2, 2])
      assert_equal [2, 2], reshaped.shape
    end
  end
  
  # Test EmbeddingModel on each device
  DeviceTestHelper.devices_to_test.each do |device_type|
    define_method "test_embedding_model_on_#{device_type}" do
      skip_unless_device_available(device_type)
      
      device = create_device(device_type)
      
      model = Candle::EmbeddingModel.from_pretrained("jinaai/jina-embeddings-v2-base-en", device: device)
      
      # Test single embedding
      text = "Hello world"
      embedding = model.embedding(text)
      
      assert_equal [1, 768], embedding.shape
      assert_kind_of Candle::Tensor, embedding
      
      # Test that embeddings are deterministic
      embedding2 = model.embedding(text)
      
      # Convert to arrays and compare
      emb1_array = embedding.to_a
      emb2_array = embedding2.to_a
      
      emb1_array[0].zip(emb2_array[0]).each do |v1, v2|
        assert_in_delta v1, v2, 0.0001
      end
    end
  end
  
  # Test Reranker on each device
  DeviceTestHelper.devices_to_test.each do |device_type|
    define_method "test_reranker_on_#{device_type}" do
      skip_unless_device_available(device_type)
      
      device = create_device(device_type)
      
      reranker = Candle::Reranker.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2", device: device)
      
      query = "What is machine learning?"
      documents = [
        "Machine learning is a type of artificial intelligence.",
        "The weather is nice today.",
        "AI and ML are related fields."
      ]
      
      results = reranker.rerank(query, documents)
      
      # Verify results structure
      assert_equal 3, results.length
      assert_kind_of Hash, results[0]
      assert results[0].key?(:score)
      assert results[0].key?(:text)
      assert results[0].key?(:doc_id)
      
      # Verify sorting (highest score first)
      assert results[0][:score] >= results[1][:score]
      assert results[1][:score] >= results[2][:score]
      
      # The ML-related documents should score higher than weather
      ml_docs = results.select { |r| r[:text].downcase.include?("machine") || r[:text].downcase.include?("ai") }
      weather_doc = results.find { |r| r[:text].downcase.include?("weather") }
      
      assert ml_docs.first[:score] > weather_doc[:score]
    end
  end
  
  # Test Reranker pooling methods
  DeviceTestHelper.devices_to_test.each do |device_type|
    [:pooler, :cls, :mean].each do |pooling_method|
      define_method "test_reranker_#{pooling_method}_pooling_on_#{device_type}" do
        skip_unless_device_available(device_type)
        
        device = create_device(device_type)
        
        reranker = Candle::Reranker.from_pretrained(device: device)
        
        query = "test query"
        documents = ["doc1", "doc2", "doc3"]
        
        results = reranker.rerank(query, documents, pooling_method: pooling_method.to_s)
        
        assert_equal 3, results.length
        results.each do |result|
          assert_kind_of Float, result[:score]
          assert result[:score] >= 0.0
          assert result[:score] <= 1.0
        end
      end
    end
  end
  
  # Test LLM on each device
  DeviceTestHelper.devices_to_test.each do |device_type|
    define_method "test_llm_generation_on_#{device_type}" do
      skip_unless_device_available(device_type)
      
      device = create_device(device_type)
      
      # Use Mistral model (currently the only supported type)
      llm = Candle::LLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: device)
      
      config = Candle::GenerationConfig.new(
        max_length: 20,
        temperature: 0.0  # Deterministic
      )
      
      prompt = "The capital of France is"
      response = llm.generate(prompt, config: config)
      
      assert_kind_of String, response
      refute_empty response
      assert response.length > prompt.length
      
      # Should contain "Paris" in the response
      assert_match(/paris/i, response)
    end
  end
  
  # Test LLM streaming on each device
  DeviceTestHelper.devices_to_test.each do |device_type|
    define_method "test_llm_streaming_on_#{device_type}" do
      skip_unless_device_available(device_type)
      
      device = create_device(device_type)
      
      llm = Candle::LLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: device)
      
      config = Candle::GenerationConfig.new(
        max_length: 10,
        temperature: 0.7
      )
      
      tokens = []
      llm.generate_stream("Once upon a time", config: config) do |token|
        tokens << token
      end
      
      refute_empty tokens
      assert tokens.length > 0
      
      # Verify we got individual tokens, not the full response at once
      assert tokens.length > 1
    end
  end
  
  # Test NER on each device
  DeviceTestHelper.devices_to_test.each do |device_type|
    define_method "test_ner_on_#{device_type}" do
      skip_unless_device_available(device_type)
      
      device = create_device(device_type)
      
      # Load NER model on specific device
      ner = Candle::NER.from_pretrained(
        "Babelscape/wikineural-multilingual-ner",
        device: device
      )
      
      # Test entity extraction
      text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
      entities = ner.extract_entities(text, confidence_threshold: 0.7)
      
      assert_kind_of Array, entities
      refute_empty entities
      
      # Check entity structure
      entities.each do |entity|
        assert_kind_of Hash, entity
        assert entity.key?("text")
        assert entity.key?("label")
        assert entity.key?("start")
        assert entity.key?("end")
        assert entity.key?("confidence")
        
        assert_kind_of String, entity["text"]
        assert_kind_of String, entity["label"]
        assert_kind_of Integer, entity["start"]
        assert_kind_of Integer, entity["end"]
        assert_kind_of Float, entity["confidence"]
        
        # Verify confidence threshold
        assert entity["confidence"] >= 0.7
      end
      
      # Verify we found expected entities
      entity_labels = entities.map { |e| e["label"] }
      
      # Should find organization (Apple Inc.) and person (Steve Jobs)
      assert entity_labels.include?("ORG") || entity_labels.include?("CORP")
      assert entity_labels.include?("PER") || entity_labels.include?("PERSON")
      
      # Test token predictions
      tokens = ner.predict_tokens("John works at Google")
      
      assert_kind_of Array, tokens
      refute_empty tokens
      
      tokens.each do |token_info|
        assert_kind_of Hash, token_info
        assert token_info.key?("token")
        assert token_info.key?("label")
        assert token_info.key?("confidence")
        assert token_info.key?("probabilities")
      end
    end
  end
  
  # Test NER with different confidence thresholds on each device
  DeviceTestHelper.devices_to_test.each do |device_type|
    define_method "test_ner_confidence_thresholds_on_#{device_type}" do
      skip_unless_device_available(device_type)
      
      device = create_device(device_type)
      
      ner = Candle::NER.from_pretrained(
        "Babelscape/wikineural-multilingual-ner",
        device: device
      )
      
      text = "Microsoft Corporation is based in Redmond."
      
      # Test with different thresholds
      low_threshold_entities = ner.extract_entities(text, confidence_threshold: 0.5)
      high_threshold_entities = ner.extract_entities(text, confidence_threshold: 0.9)
      
      # Lower threshold should return same or more entities
      assert low_threshold_entities.length >= high_threshold_entities.length
      
      # All high threshold entities should be in low threshold results
      high_texts = high_threshold_entities.map { |e| e["text"] }
      low_texts = low_threshold_entities.map { |e| e["text"] }
      
      high_texts.each do |text|
        assert low_texts.include?(text)
      end
    end
  end
  
  # Test Device.best
  def test_device_best
    best = Candle::Device.best
    assert_kind_of Candle::Device, best
    
    # Should prefer Metal > CUDA > CPU
    if available_devices[:metal]
      assert_equal "metal", best.to_s
    elsif available_devices[:cuda]
      assert_equal "cuda", best.to_s
    else
      assert_equal "cpu", best.to_s
    end
  end
  
  # Test deprecated DeviceUtils still works
  def test_device_utils_best_device_deprecated
    # Test that DeviceUtils.best_device still works
    best = Candle::DeviceUtils.best_device
    assert_kind_of Candle::Device, best
    # Should return same as Device.best
    assert_equal Candle::Device.best, best
    
    # Note: Testing the deprecation warning is tricky because warn goes to stderr
    # and capture_io doesn't always work reliably with it. The important thing
    # is that the method still works and returns the correct result.
  end
end