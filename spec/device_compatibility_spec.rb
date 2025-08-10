require 'spec_helper'

RSpec.describe "DeviceCompatibility" do
  include DeviceHelpers
  
  # Test tensor operations on each device
  DeviceHelpers.devices_to_test.each do |device_type|
    describe "tensor operations on #{device_type}" do
      it "performs tensor operations" do
        skip_unless_device_available(device_type)
        
        device = create_device(device_type)
        
        # Test creating tensors directly on device
        tensor = Candle::Tensor.ones([4], device: device)
        expect(tensor.device.to_s).to eq(device.to_s)
        
        # Create and manipulate tensors from data
        data = [1.0, 2.0, 3.0, 4.0]
        tensor2 = Candle::Tensor.new(data, :f32).to_device(device)
        
        # Test various operations
        expect(tensor2.device.to_s).to eq(device.to_s)
        
        sum = tensor2.sum(0)
        expect(sum.to_f).to be_within(0.001).of(10.0)
        
        mean = tensor2.mean(0)
        expect(mean.to_f).to be_within(0.001).of(2.5)
        
        reshaped = tensor2.reshape([2, 2])
        expect(reshaped.shape).to eq([2, 2])
      end
    end
  end
  
  # Test EmbeddingModel on each device
  DeviceHelpers.devices_to_test.each do |device_type|
    describe "EmbeddingModel on #{device_type}" do
      let(:model) do
        @embedding_model ||= {}
        @embedding_model[device_type] ||= begin
          skip_unless_device_available(device_type)
          device = create_device(device_type)
          Candle::EmbeddingModel.from_pretrained("jinaai/jina-embeddings-v2-base-en", device: device)
        end
      end
      
      after(:all) do
        @embedding_model = nil
        GC.start
      end
      
      it "generates embeddings" do
        skip_unless_device_available(device_type)
        
        # Test single embedding
        text = "Hello world"
        embedding = model.embedding(text)
        
        expect(embedding.shape).to eq([1, 768])
        expect(embedding).to be_a(Candle::Tensor)
        
        # Test that embeddings are deterministic
        embedding2 = model.embedding(text)
        
        # Convert to arrays and compare
        emb1_array = embedding.to_a
        emb2_array = embedding2.to_a
        
        emb1_array[0].zip(emb2_array[0]).each do |v1, v2|
          expect(v1).to be_within(0.0001).of(v2)
        end
      end
    end
  end
  
  # Test Reranker on each device
  DeviceHelpers.devices_to_test.each do |device_type|
    describe "Reranker on #{device_type}" do
      let(:reranker) do
        @reranker ||= {}
        @reranker[device_type] ||= begin
          skip_unless_device_available(device_type)
          device = create_device(device_type)
          Candle::Reranker.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2", device: device)
        end
      end
      
      after(:all) do
        @reranker = nil
        GC.start
      end
      
      it "reranks documents" do
        skip_unless_device_available(device_type)
        
        query = "What is machine learning?"
        documents = [
          "Machine learning is a type of artificial intelligence.",
          "The weather is nice today.",
          "AI and ML are related fields."
        ]
        
        results = reranker.rerank(query, documents)
        
        # Verify results structure
        expect(results.length).to eq(3)
        expect(results[0]).to be_a(Hash)
        expect(results[0]).to have_key(:score)
        expect(results[0]).to have_key(:text)
        expect(results[0]).to have_key(:doc_id)
        
        # Verify sorting (highest score first)
        expect(results[0][:score]).to be >= results[1][:score]
        expect(results[1][:score]).to be >= results[2][:score]
        
        # The ML-related documents should score higher than weather
        ml_docs = results.select { |r| r[:text].downcase.include?("machine") || r[:text].downcase.include?("ai") }
        weather_doc = results.find { |r| r[:text].downcase.include?("weather") }
        
        expect(ml_docs.first[:score]).to be > weather_doc[:score]
      end
    end
  end
  
  # Test Reranker pooling methods
  DeviceHelpers.devices_to_test.each do |device_type|
    [:pooler, :cls, :mean].each do |pooling_method|
      describe "Reranker #{pooling_method} pooling on #{device_type}" do
        it "works with #{pooling_method} pooling" do
          skip_unless_device_available(device_type)
          
          device = create_device(device_type)
          
          reranker = Candle::Reranker.from_pretrained(device: device)
          
          query = "test query"
          documents = ["doc1", "doc2", "doc3"]
          
          results = reranker.rerank(query, documents, pooling_method: pooling_method.to_s)
          
          expect(results.length).to eq(3)
          results.each do |result|
            expect(result[:score]).to be_a(Float)
            expect(result[:score]).to be >= 0.0
            expect(result[:score]).to be <= 1.0
          end
        end
      end
    end
  end
  
  # Test LLM on each device
  DeviceHelpers.devices_to_test.each do |device_type|
    describe "LLM generation on #{device_type}" do
      let(:llm) do
        @llm ||= {}
        @llm[device_type] ||= begin
          skip_unless_device_available(device_type)
          device = create_device(device_type)
          Candle::LLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: device)
        end
      end
      
      after(:all) do
        @llm = nil
        GC.start
      end
      
      it "generates text" do
        skip_unless_device_available(device_type)
        
        config = Candle::GenerationConfig.new(
          max_length: 20,
          temperature: 0.0  # Deterministic
        )
        
        prompt = "The capital of France is"
        response = llm.generate(prompt, config: config)
        
        expect(response).to be_a(String)
        expect(response).not_to be_empty
        expect(response.length).to be > prompt.length
        
        # Should contain "Paris" in the response
        expect(response).to match(/paris/i)
      end
    end
  end
  
  # Test LLM streaming on each device
  DeviceHelpers.devices_to_test.each do |device_type|
    describe "LLM streaming on #{device_type}" do
      let(:llm) do
        @streaming_llm ||= {}
        @streaming_llm[device_type] ||= begin
          skip_unless_device_available(device_type)
          device = create_device(device_type)
          Candle::LLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: device)
        end
      end
      
      after(:all) do
        @streaming_llm = nil
        GC.start
      end
      
      it "streams tokens" do
        skip_unless_device_available(device_type)
        
        config = Candle::GenerationConfig.new(
          max_length: 10,
          temperature: 0.7
        )
        
        tokens = []
        llm.generate_stream("Once upon a time", config: config) do |token|
          tokens << token
        end
        
        expect(tokens).not_to be_empty
        expect(tokens.length).to be > 0
        
        # Verify we got individual tokens, not the full response at once
        expect(tokens.length).to be > 1
      end
    end
  end
  
  # Test NER on each device
  DeviceHelpers.devices_to_test.each do |device_type|
    describe "NER on #{device_type}" do
      let(:ner) do
        @ner ||= {}
        @ner[device_type] ||= begin
          skip_unless_device_available(device_type)
          device = create_device(device_type)
          Candle::NER.from_pretrained(
            "Babelscape/wikineural-multilingual-ner",
            device: device
          )
        end
      end
      
      after(:all) do
        @ner = nil
        GC.start
      end
      
      it "extracts entities" do
        skip_unless_device_available(device_type)
        
        # Test entity extraction
        text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
        entities = ner.extract_entities(text, confidence_threshold: 0.7)
        
        expect(entities).to be_an(Array)
        expect(entities).not_to be_empty
        
        # Check entity structure
        entities.each do |entity|
          expect(entity).to be_a(Hash)
          expect(entity).to have_key(:text)
          expect(entity).to have_key(:label)
          expect(entity).to have_key(:start)
          expect(entity).to have_key(:end)
          expect(entity).to have_key(:confidence)
          
          expect(entity[:text]).to be_a(String)
          expect(entity[:label]).to be_a(String)
          expect(entity[:start]).to be_an(Integer)
          expect(entity[:end]).to be_an(Integer)
          expect(entity[:confidence]).to be_a(Float)
          
          # Verify confidence threshold
          expect(entity[:confidence]).to be >= 0.7
        end
        
        # Verify we found expected entities
        entity_labels = entities.map { |e| e[:label] }
        
        # Should find organization (Apple Inc.) and person (Steve Jobs)
        expect(entity_labels).to include("ORG").or include("CORP")
        expect(entity_labels).to include("PER").or include("PERSON")
        
        # Test token predictions
        tokens = ner.predict_tokens("John works at Google")
        
        expect(tokens).to be_an(Array)
        expect(tokens).not_to be_empty
        
        tokens.each do |token_info|
          expect(token_info).to be_a(Hash)
          expect(token_info).to have_key("token")
          expect(token_info).to have_key("label")
          expect(token_info).to have_key("confidence")
          expect(token_info).to have_key("probabilities")
        end
      end
      
      it "respects confidence thresholds" do
        skip_unless_device_available(device_type)
        
        text = "Microsoft Corporation is based in Redmond."
        
        # Test with different thresholds
        low_threshold_entities = ner.extract_entities(text, confidence_threshold: 0.5)
        high_threshold_entities = ner.extract_entities(text, confidence_threshold: 0.9)
        
        # Lower threshold should return same or more entities
        expect(low_threshold_entities.length).to be >= high_threshold_entities.length
        
        # All high threshold entities should be in low threshold results
        high_texts = high_threshold_entities.map { |e| e[:text] }
        low_texts = low_threshold_entities.map { |e| e[:text] }
        
        high_texts.each do |text|
          expect(low_texts).to include(text)
        end
      end
    end
  end
  
  # Test Device.best
  describe "Device.best" do
    it "returns the best available device" do
      best = Candle::Device.best
      expect(best).to be_a(Candle::Device)
      
      # Should prefer Metal > CUDA > CPU
      # Check what's actually available on this system
      build_info = Candle::BuildInfo.summary
      
      if build_info[:metal_available]
        expect(best.to_s).to eq("metal")
      elsif build_info[:cuda_available]
        expect(best.to_s).to eq("cuda")
      else
        expect(best.to_s).to eq("cpu")
      end
    end
  end
  
  # Test deprecated DeviceUtils still works
  describe "DeviceUtils.best_device (deprecated)" do
    it "still works for backward compatibility" do
      # Test that DeviceUtils.best_device still works
      best = Candle::DeviceUtils.best_device
      expect(best).to be_a(Candle::Device)
      # Should return same as Device.best
      expect(best).to eq(Candle::Device.best)
      
      # Note: Testing the deprecation warning is tricky because warn goes to stderr
      # and capture_io doesn't always work reliably with it. The important thing
      # is that the method still works and returns the correct result.
    end
  end
end