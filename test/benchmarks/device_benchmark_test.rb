require 'test_helper'
require 'benchmark'

class DeviceBenchmarkTest < Minitest::Test
  include DeviceTestHelper
  
  # Skip benchmarks unless explicitly requested via CANDLE_RUN_BENCHMARKS=true
  # This only applies when running the full test suite (rake test)
  # When running rake test:benchmark directly, benchmarks always run
  def setup
    # Skip if this appears to be a full test run and benchmarks weren't requested
    unless ENV['CANDLE_RUN_BENCHMARKS'] == 'true'
      skip("Set CANDLE_RUN_BENCHMARKS=true to run benchmarks with full test suite")
    end
  end
  
  # Benchmark tensor operations across devices
  def test_tensor_operation_performance
    results = {}
    
    DeviceTestHelper.devices_to_test.each do |device_type|
      next unless available_devices[device_type]
      
      device = create_device(device_type)
      
      # Benchmark various tensor sizes
      sizes = [100, 1000, 10000, 100000]
      
      results[device_type] = {}
      
      sizes.each do |size|
        # Benchmark tensor creation
        creation_time = Benchmark.realtime do
          100.times { Candle::Tensor.ones([size], device: device) }
        end
        
        # Benchmark tensor operations
        tensor = Candle::Tensor.randn([size], device: device)
        
        operation_time = Benchmark.realtime do
          100.times do
            tensor.sum(0)
            tensor.mean(0)
            tensor.reshape([size])
          end
        end
        
        results[device_type][size] = creation_time + operation_time
      end
    end
    
    print_benchmark_results("Tensor Operations", results)
  end
  
  # Benchmark EmbeddingModel performance
  def test_embedding_model_performance
    results = {}
    texts = [
      "Short text",
      "A medium length sentence that contains more words to process",
      "This is a much longer text that should take more time to process. " * 10
    ]
    
    DeviceTestHelper.devices_to_test.each do |device_type|
      next unless available_devices[device_type]
      
      device = create_device(device_type)
      
      # Load model once
      model = Candle::EmbeddingModel.new(
        model_path: "jinaai/jina-embeddings-v2-base-en",
        device: device
      )
      
      results[device_type] = {}
      
      texts.each_with_index do |text, i|
        # Warm up
        model.embedding(text)
        
        # Benchmark
        time = Benchmark.realtime do
          10.times { model.embedding(text) }
        end
        
        results[device_type]["text_#{i}_length_#{text.length}"] = time / 10.0
      end
    end
    
    print_benchmark_results("EmbeddingModel", results)
  end
  
  # Benchmark Reranker performance
  def test_reranker_performance
    results = {}
    
    query = "What is machine learning?"
    document_sets = {
      small: Array.new(5) { |i| "Document #{i}: Machine learning is a field of AI." },
      medium: Array.new(20) { |i| "Document #{i}: Machine learning is a field of AI." },
      large: Array.new(100) { |i| "Document #{i}: Machine learning is a field of AI." }
    }
    
    DeviceTestHelper.devices_to_test.each do |device_type|
      next unless available_devices[device_type]
      
      device = create_device(device_type)
      
      reranker = Candle::Reranker.new(
        model_path: "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: device
      )
      
      results[device_type] = {}
      
      document_sets.each do |size, documents|
        # Warm up
        reranker.rerank(query, documents)
        
        # Benchmark
        time = Benchmark.realtime do
          5.times { reranker.rerank(query, documents) }
        end
        
        results[device_type][size] = time / 5.0
      end
    end
    
    print_benchmark_results("Reranker", results)
  end
  
  # Benchmark LLM performance
  def test_llm_generation_performance
    skip("LLM tests skipped via CANDLE_TEST_SKIP_LLM") if ENV['CANDLE_TEST_SKIP_LLM'] == 'true'
    results = {}
    
    prompts = {
      short: "Hi",
      medium: "What is the capital of France?",
      long: "Explain the theory of relativity in simple terms."
    }
    
    DeviceTestHelper.devices_to_test.each do |device_type|
      next unless available_devices[device_type]
      
      device = create_device(device_type)
      
      llm = Candle::LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device)
      
      config = Candle::GenerationConfig.new(
        max_length: 50,
        temperature: 0.7
      )
      
      results[device_type] = {}
      
      prompts.each do |size, prompt|
        # Warm up
        llm.generate(prompt, config)
        
        # Benchmark
        time = Benchmark.realtime do
          3.times { llm.generate(prompt, config) }
        end
        
        results[device_type][size] = time / 3.0
      end
    end
    
    print_benchmark_results("LLM Generation", results)
  end
  
  # Compare device transfer overhead
  def test_device_transfer_overhead
    skip_unless_device_available(:cpu)
    
    results = {}
    sizes = [100, 1000, 10000, 100000]
    
    # Test CPU to GPU transfers
    DeviceTestHelper.devices_to_test.reject { |d| d == :cpu }.each do |device_type|
      next unless available_devices[device_type]
      
      device = create_device(device_type)
      results[device_type] = {}
      
      sizes.each do |size|
        # Create on CPU
        cpu_tensor = Candle::Tensor.randn([size])
        
        # Measure transfer time
        transfer_time = Benchmark.realtime do
          100.times { cpu_tensor.to_device(device) }
        end
        
        # Measure direct creation time
        direct_time = Benchmark.realtime do
          100.times { Candle::Tensor.randn([size], device: device) }
        end
        
        results[device_type][size] = {
          transfer: transfer_time / 100.0,
          direct: direct_time / 100.0,
          overhead: (transfer_time - direct_time) / 100.0
        }
      end
    end
    
    print_transfer_results(results)
  end
  
  private
  
  def print_benchmark_results(title, results)
    puts "\n" + "=" * 60
    puts "#{title} Benchmark Results"
    puts "=" * 60
    
    results.each do |device, timings|
      puts "\n#{device.to_s.upcase}:"
      timings.each do |test, time|
        puts "  #{test}: #{'%.4f' % time}s"
      end
    end
    
    # Find best device for each test
    if results.any?
      puts "\nBest devices:"
      first_device_tests = results.values.first.keys
      
      first_device_tests.each do |test|
        best_device = results.min_by { |_, timings| timings[test] || Float::INFINITY }[0]
        best_time = results[best_device][test]
        puts "  #{test}: #{best_device.to_s.upcase} (#{'%.4f' % best_time}s)"
      end
    end
  end
  
  def print_transfer_results(results)
    puts "\n" + "=" * 60
    puts "Device Transfer Overhead Results"
    puts "=" * 60
    
    results.each do |device, sizes|
      puts "\nCPU to #{device.to_s.upcase} transfer overhead:"
      sizes.each do |size, timings|
        overhead_percent = (timings[:overhead] / timings[:direct]) * 100
        puts "  Size #{size}: #{'%.6f' % timings[:overhead]}s (#{'%.1f' % overhead_percent}% overhead)"
      end
    end
  end
end