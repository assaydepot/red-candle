#!/usr/bin/env ruby
# Script to pre-download all models needed for tests
# This avoids rate limiting during test runs

require 'bundler/setup'
require 'net/http'
require 'json'
require 'fileutils'

# Try to load candle, but handle if it's not compiled yet
begin
  require 'candle'
  CANDLE_AVAILABLE = true
rescue LoadError => e
  puts "Warning: Candle library not available (#{e.message})"
  puts "Please run 'rake compile' first to build the native extension."
  exit 1
end

class ModelDownloader
  HUGGINGFACE_HUB = "https://huggingface.co"
  
  def initialize
    @hf_token = ENV['HF_TOKEN']
    @hf_home = ENV['HF_HOME'] || File.expand_path("~/.cache/huggingface")
    @manifest_file = File.join(__dir__, '..', 'test', 'model_manifest.txt')
    @models_to_download = []
    @failed_downloads = []
    
    parse_manifest
  end
  
  def parse_manifest
    return unless File.exist?(@manifest_file)
    
    File.readlines(@manifest_file).each do |line|
      line = line.strip
      next if line.empty? || line.start_with?('#')
      
      parts = line.split(/\s+/)
      model_id = parts[0]
      
      # Check for specific files (e.g., GGUF)
      if parts[1] && parts[1].start_with?('gguf:')
        gguf_file = parts[1].sub('gguf:', '')
        @models_to_download << { type: :gguf, model_id: model_id, gguf_file: gguf_file }
      else
        @models_to_download << { type: :standard, model_id: model_id }
      end
    end
  end
  
  def download_all
    puts "=" * 60
    puts "Model Pre-download for CI"
    puts "=" * 60
    puts "HuggingFace cache: #{@hf_home}"
    puts "HF_TOKEN: #{@hf_token ? 'Set' : 'Not set (using anonymous access)'}"
    puts "Models to download: #{@models_to_download.size}"
    puts
    
    @models_to_download.each_with_index do |model_info, idx|
      puts "[#{idx + 1}/#{@models_to_download.size}] Downloading #{model_info[:model_id]}..."
      
      begin
        case model_info[:type]
        when :gguf
          download_gguf_model(model_info[:model_id], model_info[:gguf_file])
        when :standard
          download_standard_model(model_info[:model_id])
        end
        
        puts "  ✓ Successfully cached"
      rescue => e
        puts "  ✗ Failed: #{e.message}"
        @failed_downloads << model_info
        
        # Don't fail on individual model errors, continue with others
        if e.message.include?("429") || e.message.include?("Too Many Requests")
          puts "  ⚠ Rate limited. Waiting 30 seconds..."
          sleep 30
        end
      end
      
      # Small delay between downloads to be nice to HuggingFace
      sleep 1
    end
    
    print_summary
  end
  
  private
  
  def download_standard_model(model_id)
    # Try different model types to trigger downloads
    case model_id
    when /bert-base-uncased|gpt2/
      # Tokenizer
      puts "  Type: Tokenizer"
      Candle::Tokenizer.from_pretrained(model_id)
      
    when /sentence-transformers|jina.*embedding/
      # Embedding model
      puts "  Type: Embedding Model"
      # Determine model type
      model_type = case model_id
      when /minilm/i
        Candle::EmbeddingModelType::MINILM
      when /jina/i
        nil  # Auto-detect
      else
        nil
      end
      
      Candle::EmbeddingModel.from_pretrained(
        model_id,
        tokenizer: model_id,
        model_type: model_type
      )
      
    when /cross-encoder/
      # Reranker
      puts "  Type: Reranker"
      Candle::Reranker.from_pretrained(model_id)
      
    when /NER/
      # NER model
      puts "  Type: NER Model"
      Candle::NER.from_pretrained(model_id)
      
    when /phi-2/
      # LLM
      puts "  Type: LLM (this may take a while...)"
      # Just trigger the download, don't actually load
      # We'll use a subprocess to avoid memory issues
      system("ruby", "-e", "require 'candle'; Candle::LLM.from_pretrained('#{model_id}')")
      
    else
      puts "  Type: Unknown, attempting as tokenizer"
      Candle::Tokenizer.from_pretrained(model_id)
    end
  end
  
  def download_gguf_model(model_id, gguf_file)
    puts "  Type: GGUF Model"
    puts "  File: #{gguf_file}"
    
    # GGUF models download differently
    # The Ruby library will handle the download when we try to load
    Candle::LLM.from_pretrained(
      model_id,
      gguf_file: gguf_file,
      device: Candle::Device.cpu
    )
  end
  
  def print_summary
    puts
    puts "=" * 60
    puts "Download Summary"
    puts "=" * 60
    
    successful = @models_to_download.size - @failed_downloads.size
    puts "✓ Successfully cached: #{successful}/#{@models_to_download.size}"
    
    if @failed_downloads.any?
      puts "✗ Failed downloads: #{@failed_downloads.size}"
      @failed_downloads.each do |model_info|
        puts "  - #{model_info[:model_id]}"
      end
      
      puts
      puts "Note: Failed downloads will be retried during test execution."
      puts "Consider adding an HF_TOKEN for better rate limits."
    end
    
    # Check cache directory size
    if Dir.exist?(@hf_home)
      size_output = `du -sh #{@hf_home} 2>/dev/null`.strip.split("\t").first
      puts
      puts "Cache size: #{size_output}"
    end
  end
end

# Run the downloader
if __FILE__ == $0
  downloader = ModelDownloader.new
  downloader.download_all
end