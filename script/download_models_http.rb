#!/usr/bin/env ruby
# Alternative model downloader using direct HTTP requests
# This doesn't require the Candle library to be compiled

require 'net/http'
require 'json'
require 'fileutils'
require 'open-uri'
require 'digest'

class HTTPModelDownloader
  HUGGINGFACE_API = "https://huggingface.co/api"
  HUGGINGFACE_CDN = "https://cdn-lfs.huggingface.co"
  
  def initialize
    @hf_token = ENV['HF_TOKEN']
    @hf_home = ENV['HF_HOME'] || File.expand_path("~/.cache/huggingface")
    @manifest_file = File.join(__dir__, '..', 'test', 'model_manifest.txt')
    @models = parse_manifest
    
    FileUtils.mkdir_p(@hf_home)
  end
  
  def parse_manifest
    models = []
    return models unless File.exist?(@manifest_file)
    
    File.readlines(@manifest_file).each do |line|
      line = line.strip
      next if line.empty? || line.start_with?('#')
      
      parts = line.split(/\s+/)
      model_id = parts[0]
      
      # Handle GGUF files specially
      if parts[1] && parts[1].start_with?('gguf:')
        gguf_file = parts[1].sub('gguf:', '')
        models << { model_id: model_id, type: :gguf, files: [gguf_file] }
      else
        # For standard models, download key files
        models << { model_id: model_id, type: :standard, files: standard_files_for(model_id) }
      end
    end
    
    models
  end
  
  def standard_files_for(model_id)
    # Common files needed for different model types
    base_files = ['config.json']
    
    case model_id
    when /tokenizer|bert-base-uncased|gpt2/
      base_files + ['tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'vocab.json', 'merges.txt']
    when /embedding|sentence-transformers/
      base_files + ['tokenizer.json', 'tokenizer_config.json', 'pytorch_model.bin', 'model.safetensors']
    when /reranker|cross-encoder/
      base_files + ['tokenizer.json', 'tokenizer_config.json', 'pytorch_model.bin', 'model.safetensors']
    when /NER/
      base_files + ['tokenizer.json', 'tokenizer_config.json', 'pytorch_model.bin', 'model.safetensors']
    when /phi-2|llama|mistral/
      # Large models - just download config for caching purposes
      base_files + ['tokenizer.json', 'tokenizer_config.json']
    else
      base_files
    end
  end
  
  def download_all
    puts "=" * 60
    puts "HTTP Model Pre-download for CI"
    puts "=" * 60
    puts "Cache directory: #{@hf_home}"
    puts "HF_TOKEN: #{@hf_token ? 'Set' : 'Not set'}"
    puts "Models to process: #{@models.size}"
    puts
    
    @models.each_with_index do |model_info, idx|
      model_id = model_info[:model_id]
      puts "[#{idx + 1}/#{@models.size}] Processing #{model_id}..."
      
      model_cache_dir = File.join(@hf_home, 'hub', "models--#{model_id.gsub('/', '--')}")
      
      # Check if already cached
      if Dir.exist?(model_cache_dir) && !Dir.empty?(model_cache_dir)
        puts "  ✓ Already cached at #{model_cache_dir}"
        next
      end
      
      # Create cache directory structure
      FileUtils.mkdir_p(File.join(model_cache_dir, 'snapshots'))
      FileUtils.mkdir_p(File.join(model_cache_dir, 'refs'))
      
      # Download files
      success = true
      model_info[:files].each do |filename|
        if download_file(model_id, filename, model_cache_dir)
          puts "    ✓ #{filename}"
        else
          puts "    ✗ #{filename} (not found or rate limited)"
          # Don't fail completely if optional files are missing
        end
      end
      
      # Small delay to be nice to HuggingFace
      sleep 0.5
    end
    
    puts
    puts "Download complete!"
    
    # Show cache size
    if Dir.exist?(@hf_home)
      size = `du -sh #{@hf_home} 2>/dev/null`.strip.split("\t").first rescue "unknown"
      puts "Cache size: #{size}"
    end
  end
  
  private
  
  def download_file(model_id, filename, cache_dir)
    url = "https://huggingface.co/#{model_id}/resolve/main/#{filename}"
    
    # Set up headers
    headers = {}
    headers['Authorization'] = "Bearer #{@hf_token}" if @hf_token
    
    # Create snapshot directory
    snapshot_dir = File.join(cache_dir, 'snapshots', 'main')
    FileUtils.mkdir_p(snapshot_dir)
    
    output_path = File.join(snapshot_dir, filename)
    
    # Skip if already exists
    if File.exist?(output_path) && File.size(output_path) > 0
      return true
    end
    
    begin
      # Download the file
      URI.open(url, headers) do |remote_file|
        File.open(output_path, 'wb') do |local_file|
          local_file.write(remote_file.read)
        end
      end
      
      # Create ref file
      File.write(File.join(cache_dir, 'refs', 'main'), 'main')
      
      true
    rescue OpenURI::HTTPError => e
      if e.message.include?('404')
        # File doesn't exist, that's okay for optional files
        false
      elsif e.message.include?('429')
        puts "    ⚠ Rate limited! Waiting 10 seconds..."
        sleep 10
        false
      else
        puts "    Error: #{e.message}"
        false
      end
    rescue => e
      puts "    Error: #{e.message}"
      false
    end
  end
end

# Run if executed directly
if __FILE__ == $0
  downloader = HTTPModelDownloader.new
  downloader.download_all
end