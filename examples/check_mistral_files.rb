#!/usr/bin/env ruby

# Quick script to check what files are available for Mistral models on HuggingFace
# This helps understand the actual file structure

require 'net/http'
require 'json'
require 'uri'

def check_model_files(model_id)
  puts "Checking files for: #{model_id}"
  
  # HuggingFace API endpoint
  uri = URI("https://huggingface.co/api/models/#{model_id}")
  
  begin
    response = Net::HTTP.get_response(uri)
    if response.code == '200'
      data = JSON.parse(response.body)
      
      # List siblings (files in the repo)
      if data['siblings']
        puts "\nFiles in repository:"
        data['siblings'].each do |file|
          filename = file['rfilename']
          size = file['size']
          size_mb = size ? (size / 1024.0 / 1024.0).round(2) : 0
          
          # Highlight model files
          if filename.match?(/\.(safetensors|bin|pt|pth)$/)
            puts "  * #{filename} (#{size_mb} MB)"
          else
            puts "    #{filename}"
          end
        end
      end
    else
      puts "Error: HTTP #{response.code} - #{response.message}"
    end
  rescue => e
    puts "Error checking model: #{e.message}"
  end
  
  puts "\n" + "-" * 50 + "\n"
end

# Check various Mistral models
models = [
  "mistralai/Mistral-7B-Instruct-v0.1",
  "mistralai/Mistral-7B-Instruct-v0.2", 
  "mistralai/Mistral-7B-Instruct-v0.3",
  "mistralai/Mistral-7B-v0.1"
]

models.each { |model| check_model_files(model) }