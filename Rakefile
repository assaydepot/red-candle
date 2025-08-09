# frozen_string_literal: true

require "bundler/gem_tasks"
require "rake/extensiontask"
require "rspec/core/rake_task"

task default: :spec

spec = Bundler.load_gemspec("candle.gemspec")
Rake::ExtensionTask.new("candle", spec) do |c|
  c.lib_dir = "lib/candle"
  c.cross_compile = true
  c.cross_platform = %w[
    aarch64-linux
    arm64-darwin
    x64-mingw-ucrt
    x64-mingw32
    x86_64-darwin
    x86_64-linux
    x86_64-linux-musl
  ]
end


namespace :doc do
  task default: %i[rustdoc yard]

  desc "Generate YARD documentation"
  task :yard do
    sh <<~CMD
      yard doc \
        --plugin rustdoc -- lib tmp/doc/candle.json
    CMD
  end

  desc "Generate Rust documentation as JSON"
  task :rustdoc do
    sh <<~CMD
      cargo +nightly rustdoc \
        --target-dir tmp/doc/target \
        -p candle \
        -- -Zunstable-options --output-format json \
        --document-private-items
    CMD

    cp "tmp/doc/target/doc/candle.json", "tmp/doc/candle.json"
  end
end

task doc: "doc:default"

namespace :rust do
  desc "Run Rust tests with code coverage"
  namespace :coverage do
    desc "Generate HTML coverage report"
    task :html do
      sh "cd ext/candle && cargo llvm-cov --html"
      puts "Coverage report generated in target/llvm-cov/html/index.html"
    end

    desc "Generate coverage report in terminal"
    task :report do
      sh "cd ext/candle && cargo llvm-cov"
    end

    desc "Show coverage summary"
    task :summary do
      sh "cd ext/candle && cargo llvm-cov --summary-only"
    end

    desc "Generate lcov format coverage report"
    task :lcov do
      sh "cd ext/candle && cargo llvm-cov --lcov --output-path ../../coverage/lcov.info"
      puts "LCOV report generated in coverage/lcov.info"
    end

    desc "Clean coverage data"
    task :clean do
      sh "cd ext/candle && cargo llvm-cov clean"
    end
  end

  desc "Run Rust tests"
  task :test do
    sh "cd ext/candle && cargo test"
  end
end

desc "Run Rust tests with coverage (alias)"
task "coverage:rust" => "rust:coverage:html"

# RSpec tasks
desc "Run RSpec tests"
RSpec::Core::RakeTask.new(:spec) do |t|
  t.rspec_opts = "--format progress"
end

# Add compile as a dependency for spec task
task spec: :compile

namespace :spec do
  desc "Run RSpec tests with all devices"
  RSpec::Core::RakeTask.new(:device) do |t|
    t.rspec_opts = "--format documentation --tag device"
  end
  
  desc "Run RSpec tests with coverage"
  task :coverage do
    ENV['COVERAGE'] = 'true'
    Rake::Task["spec"].invoke
  end
  
  desc "Run RSpec tests in parallel (requires parallel_tests gem)"
  task :parallel do
    begin
      require 'parallel_tests'
      sh "parallel_rspec spec/"
    rescue LoadError
      puts "parallel_tests gem not installed. Run: gem install parallel_tests"
    end
  end
  
  desc "Run specific device tests"
  %w[cpu metal cuda].each do |device|
    desc "Run tests on #{device.upcase} only"
    task "device:#{device}" => :compile do
      ENV['CANDLE_TEST_DEVICES'] = device
      sh "rspec spec/device_compatibility_spec.rb --format documentation"
    end
  end
  
  desc "Run LLM tests for specific models"
  namespace :llm do
    desc "Run tests for Gemma models"
    task :gemma => :compile do
      sh "rspec spec/llm/gemma_spec.rb --format documentation"
    end
    
    desc "Run tests for Phi models"
    task :phi => :compile do
      sh "rspec spec/llm/phi_spec.rb --format documentation"
    end
    
    desc "Run tests for Qwen models"
    task :qwen => :compile do
      sh "rspec spec/llm/qwen_spec.rb --format documentation"
    end
    
    desc "Run tests for Mistral models"
    task :mistral => :compile do
      sh "rspec spec/llm/mistral_spec.rb --format documentation"
    end
    
    desc "Run tests for Llama models"
    task :llama => :compile do
      sh "rspec spec/llm/llama_spec.rb --format documentation"
    end

    desc "Run tests for TinyLlama models"
    task :tinyllama => :compile do
      sh "rspec spec/llm/tinyllama_spec.rb --format documentation"
    end

    desc "Run all LLM tests (WARNING: downloads large models)"
    task :all => [:gemma, :phi, :qwen, :mistral, :llama, :tinyllama]
  end
end
