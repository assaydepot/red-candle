# frozen_string_literal: true

require "bundler/gem_tasks"
require "rake/testtask"
require "rake/extensiontask"

task default: :test
Rake::TestTask.new do |t|
  t.deps << :compile
  t.libs << "test"
  t.test_files = FileList["test/**/*_test.rb"]
end

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

desc "benchmark"
task bench: :compile do
  ruby "test/bench.rb"
end

desc "Run device compatibility tests"
task device_test: :compile do
  puts "Running comprehensive device tests..."
  sh "ruby examples/comprehensive_device_test.rb --verbose"
end

desc "Run device compatibility tests with benchmarks"
task "device_test:benchmark" => :compile do
  puts "Running device tests with performance benchmarks..."
  sh "ruby examples/comprehensive_device_test.rb --verbose --benchmark"
end

desc "Run device tests on specific devices"
task "device_test:cpu" => :compile do
  sh "ruby examples/comprehensive_device_test.rb --devices cpu --verbose"
end

task "device_test:metal" => :compile do
  sh "ruby examples/comprehensive_device_test.rb --devices metal --verbose"
end

task "device_test:cuda" => :compile do
  sh "ruby examples/comprehensive_device_test.rb --devices cuda --verbose"
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
