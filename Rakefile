# frozen_string_literal: true

require "bundler/gem_tasks"
require "rake/testtask"
require "rake/extensiontask"

ENV['CANDLE_TEST_SKIP_LLM'] = 'true'

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

desc "Run device compatibility tests"
Rake::TestTask.new("test:device") do |t|
  t.deps << :compile
  t.libs << "test"
  t.test_files = FileList["test/device_compatibility_test.rb"]
  t.verbose = true
end

desc "Run benchmark tests"
Rake::TestTask.new("test:benchmark") do |t|
  ENV['CANDLE_RUN_BENCHMARKS'] = 'true'
  t.deps << :compile
  t.libs << "test"
  t.test_files = FileList["test/benchmarks/**/*_test.rb"]
  t.verbose = true
end

desc "Run all tests including benchmarks"
task "test:all" => [:test, "test:benchmark"]

desc "Run tests on specific devices"
namespace :test do
  %w[cpu metal cuda].each do |device|
    desc "Run tests on #{device.upcase} only"
    task "device:#{device}" => :compile do
      ENV['CANDLE_TEST_DEVICES'] = device
      Rake::Task["test:device"].invoke
    end
  end
end

desc "Run benchmarks with device tests"
task "test:device:benchmark" => :compile do
  ENV['CANDLE_RUN_BENCHMARKS'] = 'true'
  ENV['CANDLE_TEST_VERBOSE'] = 'true'
  Rake::Task["test:device"].invoke
  Rake::Task["test:benchmark"].invoke
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
