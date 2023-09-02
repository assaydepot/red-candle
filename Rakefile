# frozen_string_literal: true

require 'rake/testtask'
require 'rake/extensiontask'

task default: :test

spec = Bundler.load_gemspec('candle.gemspec')
spec.requirements.clear
spec.required_ruby_version = nil
spec.required_rubygems_version = nil
spec.extensions.clear
spec.files -= Dir['ext/**/*']

Rake::ExtensionTask.new('candle', spec) do |c|
  c.lib_dir = 'lib/candle'
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

task :dev do
  ENV['RB_SYS_CARGO_PROFILE'] = 'dev'
end

Rake::TestTask.new do |t|
  t.deps << :compile
  t.test_files = FileList[File.expand_path('test/*_test.rb', __dir__)]
end

task bench: :compile do
  ruby 'test/bench.rb'
end
