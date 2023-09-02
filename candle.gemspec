require_relative 'lib/candle/version'

Gem::Specification.new do |spec|
  spec.name        = 'candle'
  spec.version     = Candle::VERSION
  spec.summary     = 'huggingface/candle for ruby'
  spec.description = 'POC huggingface/candle for Ruby'
  spec.files       = Dir['lib/**/*.rb', 'ext/candle/src/**/*.rs', 'ext/candle/Cargo.toml', 'Cargo.toml', 'Cargo.lock',
                         'README.md']
  spec.extensions  = ['ext/candle/Cargo.toml']
  spec.authors     = ['kojix2']
  spec.email       = ['2xijok@gmail.com']
  spec.homepage    = 'https://github.com/kojix2/ruby-candle'
  spec.license     = 'MIT'

  spec.requirements = ['Rust >= 1.61']
  spec.required_ruby_version = '>= 2.7.0'
  spec.required_rubygems_version = '>= 3.3.26'

  spec.add_development_dependency 'minitest'
  spec.add_development_dependency 'rake-compiler', '~> 1.2'
  spec.add_development_dependency 'rb_sys', '~> 0.9'
end
