require_relative "lib/candle/version"

Gem::Specification.new do |spec|
  spec.name        = "red-candle"
  spec.version     = Candle::VERSION
  spec.summary     = "huggingface/candle for ruby"
  spec.description = "huggingface/candle for Ruby"
  spec.files = Dir[
    "lib/**/*.rb",
    "ext/candle/extconf.rb",
    "ext/candle/rustfmt.toml",
    "ext/candle/Cargo.toml",
    "ext/candle/**/*.rs",
    "ext/candle/**/*.toml",
    "ext/candle/**/*.md",
    "ext/candle/**/*.txt",
    "ext/candle/**/*.rb",
    "Cargo.toml",
    "Cargo.lock",
    "README.md",
    "LICENSE",
    "Rakefile",
    "Gemfile",
    "bin/*"
  ]
  spec.extensions  = ["ext/candle/extconf.rb"]
  spec.authors     = ["Christopher Petersen", "kojix2"]
  spec.email       = ["chris@petersen.io", "2xijok@gmail.com"]
  spec.homepage    = "https://github.com/assaydepot/red-candle"
  spec.license     = "MIT"

  spec.requirements = ["Rust >= 1.61"]
  spec.required_ruby_version = ">= 2.7.0"
  spec.required_rubygems_version = ">= 3.3.26"

  spec.add_dependency "rb_sys"
  
  # Development dependencies
  spec.add_development_dependency "minitest"
  spec.add_development_dependency "rake"
  spec.add_development_dependency "rake-compiler"
  spec.add_development_dependency "yard"
  spec.add_development_dependency "yard-rustdoc"
  spec.add_development_dependency "redcarpet", "~> 3.6"
  spec.add_development_dependency "informers"
  spec.add_development_dependency "simplecov"
end
