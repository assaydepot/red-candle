# ext
begin
  require "candle/#{RUBY_VERSION.to_f}/candle"
rescue LoadError
  require "candle/candle"
end