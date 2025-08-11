# frozen_string_literal: true

module Candle
  # Validates regex patterns for potential ReDoS vulnerabilities
  class PatternValidator
    # Common ReDoS patterns to detect
    REDOS_PATTERNS = [
      # Nested quantifiers
      /(\*|\+|\{[^}]*\}).*(\*|\+|\{[^}]*\})/,
      # Alternation with overlapping patterns
      /\([^|)]*\|[^|)]*\)[*+]/,
      # Catastrophic backtracking patterns
      /\(\.\*\)\*/,
      /\(\.\+\)\+/,
      /\(\\[sw]\*\)\*/,
      /\(\\[sw]\+\)\+/
    ].freeze
    
    # Maximum recommended length for unbounded quantifiers
    MAX_UNBOUNDED_LENGTH = 1000
    
    class << self
      # Check if a pattern is potentially vulnerable to ReDoS
      # @param pattern [Regexp, String] The pattern to validate
      # @return [Hash] Validation result with :safe, :warnings, and :suggestions
      def validate(pattern)
        source = pattern.is_a?(Regexp) ? pattern.source : pattern.to_s
        result = {
          safe: true,
          warnings: [],
          suggestions: []
        }
        
        # Check for nested quantifiers
        if source =~ /([*+]|\{\d*,\d*\}).*([*+]|\{\d*,\d*\})/
          result[:safe] = false
          result[:warnings] << "Pattern contains nested quantifiers which may cause ReDoS"
          result[:suggestions] << suggest_fix_for_nested_quantifiers(source)
        end
        
        # Check for unbounded quantifiers on character classes
        if source =~ /\[[^\]]+\]([*+]|\{(\d+),\})/
          captures = Regexp.last_match
          if captures[1] == '+' || captures[1] == '*' || captures[2].nil? || captures[2].to_i > MAX_UNBOUNDED_LENGTH
            result[:warnings] << "Pattern contains unbounded quantifier on character class"
            result[:suggestions] << "Consider adding an upper bound, e.g., {2,100} instead of {2,} or +"
          end
        end
        
        # Also check for unbounded {n,} anywhere in the pattern
        if source =~ /\{(\d+),\}/
          result[:warnings] << "Pattern contains unbounded quantifier {#{$1},}"
          result[:suggestions] << "Consider adding an upper bound, e.g., {#{$1},100}"
        end
        
        # Check for alternation with quantifiers
        if source =~ /\([^)]*\|[^)]*\)[*+]/
          result[:warnings] << "Pattern contains alternation with quantifier which may cause backtracking"
          result[:suggestions] << "Consider using atomic groups (?>) or possessive quantifiers (++)"
        end
        
        # Check for leading/trailing .* without anchors
        if source =~ /^\.\*/ || source =~ /\.\*$/
          unless source =~ /^\^/ || source =~ /\$$/
            result[:warnings] << "Pattern contains unanchored .* which may match too broadly"
            result[:suggestions] << "Consider adding anchors (^ and $) or limiting the scope"
          end
        end
        
        result
      end
      
      # Check if a pattern is safe for production use
      # @param pattern [Regexp, String] The pattern to check
      # @return [Boolean] true if pattern is safe
      def safe?(pattern)
        validate(pattern)[:safe]
      end
      
      # Get a bounded version of a pattern
      # @param pattern [Regexp, String] The pattern to bound
      # @param max_length [Integer] Maximum repetition count
      # @return [String] Bounded version of the pattern
      def make_bounded(pattern, max_length = 100)
        source = pattern.is_a?(Regexp) ? pattern.source : pattern.to_s
        
        # Replace unbounded quantifiers with bounded ones
        bounded = source.gsub(/\{(\d+),\}/) do |match|
          min = Regexp.last_match(1)
          "{#{min},#{max_length}}"
        end
        
        # Replace + with {1,max_length}
        bounded = bounded.gsub(/([^\\])\+/) do |match|
          "#{Regexp.last_match(1)}{1,#{max_length}}"
        end
        
        # Replace * with {0,max_length}
        bounded = bounded.gsub(/([^\\])\*/) do |match|
          "#{Regexp.last_match(1)}{0,#{max_length}}"
        end
        
        bounded
      end
      
      private
      
      def suggest_fix_for_nested_quantifiers(source)
        # Try to identify the pattern and suggest a fix
        if source =~ /\[A-Z\]\[A-Z0-9\]\{2,\}/
          "Consider using bounded quantifiers: /\\b[A-Z][A-Z0-9]{2,10}\\b/"
        elsif source =~ /\(\w\+\)\*/
          "Consider using atomic groups: /(?>\\w+)*/ or limiting repetitions"
        else
          "Consider using atomic groups (?>) or possessive quantifiers (++) to prevent backtracking"
        end
      end
    end
  end
end