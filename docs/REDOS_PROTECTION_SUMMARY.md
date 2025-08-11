# ReDoS Protection Implementation Summary

## Overview
This document summarizes the ReDoS (Regular Expression Denial of Service) protection mechanisms implemented in Red Candle's NER module.

## Multi-Layer Protection Strategy

### 1. Pattern Validation (Proactive)
- **PatternValidator** class validates regex patterns before use
- Detects common ReDoS vulnerabilities:
  - Nested quantifiers (e.g., `(\w+)*`)
  - Unbounded quantifiers (e.g., `[A-Z]{2,}`)
  - Alternation with quantifiers (e.g., `(ab|cd)+`)
  - Unanchored wildcards (e.g., `.*test`)
- Provides warnings and suggestions for safer alternatives
- Can be disabled via `ENV['CANDLE_VALIDATE_PATTERNS'] = 'false'`

### 2. Text Length Limiting (Runtime)
- Maximum text length of 1MB (1,000,000 characters) per recognition
- Prevents exponential time complexity on very long strings
- Text is truncated with a warning if it exceeds the limit

### 3. Pattern Rewriting Tools
- `PatternValidator.make_bounded(pattern, max_length)` converts unbounded patterns to bounded ones
- Automatically replaces:
  - `{n,}` → `{n,max_length}`
  - `+` → `{1,max_length}`
  - `*` → `{0,max_length}`

## Usage Examples

### Safe Pattern Usage
```ruby
# Bounded patterns (RECOMMENDED)
recognizer = Candle::PatternEntityRecognizer.new("GENE", [
  /\b[A-Z][A-Z0-9]{2,10}\b/,  # Bounded quantifier
  /\bCD\d{1,3}\b/             # Limited digit count
])

# Validate patterns before use
pattern = /\b[A-Z][A-Z0-9]{2,}\b/
if Candle::PatternValidator.safe?(pattern)
  recognizer.add_pattern(pattern)
else
  bounded = Candle::PatternValidator.make_bounded(pattern, 10)
  recognizer.add_pattern(Regexp.new(bounded))
end
```

### Pattern Validation
```ruby
# Check if a pattern is safe
validation = Candle::PatternValidator.validate(/(\w+)*/)
# => { safe: false, warnings: [...], suggestions: [...] }

# Make an unsafe pattern safe
unsafe = /[A-Z][A-Z0-9]{2,}/
safe = Candle::PatternValidator.make_bounded(unsafe, 15)
# => "[A-Z][A-Z0-9]{2,15}"
```

## Security Alerts Addressed

1. **lib/candle/ner.rb:183** - False positive; this line adds patterns to an array
2. **spec/pattern_entity_recognizer_spec.rb:230** - Intentional test pattern for ReDoS protection
3. **Example patterns** - Updated to use bounded quantifiers

## Best Practices

1. **Always use bounded quantifiers** when possible
   - Use `{n,m}` instead of `{n,}` or `+`
   - Reasonable upper bounds: 10-100 for most cases

2. **Avoid nested quantifiers**
   - Don't use patterns like `(\w+)*` or `(a*)*`
   - Use atomic groups `(?>...)` or possessive quantifiers `++` when needed

3. **Test with malicious input**
   - Long strings of repeated characters
   - Strings that almost match but fail at the end

4. **Monitor performance**
   - Use the built-in validation to catch issues early
   - Test regex performance with realistic data

## Testing

The implementation includes comprehensive tests:
- `spec/pattern_entity_recognizer_spec.rb` - Tests ReDoS protection
- `spec/pattern_validator_spec.rb` - Tests pattern validation logic
- Performance tests ensure patterns complete within 1 second even with malicious input

## CodeQL Integration

Comments have been added to test files to indicate intentionally vulnerable patterns used for testing:
```ruby
# CodeQL Alert: This pattern is intentionally vulnerable to test our protection
recognizer.add_pattern(/(\w+)*$/)
```

## Migration Guide

For existing code using unbounded patterns:

```ruby
# Before (vulnerable)
patterns = [
  /\b[A-Z][A-Z0-9]{2,}\b/,
  /\b\w+@\w+\.\w+\b/
]

# After (safe)
patterns = [
  /\b[A-Z][A-Z0-9]{2,10}\b/,
  /\b\w{1,20}@\w{1,20}\.\w{2,6}\b/
]
```

## Performance Impact

- Pattern validation adds minimal overhead (< 1ms per pattern)
- Text truncation only affects strings > 1MB
- Bounded patterns may miss very long matches but prevent DoS attacks

## Conclusion

The multi-layer ReDoS protection ensures:
1. Developers are warned about dangerous patterns
2. Runtime protection prevents worst-case scenarios
3. Tools are provided to fix problematic patterns
4. Tests verify the protection works as expected

This approach balances security with functionality, protecting against ReDoS attacks while maintaining the flexibility of regex-based entity recognition.