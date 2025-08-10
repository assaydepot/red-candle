# ReDoS Prevention in Pattern Recognition

## Background

Regular Expression Denial of Service (ReDoS) occurs when certain regex patterns cause catastrophic backtracking, leading to exponential time complexity. Ruby versions before 3.2 are particularly vulnerable.

## Problematic Pattern Examples

### ❌ BAD: Nested Quantifiers
```ruby
/\b[A-Z][A-Z0-9]{2,}\b/  # Can backtrack catastrophically
```

This pattern is problematic because:
- `[A-Z0-9]{2,}` creates ambiguity about where to stop matching
- When there's no word boundary at the end, the engine tries every possible combination
- Input like "AAAA0000" followed by a non-word character causes exponential backtracking

### ✅ GOOD: Atomic Groups or Possessive Quantifiers
```ruby
# Option 1: Use possessive quantifier (Ruby 1.9+)
/\b[A-Z][A-Z0-9]{2,}+\b/  # The + after {2,} makes it possessive

# Option 2: Use atomic grouping
/\b[A-Z](?>[A-Z0-9]{2,})\b/  # (?>...) prevents backtracking

# Option 3: Be more specific
/\b[A-Z][A-Z0-9]{2,10}\b/  # Limit the maximum length

# Option 4: Simplify the pattern
/\b[A-Z][A-Z0-9]+\b/  # Often sufficient and safer
```

## Safe Pattern Guidelines

### 1. Avoid Nested Quantifiers
```ruby
# BAD
/(\d+)*$/       # Nested quantifiers
/(a*)*b/        # Extremely dangerous

# GOOD
/\d+$/          # Single quantifier
/a*b/           # Simplified
```

### 2. Use Possessive Quantifiers When Appropriate
```ruby
# BAD
/\d+\d+/        # Ambiguous where first ends

# GOOD  
/\d++\d+/       # First part won't backtrack
```

### 3. Limit Repetition Ranges
```ruby
# BAD
/\w{2,}/        # Unbounded

# GOOD
/\w{2,100}/     # Bounded maximum
```

### 4. Use Atomic Groups for Complex Patterns
```ruby
# BAD
/(\w+\s*)+$/    # Can backtrack exponentially

# GOOD
/(?>\w+\s*)+$/  # Atomic group prevents backtracking
```

## Testing for ReDoS Vulnerabilities

```ruby
# Test pattern with increasingly long inputs
def test_redos(pattern, base_string, suffix)
  [10, 100, 1000, 10000].each do |n|
    input = base_string * n + suffix
    start = Time.now
    input.match(pattern)
    elapsed = Time.now - start
    
    if elapsed > 1.0
      puts "WARNING: Pattern may be vulnerable to ReDoS"
      puts "Time for #{n} repetitions: #{elapsed}s"
      break
    end
  end
end

# Example test
test_redos(/^(\d+)*$/, "1", "X")  # Will show exponential growth
```

## Mitigations in Red Candle

The `PatternEntityRecognizer` class includes several protections:

1. **Text Length Limiting**: Texts over 1MB are truncated
2. **Pattern Validation**: Warns about patterns with nested quantifiers
3. **Ruby 3.2+**: Benefits from built-in ReDoS protection

## Recommended Patterns for Common Use Cases

### Gene/Protein Names
```ruby
# Instead of: /\b[A-Z][A-Z0-9]{2,}\b/
# Use: 
/\b[A-Z][A-Z0-9]{2,10}\b/     # Limit length
# Or:
/\b[A-Z](?>[A-Z0-9]{2,})\b/   # Atomic group
```

### Email Addresses
```ruby
# Safe email pattern (simplified but ReDoS-resistant)
/\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,6}\b/
```

### URLs
```ruby
# Safe URL pattern
/https?:\/\/[^\s<>"{}|\\^`\[\]]+/
```

### Phone Numbers
```ruby
# Safe phone pattern
/\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/
```

## References

- [OWASP ReDoS Guide](https://owasp.org/www-community/attacks/Regular_expression_Denial_of_Service_-_ReDoS)
- [Ruby 3.2 ReDoS Protection](https://bugs.ruby-lang.org/issues/17931)
- [Regular Expressions Info](https://www.regular-expressions.info/catastrophic.html)