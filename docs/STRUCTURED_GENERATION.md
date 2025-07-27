# Structured Generation with Red Candle

Red Candle supports structured generation, which constrains language model outputs to follow specific patterns like JSON schemas or regular expressions. This ensures outputs are valid and parseable.

## Overview

Structured generation uses finite-state machines to guide token selection during generation. At each step, only tokens that maintain valid output structure are allowed.

## Quick Start

```ruby
require 'candle'

# Load a model
llm = Candle::LLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Method 1: generate_structured (returns parsed JSON)
schema = {
  type: "object",
  properties: {
    answer: { type: "string", enum: ["yes", "no"] },
    confidence: { type: "number", minimum: 0, maximum: 1 }
  },
  required: ["answer"]
}

result = llm.generate_structured("Is Ruby a programming language?", schema: schema)
# Returns: {"answer" => "yes", "confidence" => 0.95}  # Already parsed!

# Method 2: Manual constraint (returns string)
constraint = llm.constraint_from_schema(schema)
config = Candle::GenerationConfig.balanced(constraint: constraint)
result = llm.generate("Is Ruby a programming language?", config: config)
# Returns: '{"answer": "yes", "confidence": 0.95}'  # Need to parse
```

## Creating Constraints

### From JSON Schema

```ruby
# Simple boolean
bool_constraint = llm.constraint_from_schema({ type: "boolean" })

# Object with properties
object_schema = {
  type: "object",
  properties: {
    name: { type: "string" },
    age: { type: "integer", minimum: 0 },
    email: { type: "string", format: "email" }
  },
  required: ["name", "age"]
}
object_constraint = llm.constraint_from_schema(object_schema)

# Array of items
array_schema = {
  type: "array",
  items: {
    type: "object",
    properties: {
      id: { type: "integer" },
      label: { type: "string" }
    }
  }
}
array_constraint = llm.constraint_from_schema(array_schema)

# Enum values
enum_schema = {
  type: "string",
  enum: ["small", "medium", "large"]
}
enum_constraint = llm.constraint_from_schema(enum_schema)
```

### From Regular Expressions

```ruby
# Phone number
phone_constraint = llm.constraint_from_regex('\+?1?\s?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')

# Email
email_constraint = llm.constraint_from_regex('[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

# Date (YYYY-MM-DD)
date_constraint = llm.constraint_from_regex('\d{4}-\d{2}-\d{2}')

# Time (HH:MM)
time_constraint = llm.constraint_from_regex('([01]?[0-9]|2[0-3]):[0-5][0-9]')
```

## Use Cases

### 1. Multiple Choice Questions

```ruby
schema = {
  type: "object",
  properties: {
    choice: { type: "string", enum: ["A", "B", "C", "D"] },
    reasoning: { type: "string" }
  },
  required: ["choice"]
}

# Using generate_structured (recommended)
result = llm.generate_structured(
  "What is 2+2? A) 3 B) 4 C) 5 D) 6", 
  schema: schema
)
puts result["choice"]  # "B"
puts result["reasoning"]  # "2+2 equals 4"
```

### 2. Entity Extraction

```ruby
schema = {
  type: "object",
  properties: {
    entities: {
      type: "array",
      items: {
        type: "object",
        properties: {
          text: { type: "string" },
          type: { type: "string", enum: ["person", "organization", "location"] },
          confidence: { type: "number" }
        },
        required: ["text", "type"]
      }
    }
  }
}

constraint = llm.constraint_from_schema(schema)
result = llm.generate(
  "Extract entities from: Tim Cook is the CEO of Apple in Cupertino.",
  config: Candle::GenerationConfig.balanced(constraint: constraint)
)
# Output: {"entities": [{"text": "Tim Cook", "type": "person"}, {"text": "Apple", "type": "organization"}, {"text": "Cupertino", "type": "location"}]}
```

### 3. Code Generation

```ruby
schema = {
  type: "object",
  properties: {
    language: { type: "string", enum: ["python", "ruby", "javascript"] },
    code: { type: "string" },
    explanation: { type: "string" }
  },
  required: ["language", "code"]
}

constraint = llm.constraint_from_schema(schema)
result = llm.generate(
  "Write a function to calculate factorial",
  config: Candle::GenerationConfig.balanced(constraint: constraint)
)
# Output: {"language": "python", "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)", "explanation": "Recursive implementation"}
```

### 4. Structured Forms

```ruby
# Generate structured data for forms
form_schema = {
  type: "object",
  properties: {
    first_name: { type: "string", minLength: 1 },
    last_name: { type: "string", minLength: 1 },
    age: { type: "integer", minimum: 18, maximum: 120 },
    email: { type: "string", format: "email" },
    phone: { type: "string", pattern: "\\d{3}-\\d{3}-\\d{4}" }
  },
  required: ["first_name", "last_name", "email"]
}

constraint = llm.constraint_from_schema(form_schema)
```

## Configuration Options

Constraints work with all generation configuration options:

```ruby
config = Candle::GenerationConfig.new(
  constraint: constraint,
  temperature: 0.7,      # Sampling temperature
  max_length: 200,       # Maximum tokens to generate
  top_p: 0.9,           # Nucleus sampling
  top_k: 40,            # Top-k sampling
  seed: 42              # For reproducible outputs
)
```

## Best Practices

1. **Schema Design**: Keep schemas as simple as possible while meeting requirements
2. **Max Length**: Set appropriate max_length to avoid truncated JSON
3. **Temperature**: Lower temperatures (0.1-0.5) often work better for structured output
4. **Testing**: Test constraints with various prompts to ensure robustness
5. **Error Handling**: Always validate generated output matches expected structure

## Limitations

- Complex nested schemas may be slower to generate
- Very long outputs may hit token limits before completion
- Some models may perform better than others with constraints
- Regular expressions must be compatible with the tokenizer vocabulary

## Reliability and Error Handling

Structured generation reliability depends heavily on model size and schema complexity:

### Success Rates (Approximate)
- **Large models (7B+)**: 90-95% success with most schemas
- **Small models (1-3B)**: 60-80% success with simple schemas, lower with complex ones
- **Complex nested schemas**: Lower success rates across all model sizes

### Common Failure Modes
1. **Incomplete JSON**: Model reaches max_length before closing all brackets
2. **Invalid tokens**: Model cannot find valid tokens that satisfy constraints
3. **Schema misunderstanding**: Small models may not follow complex schemas correctly

### Recommended Practices

```ruby
# 1. Use generate_structured with error handling
begin
  result = llm.generate_structured(prompt, schema: schema)
rescue JSON::ParserError => e
  # Handle parsing failure
  # Consider retry, simplification, or fallback
end

# 2. Set appropriate max_length
result = llm.generate_structured(
  prompt, 
  schema: schema,
  max_length: 200  # Ensure enough tokens for complete output
)

# 3. Use simpler schemas when possible
# Instead of deeply nested objects, consider flatter structures

# 4. Test with your specific model
# Different models have different strengths with structured generation
```

For production applications, implement appropriate error handling and consider fallback strategies based on your specific requirements.

## Implementation Details

Structured generation uses the [Outlines](https://github.com/outlines-dev/outlines) library's core functionality:

1. JSON schemas are converted to regular expressions
2. Regular expressions are compiled into finite-state machines
3. At each generation step, only valid next tokens are allowed
4. Generation stops when reaching a valid final state

This ensures 100% valid output structure while maintaining generation quality.