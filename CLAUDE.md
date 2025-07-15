# Red Candle Development Guide

This guide captures the coding conventions and patterns used in the red-candle Ruby gem.

## Project Overview

Red Candle is a Ruby gem that uses the Magnus Rust crate to embed Rust code in Ruby, providing access to the Candle ML library from Hugging Face. It enables Ruby developers to use embedding models, rerankers, and LLMs.

## Architecture Overview

```mermaid
graph TB
    subgraph "Ruby Layer"
        A[Ruby Application]
        B[Candle Module]
        C[Model Classes]
        D[Device Utils]
    end
    
    subgraph "Native Extension (Rust)"
        E[Magnus Bindings]
        F[Candle Core]
        G[Model Implementations]
        H[Hardware Abstraction]
    end
    
    subgraph "Hardware"
        I[CPU]
        J[Metal/GPU]
        K[CUDA/GPU]
    end
    
    A --> B
    B --> C
    C --> E
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    H --> J
    H --> K
```

## Module Structure

### Ruby Module Structure

```mermaid
graph LR
    subgraph "Candle Module"
        A[Candle::Tensor]
        B[Candle::Device]
        C[Candle::DType]
        D[Candle::EmbeddingModel]
        E[Candle::LLM]
        F[Candle::Reranker]
        G[Candle::GenerationConfig]
    end
    
    D --> A
    D --> B
    E --> A
    E --> B
    E --> G
    F --> A
    F --> B
    A --> C
    A --> B
```

### Rust Class Structure

```mermaid
graph TB
    subgraph "LLM Module"
        A[ModelType enum]
        B[Mistral]
        C[Llama]
        D[Gemma]
        E[QuantizedGGUF]
        
        A --> B
        A --> C
        A --> D
        A --> E
    end
    
    subgraph "Embedding Module"
        EM[EmbeddingModel]
        EMI[EmbeddingModelInner]
        EMT[EmbeddingModelType]
        EMV[EmbeddingModelVariant]
        JB[JinaBert]
        SB[StandardBert]
        DB[DistilBert]
        ML[MiniLM]
        
        EM --> EMI
        EMI --> EMV
        EMT --> JB
        EMT --> SB
        EMT --> DB
        EMT --> ML
        EMV --> JB
        EMV --> SB
        EMV --> DB
        EMV --> ML
    end
    
    subgraph "Reranker Module"
        R[Reranker]
        RM[BertModel]
        RP[Pooler Linear]
        RC[Classifier Linear]
        
        R --> RM
        R --> RP
        R --> RC
    end
    
    subgraph "Traits"
        F[TextGenerator]
        G[generate]
        H[generate_stream]
        I[clear_cache]
        
        F --> G
        F --> H
        F --> I
    end
    
    subgraph "GGUF Internals"
        J[QuantizedGGUF]
        K[ModelType::Llama]
        L[ModelType::Gemma]
        M[Architecture Detection]
        N[Tokenizer Download]
        
        J --> M
        J --> N
        J --> K
        J --> L
    end
    
    subgraph "Support Types"
        O[GenerationConfig]
        P[TokenizerWrapper]
        Q[TextGeneration]
        T[Tokenizer]
        DEV[Device]
        
        O --> Q
        P --> Q
    end
    
    B -.-> F
    C -.-> F
    D -.-> F
    E -.-> F
    
    EM -.-> T
    EM -.-> DEV
    R -.-> T
    R -.-> DEV
    A -.-> DEV
```

## Directory Structure

```
red-candle/
├── lib/              # Ruby source files
│   └── candle/       # Main module namespace
├── ext/              # Native extensions
│   └── candle/       # Rust extension
│       └── src/      # Rust source files
├── test/             # Test suite
├── examples/         # Usage examples
├── docs/             # Additional documentation
└── bin/              # Executables
```

## Ruby Conventions

### Module and Class Structure

- Single module namespace: `Candle`
- Clear class responsibilities:
  - `Tensor` - Core tensor operations
  - `LLM` - Language model functionality
  - `EmbeddingModel` - Text embeddings
  - `Reranker` - Document reranking

### Ruby Style

```ruby
module Candle
  class ClassName
    # Constants first
    CONSTANT_NAME = value
    
    # Class methods
    class << self
      def class_method
      end
    end
    
    # Public instance methods
    def public_method
    end
    
    private
    
    def private_method
    end
  end
end
```

### Naming Conventions

- Classes: `PascalCase`
- Methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Files: `snake_case.rb`
- Use modern hash syntax with symbols
- Use keyword arguments for optional parameters

## Rust Conventions

### Rust Configuration (rustfmt.toml)

- Indentation: 4 spaces
- Line width: 100 characters max
- Edition: Rust 2021

### Rust Patterns

```rust
#[magnus::wrap(class = "Candle::ClassName", free_immediately, size)]
pub struct ClassName(pub InternalType);

impl ClassName {
    pub fn new(params: Type) -> Result<Self> {
        // Implementation with proper error wrapping
    }
}
```

- Error handling: Uses `Result<T, magnus::Error>` type
- Magnus integration: Wrapper structs with `#[magnus::wrap]`
- Feature flags: Conditional compilation for CUDA/Metal support

## Testing

### Framework: Minitest

```ruby
require_relative "test_helper"

class ClassNameTest < Minitest::Test
  def test_feature_description
    # Test implementation
  end
end
```

### Test Commands

```bash
rake              # Run default tests
rake test         # Run unit tests
rake test:device  # Run device compatibility tests
rake test:benchmark # Run benchmarks
rake test:all     # Run all tests
rake test:device:cpu/metal/cuda # Test specific device
```

## Development Workflow

```mermaid
graph LR
    A[bundle install] --> B[rake compile]
    B --> C[rake test]
    C --> D{Tests Pass?}
    D -->|No| E[Fix Issues]
    E --> B
    D -->|Yes| F[Development Complete]
```

## Build Commands

- **Compile**: `rake compile`
- **Test**: `rake test`
- **Lint**: Check if lint command exists in project
- **Type check**: Check if type checking is configured

## Key Patterns

1. **Error Handling**: Consistent use of Result types with proper error wrapping
2. **Device Abstraction**: Clean abstraction for CPU/Metal/CUDA devices
3. **Feature Detection**: Automatic detection of available hardware acceleration
4. **Modular Design**: Clear separation between Ruby interface and Rust implementation
5. **Testing Strategy**: Comprehensive testing with device-specific considerations
6. **Tokenizer Registry**: Automatic tokenizer detection and fallback system for GGUF models
7. **Chat Templates**: Model-specific chat template application for proper formatting

## Data Flow

```mermaid
sequenceDiagram
    participant Ruby
    participant Magnus
    participant Rust
    participant Candle
    participant Hardware
    
    Ruby->>Magnus: Call method
    Magnus->>Rust: Convert Ruby objects
    Rust->>Candle: Execute ML operation
    Candle->>Hardware: Compute on device
    Hardware-->>Candle: Return results
    Candle-->>Rust: Tensor results
    Rust-->>Magnus: Wrap in Ruby objects
    Magnus-->>Ruby: Return Ruby objects
```

## Documentation Style

- YARD documentation for Ruby code
- Rust documentation integrated
- Clear examples in code
- Markdown files for specific topics (UPPER_CASE.md for important docs)

## Important Notes

- Do not modify the 'ignored' directory
- Use frozen string literals in Ruby files
- Follow existing patterns when adding new functionality
- Ensure tests pass on all supported devices before committing
- Keep error messages informative and actionable
- Avoid adding comments unless explicitly requested

## Tokenizer Registry System

The LLM module now includes an intelligent tokenizer registry for GGUF models:

```ruby
# Register custom tokenizer mappings
Candle::LLM.register_tokenizer("model-pattern", "tokenizer-id")
Candle::LLM.register_tokenizer(/regex-pattern/, "tokenizer-id")

# Automatic detection for common models
# TheBloke/Mistral-7B-Instruct-v0.2-GGUF -> mistralai/Mistral-7B-Instruct-v0.2
```

When loading GGUF models without embedded tokenizers:
1. First attempts to load without tokenizer
2. If missing, auto-detects appropriate tokenizer source
3. Falls back with clear error messages and solutions

## GGUF Model Loading

### Syntax Options

```ruby
# Basic GGUF loading
llm = Candle::LLM.from_pretrained("TheBloke/Model-GGUF", 
                                  gguf_file: "model.Q4_K_M.gguf")

# With explicit tokenizer
llm = Candle::LLM.from_pretrained("TheBloke/Model-GGUF",
                                  gguf_file: "model.Q4_K_M.gguf",
                                  tokenizer: "original/model-source")

# Advanced syntax (used internally)
# model_id@gguf_file@@tokenizer_source
```

### Architecture Detection

The unified GGUF loader automatically detects:
- Model architecture from GGUF metadata
- Appropriate tokenizer based on model patterns
- Correct chat template for the model type

## Chat Interface

New chat methods provide conversation-style interactions:

```ruby
messages = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "What is Ruby?" }
]

# Synchronous chat
response = llm.chat(messages)

# Streaming chat
llm.chat_stream(messages) do |token|
  print token
end
```

Model-specific templates are automatically applied:
- Llama 2: `<s>[INST] <<SYS>>...</SYS>> user [/INST] assistant </s>`
- Llama 3: `<|begin_of_text|><|start_header_id|>...<|end_header_id|>`
- Mistral: `[INST] user [/INST] assistant</s>`
- Gemma: `<start_of_turn>user...model<end_of_turn>`

## Generation Configuration

### Presets

```ruby
# Temperature = 0, fixed seed
config = Candle::GenerationConfig.deterministic

# Higher temperature, more randomness
config = Candle::GenerationConfig.creative

# Balanced settings
config = Candle::GenerationConfig.balanced

# Chain modifications
config = Candle::GenerationConfig.balanced.with(max_length: 1000)
```

### Debug Mode

```ruby
# Shows token IDs and pieces during generation
config = Candle::GenerationConfig.new(debug_tokens: true)
llm.generate("Hello", config: config)
# Output: [128000:Hello][1299: world][128001:<|eot_id|>]
```

## Error Handling Improvements

Enhanced error messages now provide:
- Specific failure reasons
- Multiple solution suggestions
- Network connectivity hints
- Authentication guidance (HF_TOKEN)
- Alternative model/tokenizer recommendations

Example error output:
```
Failed to load GGUF model with auto-detected tokenizer.

Original error: No tokenizer found in GGUF repository
Tokenizer error: Failed to find tokenizer in specified source

Possible solutions:
1. Specify a tokenizer explicitly
2. Check your network connection
3. Set HF_TOKEN environment variable
4. Try a different model source
```