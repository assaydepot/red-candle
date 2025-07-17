# frozen_string_literal: true

module Candle
  # Tokenizer class for text tokenization
  #
  # This class provides methods to encode text into tokens and decode tokens back to text.
  # It supports both single text and batch processing, with options for special tokens,
  # padding, and truncation.
  #
  # @example Create a tokenizer from a pretrained model
  #   tokenizer = Candle::Tokenizer.from_pretrained("bert-base-uncased")
  #
  # @example Encode and decode text
  #   tokens = tokenizer.encode("Hello, world!")
  #   text = tokenizer.decode(tokens)
  #
  # @example Batch encoding
  #   texts = ["Hello", "World", "Test"]
  #   batch_tokens = tokenizer.encode_batch(texts)
  #
  # @example Configure padding and truncation
  #   padded_tokenizer = tokenizer.with_padding(length: 128)
  #   truncated_tokenizer = tokenizer.with_truncation(512)
  class Tokenizer
    # These methods are implemented in Rust
    # - from_file(path) - Load tokenizer from a JSON file
    # - from_pretrained(model_id) - Load tokenizer from HuggingFace
    # - encode(text, add_special_tokens = true) - Encode text to token IDs
    # - encode_to_tokens(text, add_special_tokens = true) - Encode text to token strings
    # - encode_with_tokens(text, add_special_tokens = true) - Get both IDs and tokens
    # - encode_batch(texts, add_special_tokens = true) - Encode multiple texts to IDs
    # - encode_batch_to_tokens(texts, add_special_tokens = true) - Encode multiple texts to tokens
    # - decode(token_ids, skip_special_tokens = true) - Decode token IDs to text
    # - id_to_token(token_id) - Get token string for a token ID
    # - get_vocab(with_added_tokens = true) - Get vocabulary as hash
    # - vocab_size(with_added_tokens = true) - Get vocabulary size
    # - with_padding(options) - Create tokenizer with padding enabled
    # - with_truncation(max_length) - Create tokenizer with truncation enabled
    # - get_special_tokens - Get special tokens info
    # - inspect - String representation
    # - to_s - String representation

    # The native methods accept positional arguments, but we provide keyword argument interfaces
    # for better Ruby ergonomics. We need to call the native methods with positional args.
    
    alias_method :_native_encode, :encode
    alias_method :_native_encode_to_tokens, :encode_to_tokens
    alias_method :_native_encode_with_tokens, :encode_with_tokens
    alias_method :_native_encode_batch, :encode_batch
    alias_method :_native_encode_batch_to_tokens, :encode_batch_to_tokens
    alias_method :_native_decode, :decode
    alias_method :_native_get_vocab, :get_vocab
    alias_method :_native_vocab_size, :vocab_size
    alias_method :_native_with_padding, :with_padding

    # Encode text with convenient keyword arguments
    #
    # @param text [String] The text to encode
    # @param add_special_tokens [Boolean] Whether to add special tokens (default: true)
    # @return [Array<Integer>] Token IDs
    def encode(text, add_special_tokens: true)
      _native_encode(text, add_special_tokens)
    end

    # Encode text into token strings (words/subwords)
    #
    # @param text [String] The text to encode
    # @param add_special_tokens [Boolean] Whether to add special tokens (default: true)
    # @return [Array<String>] Token strings
    def encode_to_tokens(text, add_special_tokens: true)
      _native_encode_to_tokens(text, add_special_tokens)
    end

    # Encode text and return both IDs and token strings
    #
    # @param text [String] The text to encode
    # @param add_special_tokens [Boolean] Whether to add special tokens (default: true)
    # @return [Hash] Hash with :ids and :tokens arrays
    def encode_with_tokens(text, add_special_tokens: true)
      _native_encode_with_tokens(text, add_special_tokens)
    end

    # Encode multiple texts with convenient keyword arguments
    #
    # @param texts [Array<String>] The texts to encode
    # @param add_special_tokens [Boolean] Whether to add special tokens (default: true)
    # @return [Array<Array<Integer>>] Arrays of token IDs
    def encode_batch(texts, add_special_tokens: true)
      _native_encode_batch(texts, add_special_tokens)
    end

    # Encode multiple texts into token strings
    #
    # @param texts [Array<String>] The texts to encode
    # @param add_special_tokens [Boolean] Whether to add special tokens (default: true)
    # @return [Array<Array<String>>] Arrays of token strings
    def encode_batch_to_tokens(texts, add_special_tokens: true)
      _native_encode_batch_to_tokens(texts, add_special_tokens)
    end

    # Decode token IDs with convenient keyword arguments
    #
    # @param token_ids [Array<Integer>] The token IDs to decode
    # @param skip_special_tokens [Boolean] Whether to skip special tokens (default: true)
    # @return [String] Decoded text
    def decode(token_ids, skip_special_tokens: true)
      _native_decode(token_ids, skip_special_tokens)
    end

    # Get vocabulary with convenient keyword arguments
    #
    # @param with_added_tokens [Boolean] Include added tokens (default: true)
    # @return [Hash<String, Integer>] Token to ID mapping
    def get_vocab(with_added_tokens: true)
      _native_get_vocab(with_added_tokens)
    end

    # Get vocabulary size with convenient keyword arguments
    #
    # @param with_added_tokens [Boolean] Include added tokens (default: true)
    # @return [Integer] Vocabulary size
    def vocab_size(with_added_tokens: true)
      _native_vocab_size(with_added_tokens)
    end

    # Create a new tokenizer with padding configuration
    #
    # @param options [Hash] Padding options
    # @option options [Integer] :length Fixed length padding
    # @option options [Boolean] :max_length Use batch longest padding
    # @option options [String] :direction Padding direction ("left" or "right")
    # @option options [Integer] :pad_id Padding token ID
    # @option options [String] :pad_token Padding token string
    # @return [Tokenizer] New tokenizer instance with padding enabled
    def with_padding(**options)
      _native_with_padding(options)
    end
  end
end