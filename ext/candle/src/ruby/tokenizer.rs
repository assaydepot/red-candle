use magnus::{class, function, method, prelude::*, Error, Module, RArray, RHash, RModule, TryConvert};
use crate::tokenizer::{TokenizerWrapper as InnerTokenizer, loader::TokenizerLoader};
use crate::ruby::Result;

#[derive(Clone, Debug)]
#[magnus::wrap(class = "Candle::Tokenizer", free_immediately, size)]
pub struct Tokenizer(pub InnerTokenizer);

impl Tokenizer {
    /// Create a new tokenizer from a file path
    pub fn from_file(path: String) -> Result<Self> {
        let tokenizer = TokenizerLoader::from_file(&path)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        Ok(Self(InnerTokenizer::new(tokenizer)))
    }

    /// Create a new tokenizer from HuggingFace model ID
    pub fn from_pretrained(model_id: String) -> Result<Self> {
        // Use tokio runtime for async operations
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to create runtime: {}", e)))?;
        
        let tokenizer = rt.block_on(async {
            TokenizerLoader::from_hf_hub(&model_id, None).await
        })
        .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        Ok(Self(InnerTokenizer::new(tokenizer)))
    }

    /// Encode text into token IDs
    pub fn encode(&self, text: String, add_special_tokens: Option<bool>) -> Result<RArray> {
        let add_special = add_special_tokens.unwrap_or(true);
        let token_ids = self.0.encode(&text, add_special)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        Ok(RArray::from_vec(token_ids.into_iter().map(|id| id as i64).collect()))
    }

    /// Encode multiple texts in batch
    pub fn encode_batch(&self, texts: RArray, add_special_tokens: Option<bool>) -> Result<RArray> {
        let texts: Vec<String> = texts.to_vec()?;
        let add_special = add_special_tokens.unwrap_or(true);
        
        let token_ids_batch = self.0.encode_batch(texts, add_special)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        let result = RArray::new();
        for token_ids in token_ids_batch {
            result.push(RArray::from_vec(token_ids.into_iter().map(|id| id as i64).collect()))?;
        }
        
        Ok(result)
    }

    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: RArray, skip_special_tokens: Option<bool>) -> Result<String> {
        let token_ids: Vec<i64> = token_ids.to_vec()?;
        let token_ids: Vec<u32> = token_ids.into_iter()
            .map(|id| id as u32)
            .collect();
        let skip_special = skip_special_tokens.unwrap_or(true);
        
        self.0.decode(&token_ids, skip_special)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))
    }

    /// Get the string representation of a single token ID
    pub fn id_to_token(&self, token_id: i64) -> Result<String> {
        self.0.token_to_piece(token_id as u32)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))
    }

    /// Get the vocabulary as a hash of token string to ID
    pub fn get_vocab(&self, with_added_tokens: Option<bool>) -> Result<RHash> {
        let with_added = with_added_tokens.unwrap_or(true);
        let vocab = self.0.inner().get_vocab(with_added);
        
        let hash = RHash::new();
        for (token, id) in vocab {
            hash.aset(token, id as i64)?;
        }
        
        Ok(hash)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self, with_added_tokens: Option<bool>) -> usize {
        let with_added = with_added_tokens.unwrap_or(true);
        self.0.inner().get_vocab_size(with_added)
    }

    /// Enable padding - returns a new tokenizer with padding enabled
    pub fn with_padding(&self, kwargs: RHash) -> Result<Self> {
        use tokenizers::{PaddingParams, PaddingStrategy, PaddingDirection};
        
        let mut params = PaddingParams::default();
        
        // Extract parameters from kwargs
        if let Some(length) = kwargs.get(magnus::Symbol::new("length")) {
            if let Ok(len) = usize::try_convert(length) {
                params.strategy = PaddingStrategy::Fixed(len);
            }
        }
        
        if let Some(max_length) = kwargs.get(magnus::Symbol::new("max_length")) {
            if let Ok(_) = usize::try_convert(max_length) {
                params.strategy = PaddingStrategy::BatchLongest;
            }
        }
        
        if let Some(direction) = kwargs.get(magnus::Symbol::new("direction")) {
            if let Ok(dir) = String::try_convert(direction) {
                params.direction = match dir.as_str() {
                    "right" => PaddingDirection::Right,
                    "left" => PaddingDirection::Left,
                    _ => PaddingDirection::Right,
                };
            }
        }
        
        if let Some(pad_id) = kwargs.get(magnus::Symbol::new("pad_id")) {
            if let Ok(id) = u32::try_convert(pad_id) {
                params.pad_id = id;
            }
        }
        
        if let Some(pad_token) = kwargs.get(magnus::Symbol::new("pad_token")) {
            if let Ok(token) = String::try_convert(pad_token) {
                params.pad_token = token;
            }
        }
        
        let mut new_tokenizer = self.0.clone();
        let _ = new_tokenizer.inner_mut().with_padding(Some(params));
        Ok(Self(new_tokenizer))
    }

    /// Enable truncation - returns a new tokenizer with truncation enabled
    pub fn with_truncation(&self, max_length: usize) -> Result<Self> {
        use tokenizers::{TruncationParams, TruncationStrategy, TruncationDirection};
        
        let params = TruncationParams {
            max_length,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
            direction: TruncationDirection::Right,
        };
        
        let mut new_tokenizer = self.0.clone();
        let _ = new_tokenizer.inner_mut().with_truncation(Some(params));
        Ok(Self(new_tokenizer))
    }

    /// Get special tokens information
    pub fn get_special_tokens(&self) -> Result<RHash> {
        let hash = RHash::new();
        
        // Common special tokens
        let special_tokens = vec![
            ("[CLS]", "cls_token"),
            ("[SEP]", "sep_token"),
            ("[PAD]", "pad_token"),
            ("[UNK]", "unk_token"),
            ("[MASK]", "mask_token"),
            ("<s>", "bos_token"),
            ("</s>", "eos_token"),
        ];
        
        let vocab = self.0.inner().get_vocab(true);
        
        for (token, name) in special_tokens {
            if let Some(id) = vocab.get(token) {
                hash.aset(name, *id as i64)?;
            }
        }
        
        Ok(hash)
    }

    /// String representation
    pub fn inspect(&self) -> String {
        format!("#<Candle::Tokenizer vocab_size={}>", self.vocab_size(Some(true)))
    }
}

pub fn init(rb_candle: RModule) -> Result<()> {
    let tokenizer_class = rb_candle.define_class("Tokenizer", class::object())?;
    
    // Class methods
    tokenizer_class.define_singleton_method("from_file", function!(Tokenizer::from_file, 1))?;
    tokenizer_class.define_singleton_method("from_pretrained", function!(Tokenizer::from_pretrained, 1))?;
    
    // Instance methods
    tokenizer_class.define_method("encode", method!(Tokenizer::encode, 2))?;
    tokenizer_class.define_method("encode_batch", method!(Tokenizer::encode_batch, 2))?;
    tokenizer_class.define_method("decode", method!(Tokenizer::decode, 2))?;
    tokenizer_class.define_method("id_to_token", method!(Tokenizer::id_to_token, 1))?;
    tokenizer_class.define_method("get_vocab", method!(Tokenizer::get_vocab, 1))?;
    tokenizer_class.define_method("vocab_size", method!(Tokenizer::vocab_size, 1))?;
    tokenizer_class.define_method("with_padding", method!(Tokenizer::with_padding, 1))?;
    tokenizer_class.define_method("with_truncation", method!(Tokenizer::with_truncation, 1))?;
    tokenizer_class.define_method("get_special_tokens", method!(Tokenizer::get_special_tokens, 0))?;
    tokenizer_class.define_method("inspect", method!(Tokenizer::inspect, 0))?;
    tokenizer_class.define_method("to_s", method!(Tokenizer::inspect, 0))?;
    
    Ok(())
}