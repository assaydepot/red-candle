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
    
    /// Encode text into token strings (words/subwords)
    pub fn encode_to_tokens(&self, text: String, add_special_tokens: Option<bool>) -> Result<RArray> {
        let add_special = add_special_tokens.unwrap_or(true);
        let token_ids = self.0.encode(&text, add_special)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        let mut tokens = Vec::new();
        for id in token_ids {
            let token = self.0.token_to_piece(id)
                .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
            tokens.push(token);
        }
        
        Ok(RArray::from_vec(tokens))
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
    
    /// Encode multiple texts in batch, returning token strings
    pub fn encode_batch_to_tokens(&self, texts: RArray, add_special_tokens: Option<bool>) -> Result<RArray> {
        let texts: Vec<String> = texts.to_vec()?;
        let add_special = add_special_tokens.unwrap_or(true);
        
        let token_ids_batch = self.0.encode_batch(texts, add_special)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        let result = RArray::new();
        for token_ids in token_ids_batch {
            let mut tokens = Vec::new();
            for id in token_ids {
                let token = self.0.token_to_piece(id)
                    .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
                tokens.push(token);
            }
            result.push(RArray::from_vec(tokens))?;
        }
        
        Ok(result)
    }

    /// Encode text and return both token IDs and token strings
    pub fn encode_with_tokens(&self, text: String, add_special_tokens: Option<bool>) -> Result<RHash> {
        let add_special = add_special_tokens.unwrap_or(true);
        let token_ids = self.0.encode(&text, add_special)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        let mut tokens = Vec::new();
        for &id in &token_ids {
            let token = self.0.token_to_piece(id)
                .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
            tokens.push(token);
        }
        
        let hash = RHash::new();
        hash.aset("ids", RArray::from_vec(token_ids.into_iter().map(|id| id as i64).collect()))?;
        hash.aset("tokens", RArray::from_vec(tokens))?;
        
        Ok(hash)
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

    /// Get tokenizer options as a hash
    pub fn options(&self) -> Result<RHash> {
        let hash = RHash::new();
        
        // Get vocab size
        hash.aset("vocab_size", self.vocab_size(Some(true)))?;
        hash.aset("vocab_size_base", self.vocab_size(Some(false)))?;
        
        // Get special tokens info
        let special_tokens = self.get_special_tokens()?;
        hash.aset("special_tokens", special_tokens)?;
        
        // Get padding/truncation info if available
        let inner_tokenizer = self.0.inner();
        
        // Check if padding is enabled
        if let Some(_padding) = inner_tokenizer.get_padding() {
            let padding_info = RHash::new();
            padding_info.aset("enabled", true)?;
            // Note: We can't easily extract all padding params from the tokenizers library
            // but we can indicate it's enabled
            hash.aset("padding", padding_info)?;
        }
        
        // Check if truncation is enabled  
        if let Some(truncation) = inner_tokenizer.get_truncation() {
            let truncation_info = RHash::new();
            truncation_info.aset("enabled", true)?;
            truncation_info.aset("max_length", truncation.max_length)?;
            hash.aset("truncation", truncation_info)?;
        }
        
        Ok(hash)
    }

    /// String representation
    pub fn inspect(&self) -> String {
        let vocab_size = self.vocab_size(Some(true));
        let special_tokens = self.get_special_tokens()
            .ok()
            .map(|h| h.len())
            .unwrap_or(0);
        
        let mut parts = vec![format!("#<Candle::Tokenizer vocab_size={}", vocab_size)];
        
        if special_tokens > 0 {
            parts.push(format!("special_tokens={}", special_tokens));
        }
        
        // Check for padding/truncation
        let inner_tokenizer = self.0.inner();
        if inner_tokenizer.get_padding().is_some() {
            parts.push("padding=enabled".to_string());
        }
        if let Some(truncation) = inner_tokenizer.get_truncation() {
            parts.push(format!("truncation={}", truncation.max_length));
        }
        
        parts.join(" ") + ">"
    }
}

pub fn init(rb_candle: RModule) -> Result<()> {
    let tokenizer_class = rb_candle.define_class("Tokenizer", class::object())?;
    
    // Class methods
    tokenizer_class.define_singleton_method("from_file", function!(Tokenizer::from_file, 1))?;
    tokenizer_class.define_singleton_method("from_pretrained", function!(Tokenizer::from_pretrained, 1))?;
    
    // Instance methods
    tokenizer_class.define_method("encode", method!(Tokenizer::encode, 2))?;
    tokenizer_class.define_method("encode_to_tokens", method!(Tokenizer::encode_to_tokens, 2))?;
    tokenizer_class.define_method("encode_with_tokens", method!(Tokenizer::encode_with_tokens, 2))?;
    tokenizer_class.define_method("encode_batch", method!(Tokenizer::encode_batch, 2))?;
    tokenizer_class.define_method("encode_batch_to_tokens", method!(Tokenizer::encode_batch_to_tokens, 2))?;
    tokenizer_class.define_method("decode", method!(Tokenizer::decode, 2))?;
    tokenizer_class.define_method("id_to_token", method!(Tokenizer::id_to_token, 1))?;
    tokenizer_class.define_method("get_vocab", method!(Tokenizer::get_vocab, 1))?;
    tokenizer_class.define_method("vocab_size", method!(Tokenizer::vocab_size, 1))?;
    tokenizer_class.define_method("with_padding", method!(Tokenizer::with_padding, 1))?;
    tokenizer_class.define_method("with_truncation", method!(Tokenizer::with_truncation, 1))?;
    tokenizer_class.define_method("get_special_tokens", method!(Tokenizer::get_special_tokens, 0))?;
    tokenizer_class.define_method("options", method!(Tokenizer::options, 0))?;
    tokenizer_class.define_method("inspect", method!(Tokenizer::inspect, 0))?;
    tokenizer_class.define_method("to_s", method!(Tokenizer::inspect, 0))?;
    
    Ok(())
}