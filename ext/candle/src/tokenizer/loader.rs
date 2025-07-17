use candle_core::Result as CandleResult;
use hf_hub::api::tokio::{Api, ApiRepo};
use tokenizers::Tokenizer;
use tokenizers::{PaddingParams, PaddingStrategy, TruncationParams};
use std::path::PathBuf;

/// Standard padding configuration for models
pub fn standard_padding_params() -> PaddingParams {
    PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        direction: tokenizers::PaddingDirection::Left,
        pad_to_multiple_of: None,
        pad_id: 0,
        pad_type_id: 0,
        pad_token: "[PAD]".to_string(),
    }
}

/// Unified tokenizer loader with common download logic
pub struct TokenizerLoader;

impl TokenizerLoader {
    /// Load tokenizer from a local file path
    pub fn from_file(path: &str) -> CandleResult<Tokenizer> {
        Tokenizer::from_file(path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer from file: {}", e)))
    }

    /// Download and load tokenizer from HuggingFace
    pub async fn from_hf_hub(repo_id: &str, filename: Option<&str>) -> CandleResult<Tokenizer> {
        let api = Api::new()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create HF API: {}", e)))?;
        
        let repo = api.model(repo_id.to_string());
        let tokenizer_path = Self::download_tokenizer_file(&repo, filename).await?;
        
        Self::from_file(tokenizer_path.to_str()
            .ok_or_else(|| candle_core::Error::Msg("Invalid tokenizer path".to_string()))?)
    }

    /// Download tokenizer file from repository
    async fn download_tokenizer_file(repo: &ApiRepo, filename: Option<&str>) -> CandleResult<PathBuf> {
        if let Some(file) = filename {
            // Try specific filename
            repo.get(file).await
                .map_err(|e| candle_core::Error::Msg(
                    format!("Failed to download tokenizer file '{}': {}", file, e)
                ))
        } else {
            // Try common tokenizer filenames in order
            let filenames = ["tokenizer.json", "tokenizer.model"];
            
            for file in filenames {
                if let Ok(path) = repo.get(file).await {
                    return Ok(path);
                }
            }
            
            Err(candle_core::Error::Msg(
                "No tokenizer file found. Tried: tokenizer.json, tokenizer.model".to_string()
            ))
        }
    }

    /// Configure tokenizer with standard padding for batch processing
    pub fn with_padding(mut tokenizer: Tokenizer, padding_params: Option<PaddingParams>) -> Tokenizer {
        let params = padding_params.unwrap_or_else(standard_padding_params);
        let _ = tokenizer.with_padding(Some(params));
        tokenizer
    }

    /// Configure tokenizer with truncation
    pub fn with_truncation(mut tokenizer: Tokenizer, max_length: usize) -> Tokenizer {
        let _ = tokenizer.with_truncation(Some(TruncationParams {
            max_length,
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            stride: 0,
            direction: tokenizers::TruncationDirection::Right,
        }));
        tokenizer
    }

    /// Download tokenizer from a specific source (for GGUF models)
    pub async fn from_source(api: &Api, source: &str) -> CandleResult<PathBuf> {
        // Check if it's a local file path
        if source.ends_with(".json") && std::path::Path::new(source).exists() {
            return Ok(PathBuf::from(source));
        }
        
        // Otherwise treat it as a HuggingFace repo
        let repo = api.model(source.to_string());
        
        // Try tokenizer.json first
        if let Ok(path) = repo.get("tokenizer.json").await {
            return Ok(path);
        }
        
        // Try tokenizer.model (for models that use sentencepiece)
        if let Ok(path) = repo.get("tokenizer.model").await {
            return Ok(path);
        }
        
        Err(candle_core::Error::Msg(format!(
            "Failed to find tokenizer in specified source: {}. Please check network connectivity and that the model repository exists.",
            source
        )))
    }
}