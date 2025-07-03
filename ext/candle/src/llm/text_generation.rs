use candle_core::{Result as CandleResult, Tensor};
use candle_transformers::generation::LogitsProcessor;
use rand::{rngs::StdRng, SeedableRng};

use super::GenerationConfig;

/// Helper struct for text generation process
pub struct TextGeneration {
    #[allow(dead_code)]
    rng: StdRng,
    logits_processor: LogitsProcessor,
    tokens: Vec<u32>,
    eos_token_id: Option<u32>,
}

impl TextGeneration {
    pub fn new(
        seed: u64,
        temperature: Option<f64>,
        top_p: Option<f64>,
        _top_k: Option<usize>,
        _repetition_penalty: f32,
        _repetition_penalty_last_n: usize,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temperature, top_p);
        
        Self {
            rng: StdRng::seed_from_u64(seed),
            logits_processor,
            tokens: Vec::new(),
            eos_token_id: None,
        }
    }

    pub fn from_config(config: &GenerationConfig) -> Self {
        Self::new(
            config.seed,
            Some(config.temperature),
            config.top_p,
            config.top_k,
            config.repetition_penalty,
            config.repetition_penalty_last_n,
        )
    }

    pub fn set_eos_token_id(&mut self, eos_token_id: u32) {
        self.eos_token_id = Some(eos_token_id);
    }

    pub fn set_tokens(&mut self, tokens: Vec<u32>) {
        self.tokens = tokens;
    }

    pub fn get_tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn push_token(&mut self, token: u32) {
        self.tokens.push(token);
    }

    /// Apply repetition penalty to logits
    pub fn apply_repetition_penalty(
        &self,
        logits: &mut Tensor,
        penalty: f32,
        context_size: usize,
    ) -> CandleResult<()> {
        if penalty == 1.0 {
            return Ok(());
        }

        let device = logits.device();
        let (_b_size, vocab_size) = logits.dims2()?;
        
        // Get the context tokens to apply penalty to
        let start = self.tokens.len().saturating_sub(context_size);
        let context_tokens = &self.tokens[start..];
        
        // Apply penalty to tokens that appear in the context
        let mut logits_vec = logits.to_vec1::<f32>()?;
        for &token in context_tokens {
            if (token as usize) < vocab_size {
                let idx = token as usize;
                if logits_vec[idx] > 0.0 {
                    logits_vec[idx] /= penalty;
                } else {
                    logits_vec[idx] *= penalty;
                }
            }
        }
        
        *logits = Tensor::from_vec(logits_vec, vocab_size, device)?;
        Ok(())
    }

    /// Sample next token from logits
    pub fn sample_next_token(
        &mut self,
        logits: &Tensor,
        repetition_penalty: Option<(f32, usize)>,
    ) -> CandleResult<u32> {
        let mut logits = logits.clone();
        
        // Apply repetition penalty if specified
        if let Some((penalty, last_n)) = repetition_penalty {
            self.apply_repetition_penalty(&mut logits, penalty, last_n)?;
        }
        
        // Sample token
        let next_token = self.logits_processor.sample(&logits)?;
        self.tokens.push(next_token);
        
        Ok(next_token)
    }

    /// Check if we should stop generation
    pub fn should_stop(&self, token: u32, max_length: usize) -> bool {
        if self.tokens.len() >= max_length {
            return true;
        }
        
        if let Some(eos) = self.eos_token_id {
            if token == eos {
                return true;
            }
        }
        
        false
    }

    /// Check if the generated text ends with any stop sequence
    pub fn check_stop_sequences(&self, text: &str, stop_sequences: &[String]) -> bool {
        for seq in stop_sequences {
            if text.ends_with(seq) {
                return true;
            }
        }
        false
    }
}