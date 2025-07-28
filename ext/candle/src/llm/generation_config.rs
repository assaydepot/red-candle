use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::Arc;
use crate::structured::Index;

/// Configuration for text generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// The maximum number of tokens to generate
    pub max_length: usize,
    /// The temperature for sampling
    pub temperature: f64,
    /// The top-p value for nucleus sampling
    pub top_p: Option<f64>,
    /// The top-k value for top-k sampling
    pub top_k: Option<usize>,
    /// The repetition penalty
    pub repetition_penalty: f32,
    /// The repetition penalty range
    pub repetition_penalty_last_n: usize,
    /// Random seed for sampling
    pub seed: u64,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
    /// Whether to return the prompt in the output
    pub include_prompt: bool,
    /// Whether to show raw tokens during generation (for debugging)
    pub debug_tokens: bool,
    /// Optional constraint index for structured generation
    pub constraint: Option<Arc<Index>>,
    /// Stop immediately when constraint is satisfied
    pub stop_on_constraint_satisfaction: bool,
    /// Whether to stop immediately when pattern is matched (vs allowing continuation)
    pub stop_on_match: bool,
}

/// Generate a random seed based on current time
fn random_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(42)
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            temperature: 0.7,
            top_p: None,
            top_k: None,
            repetition_penalty: 1.1,
            repetition_penalty_last_n: 64,
            seed: random_seed(),
            stop_sequences: vec![],
            include_prompt: false,
            debug_tokens: false,
            constraint: None,
            stop_on_constraint_satisfaction: true,
            stop_on_match: true,
        }
    }
}

