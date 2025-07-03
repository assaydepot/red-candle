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
            seed: 42,
            stop_sequences: vec![],
            include_prompt: false,
        }
    }
}

impl GenerationConfig {
    /// Create a deterministic configuration (temperature = 0)
    pub fn deterministic() -> Self {
        Self {
            temperature: 0.0,
            top_p: None,
            top_k: Some(1),
            ..Default::default()
        }
    }

    /// Create a creative configuration (higher temperature)
    pub fn creative() -> Self {
        Self {
            temperature: 1.0,
            top_p: Some(0.95),
            top_k: Some(50),
            repetition_penalty: 1.2,
            ..Default::default()
        }
    }

    /// Create a balanced configuration
    pub fn balanced() -> Self {
        Self {
            temperature: 0.7,
            top_p: Some(0.9),
            top_k: Some(40),
            ..Default::default()
        }
    }
}