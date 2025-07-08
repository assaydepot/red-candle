use magnus::{class, function, method, prelude::*, Error, RModule, Float, RArray};
use candle_transformers::models::bert::{BertModel, Config};
use candle_core::{Device as CoreDevice, Tensor, IndexOp, DType};
use candle_nn::{VarBuilder, Linear, Module, ops::sigmoid};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer, EncodeInput};
use std::thread;
use crate::ruby::{Device as RbDevice, Result as RbResult};

#[magnus::wrap(class = "Candle::Reranker", free_immediately, size)]
pub struct Reranker {
    model: BertModel,
    tokenizer: Tokenizer,
    pooler: Linear,
    classifier: Linear,
    device: CoreDevice,
}

impl Reranker {
    pub fn new(model_id: String, device: Option<RbDevice>) -> RbResult<Self> {
        let device = device.unwrap_or(RbDevice::Cpu).as_device()?;
        Self::new_with_core_device(model_id, device)
    }
        
    fn new_with_core_device(model_id: String, device: CoreDevice) -> Result<Self, Error> {
        let device_clone = device.clone();
        let handle = thread::spawn(move || -> Result<(BertModel, Tokenizer, Linear, Linear), Box<dyn std::error::Error + Send + Sync>> {
            let api = Api::new()?;
            let repo = api.repo(Repo::new(model_id.clone(), RepoType::Model));
            
            // Download model files
            let config_filename = repo.get("config.json")?;
            let tokenizer_filename = repo.get("tokenizer.json")?;
            let weights_filename = repo.get("model.safetensors")?;
            
            // Load config
            let config = std::fs::read_to_string(config_filename)?;
            let config: Config = serde_json::from_str(&config)?;

            // Setup tokenizer with padding
            let mut tokenizer = Tokenizer::from_file(tokenizer_filename)?;
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
            
            // Load model weights
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device_clone)?
            };
            
            // Load BERT model
            let model = BertModel::load(vb.pp("bert"), &config)?;
            
            // Load pooler layer (dense + tanh activation)
            let pooler = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("bert.pooler.dense"))?;
            
            // Load classifier layer for cross-encoder (single output score)
            let classifier = candle_nn::linear(config.hidden_size, 1, vb.pp("classifier"))?;
            
            Ok((model, tokenizer, pooler, classifier))
        });
        
        match handle.join() {
            Ok(Ok((model, tokenizer, pooler, classifier))) => {
                Ok(Self { model, tokenizer, pooler, classifier, device })
            }
            Ok(Err(e)) => Err(Error::new(magnus::exception::runtime_error(), format!("Failed to load model: {}", e))),
            Err(_) => Err(Error::new(magnus::exception::runtime_error(), "Thread panicked while loading model")),
        }
    }
    
    pub fn debug_tokenization(&self, query: String, document: String) -> Result<magnus::RHash, Error> {
        // Create query-document pair for cross-encoder
        let query_doc_pair: EncodeInput = (query.clone(), document.clone()).into();
        
        // Tokenize
        let encoding = self.tokenizer.encode(query_doc_pair, true)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Tokenization failed: {}", e)))?;
        
        // Get token information
        let token_ids = encoding.get_ids().to_vec();
        let token_type_ids = encoding.get_type_ids().to_vec();
        let attention_mask = encoding.get_attention_mask().to_vec();
        let tokens = encoding.get_tokens().iter().map(|t| t.to_string()).collect::<Vec<_>>();
        
        // Create result hash
        let result = magnus::RHash::new();
        result.aset("token_ids", RArray::from_vec(token_ids.iter().map(|&id| id as i64).collect::<Vec<_>>()))?;
        result.aset("token_type_ids", RArray::from_vec(token_type_ids.iter().map(|&id| id as i64).collect::<Vec<_>>()))?;
        result.aset("attention_mask", RArray::from_vec(attention_mask.iter().map(|&mask| mask as i64).collect::<Vec<_>>()))?;
        result.aset("tokens", RArray::from_vec(tokens))?;
        
        Ok(result)
    }
    
    pub fn rerank_with_options(&self, query: String, documents: RArray, pooling_method: String, apply_sigmoid: bool) -> Result<RArray, Error> {
        let documents: Vec<String> = documents.to_vec()?;
        
        // Create query-document pairs for cross-encoder
        let query_and_docs: Vec<EncodeInput> = documents
            .iter()
            .map(|d| (query.clone(), d.clone()).into())
            .collect();

        // Tokenize batch
        let encodings = self.tokenizer.encode_batch(query_and_docs, true)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Tokenization failed: {}", e)))?;
        
        // Convert to tensors
        let token_ids = encodings
            .iter()
            .map(|e| e.get_ids().to_vec())
            .collect::<Vec<_>>();
            
        let token_type_ids = encodings
            .iter()
            .map(|e| e.get_type_ids().to_vec())
            .collect::<Vec<_>>();

        let token_ids = Tensor::new(token_ids, &self.device)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to create tensor: {}", e)))?;
        let token_type_ids = Tensor::new(token_type_ids, &self.device)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to create token type ids tensor: {}", e)))?;
        let attention_mask = token_ids.ne(0u32)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to create attention mask: {}", e)))?;
        
        // Forward pass through BERT
        let embeddings = self.model.forward(&token_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Model forward pass failed: {}", e)))?;
        
        // Apply pooling based on the specified method
        let pooled_embeddings = match pooling_method.as_str() {
            "pooler" => {
                // Extract [CLS] token and apply pooler (dense + tanh)
                // Work around Metal indexing issue by using narrow instead of i((.., 0))
                let cls_embeddings = if self.device.is_metal() {
                    // Metal has issues with tensor indexing, use a different approach
                    let (batch_size, _seq_len, hidden_size) = embeddings.dims3()
                        .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to get dims: {}", e)))?;
                    
                    // Reshape to [batch * seq_len, hidden] then take first hidden vectors for each batch
                    let reshaped = embeddings.reshape((batch_size * _seq_len, hidden_size))
                        .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to reshape: {}", e)))?;
                    
                    // Extract CLS tokens (first token of each sequence)
                    let mut cls_vecs = Vec::new();
                    for i in 0..batch_size {
                        let start_idx = i * _seq_len;
                        let cls_vec = reshaped.narrow(0, start_idx, 1)
                            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to extract CLS: {}", e)))?;
                        cls_vecs.push(cls_vec);
                    }
                    
                    // Stack the CLS vectors
                    Tensor::cat(&cls_vecs, 0)
                        .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to cat CLS tokens: {}", e)))?
                        .contiguous()
                        .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to make contiguous: {}", e)))?
                } else {
                    embeddings.i((.., 0))
                        .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to extract CLS token: {}", e)))?
                };
                // Ensure tensor is contiguous before linear layer
                let cls_embeddings = cls_embeddings.contiguous()
                    .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to make cls_embeddings contiguous: {}", e)))?;
                let pooled = self.pooler.forward(&cls_embeddings)
                    .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Pooler forward failed: {}", e)))?;
                pooled.tanh()
                    .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Tanh activation failed: {}", e)))?
            },
            "cls" => {
                // Just use the [CLS] token embeddings directly (no pooler layer)
                // Work around Metal indexing issue
                let cls_embeddings = if self.device.is_metal() {
                    // Use same approach as pooler method
                    let (batch_size, _seq_len, hidden_size) = embeddings.dims3()
                        .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to get dims: {}", e)))?;
                    
                    let reshaped = embeddings.reshape((batch_size * _seq_len, hidden_size))
                        .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to reshape: {}", e)))?;
                    
                    let mut cls_vecs = Vec::new();
                    for i in 0..batch_size {
                        let start_idx = i * _seq_len;
                        let cls_vec = reshaped.narrow(0, start_idx, 1)
                            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to extract CLS: {}", e)))?;
                        cls_vecs.push(cls_vec);
                    }
                    
                    Tensor::cat(&cls_vecs, 0)
                        .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to cat CLS tokens: {}", e)))?
                        .contiguous()
                        .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to make contiguous: {}", e)))?
                } else {
                    embeddings.i((.., 0))
                        .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to extract CLS token: {}", e)))?
                };
                // Ensure contiguous for classifier
                cls_embeddings.contiguous()
                    .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to make CLS embeddings contiguous: {}", e)))?
            },
            "mean" => {
                // Mean pooling across all tokens
                let (_batch, seq_len, _hidden) = embeddings.dims3()
                    .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to get tensor dimensions: {}", e)))?;
                let sum = embeddings.sum(1)
                    .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to sum embeddings: {}", e)))?;
                (sum / (seq_len as f64))
                    .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to compute mean: {}", e)))?
            },
            _ => return Err(Error::new(magnus::exception::runtime_error(), 
                format!("Unknown pooling method: {}. Use 'pooler', 'cls', or 'mean'", pooling_method)))
        };
        
        // Apply classifier to get relevance scores (raw logits)
        // Ensure tensor is contiguous before linear layer
        let pooled_embeddings = pooled_embeddings.contiguous()
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to make pooled_embeddings contiguous: {}", e)))?;
        let logits = self.classifier.forward(&pooled_embeddings)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Classifier forward failed: {}", e)))?;
        let scores = logits.squeeze(1)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to squeeze tensor: {}", e)))?;
        
        // Optionally apply sigmoid activation
        let scores = if apply_sigmoid {
            sigmoid(&scores)
                .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Sigmoid failed: {}", e)))?
        } else {
            scores
        };
        
        let scores_vec: Vec<f32> = scores.to_vec1()
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to convert scores to vec: {}", e)))?;

        // Create tuples with document, score, and original index
        let mut ranked_docs: Vec<(String, f32, usize)> = documents
            .into_iter()
            .zip(scores_vec)
            .enumerate()
            .map(|(idx, (doc, score))| (doc, score, idx))
            .collect();
        
        // Sort documents by relevance score (descending)
        ranked_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Build result array with [doc, score, doc_id]
        let result_array = RArray::new();
        for (doc, score, doc_id) in ranked_docs {
            let tuple = RArray::new();
            tuple.push(doc)?;
            tuple.push(Float::from_f64(score as f64))?;
            tuple.push(doc_id)?;
            result_array.push(tuple)?;
        }
        Ok(result_array)
    }
}

pub fn init(rb_candle: RModule) -> Result<(), Error> {
    let c_reranker = rb_candle.define_class("Reranker", class::object())?;
    c_reranker.define_singleton_method("_create", function!(Reranker::new, 2))?;
    c_reranker.define_method("rerank_with_options", method!(Reranker::rerank_with_options, 4))?;
    c_reranker.define_method("debug_tokenization", method!(Reranker::debug_tokenization, 2))?;
    Ok(())
}