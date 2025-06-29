use magnus::{class, function, method, prelude::*, Error, RModule, Float, RArray};
use candle_transformers::models::bert::{BertModel, Config};
use candle_core::{Device, Tensor, IndexOp, DType};
use candle_nn::{VarBuilder, ops::sigmoid, Linear, Module};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer, EncodeInput};
use std::thread;

#[magnus::wrap(class = "Candle::Reranker", free_immediately, size)]
pub struct Reranker {
    model: BertModel,
    tokenizer: Tokenizer,
    linear_layer: Linear,
    device: Device,
}

impl Reranker {
    pub fn new(model_id: String) -> Result<Self, Error> {
        Self::new_with_device(model_id, Device::Cpu)
    }
    
    pub fn new_cuda(model_id: String) -> Result<Self, Error> {
        match Device::cuda_if_available(0) {
            Ok(device) => Self::new_with_device(model_id, device),
            Err(_) => Self::new_with_device(model_id, Device::Cpu),
        }
    }
    
    pub fn new_with_device(model_id: String, device: Device) -> Result<Self, Error> {
        let device_clone = device.clone();
        let handle = thread::spawn(move || -> Result<(BertModel, Tokenizer, Linear), Box<dyn std::error::Error + Send + Sync>> {
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
            
            // Initialize linear layer for cross-encoder (single output score)
            // For cross-encoder models, the linear layer is typically at "classifier"
            let linear_layer = candle_nn::linear(config.hidden_size, 1, vb.pp("classifier"))?;
            
            Ok((model, tokenizer, linear_layer))
        });
        
        match handle.join() {
            Ok(Ok((model, tokenizer, linear_layer))) => {
                Ok(Self { model, tokenizer, linear_layer, device })
            }
            Ok(Err(e)) => Err(Error::new(magnus::exception::runtime_error(), format!("Failed to load model: {}", e))),
            Err(_) => Err(Error::new(magnus::exception::runtime_error(), "Thread panicked while loading model")),
        }
    }

    pub fn rerank(&self, query: String, documents: RArray) -> Result<RArray, Error> {
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

        let token_ids = Tensor::new(token_ids, &self.device)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to create tensor: {}", e)))?;
        let token_type_ids = token_ids.zeros_like()
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to create token type ids: {}", e)))?;
        let attention_mask = token_ids.ne(0u32)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to create attention mask: {}", e)))?;
        
        // Forward pass through BERT
        let embeddings = self.model.forward(&token_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Model forward pass failed: {}", e)))?;
        
        // Extract [CLS] token embeddings
        let cls_embeddings = embeddings.i((.., 0))
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to extract CLS token: {}", e)))?;
        
        // Apply linear layer and sigmoid to get relevance scores
        let logits = self.linear_layer.forward(&cls_embeddings)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Linear layer forward failed: {}", e)))?;
        let scores = sigmoid(&logits.squeeze(1)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to squeeze tensor: {}", e)))?)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Sigmoid failed: {}", e)))?;
        
        let scores_vec: Vec<f32> = scores.to_vec1()
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to convert scores to vec: {}", e)))?;

        // Sort documents by relevance score
        let mut ranked_docs: Vec<(String, f32)> = documents.into_iter().zip(scores_vec).collect();
        ranked_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Build result array
        let result_array = RArray::new();
        for (doc, score) in ranked_docs {
            let pair = RArray::new();
            pair.push(doc)?;
            pair.push(Float::from_f64(score as f64))?;
            result_array.push(pair)?;
        }
        Ok(result_array)
    }
}

pub fn init(rb_candle: RModule) -> Result<(), Error> {
    let c_reranker = rb_candle.define_class("Reranker", class::object())?;
    c_reranker.define_singleton_method("new", function!(Reranker::new, 1))?;
    c_reranker.define_singleton_method("new_cuda", function!(Reranker::new_cuda, 1))?;
    c_reranker.define_method("rerank", method!(Reranker::rerank, 2))?;
    Ok(())
}