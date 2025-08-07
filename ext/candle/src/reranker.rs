use magnus::{class, function, method, prelude::*, Error, RModule, Float, RArray};
use candle_transformers::models::bert::{BertModel, Config};
use candle_core::{Device as CoreDevice, Tensor, IndexOp, DType};
use candle_nn::{VarBuilder, Linear, Module, ops::sigmoid};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{EncodeInput, Tokenizer};
use crate::ruby::{Device, Result};
use crate::tokenizer::{TokenizerWrapper, loader::TokenizerLoader};

#[magnus::wrap(class = "Candle::Reranker", free_immediately, size)]
pub struct Reranker {
    model: BertModel,
    tokenizer: TokenizerWrapper,
    pooler: Linear,
    classifier: Linear,
    device: CoreDevice,
    model_id: String,
}

impl Reranker {
    pub fn new(model_id: String, device: Option<Device>) -> Result<Self> {
        let device = device.unwrap_or(Device::best()).as_device()?;
        Self::new_with_core_device(model_id, device)
    }
        
    fn new_with_core_device(model_id: String, device: CoreDevice) -> std::result::Result<Self, Error> {
        let result = (|| -> std::result::Result<(BertModel, TokenizerWrapper, Linear, Linear), Box<dyn std::error::Error + Send + Sync>> {
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
            let tokenizer = Tokenizer::from_file(tokenizer_filename)?;
            let tokenizer = TokenizerLoader::with_padding(tokenizer, None);
            
            // Load model weights
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)?
            };
            
            // Load BERT model
            let model = BertModel::load(vb.pp("bert"), &config)?;
            
            // Load pooler layer (dense + tanh activation)
            let pooler = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("bert.pooler.dense"))?;
            
            // Load classifier layer for cross-encoder (single output score)
            let classifier = candle_nn::linear(config.hidden_size, 1, vb.pp("classifier"))?;
            
            Ok((model, TokenizerWrapper::new(tokenizer), pooler, classifier))
        })();
        
        match result {
            Ok((model, tokenizer, pooler, classifier)) => {
                Ok(Self { model, tokenizer, pooler, classifier, device, model_id })
            }
            Err(e) => Err(Error::new(magnus::exception::runtime_error(), format!("Failed to load model: {}", e))),
        }
    }
    
    /// Extract CLS embeddings from the model output, handling Metal device workarounds
    fn extract_cls_embeddings(&self, embeddings: &Tensor) -> std::result::Result<Tensor, Error> {
        let cls_embeddings = if self.device.is_metal() {
            // Metal has issues with tensor indexing, use a different approach
            let (batch_size, seq_len, hidden_size) = embeddings.dims3()
                .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to get dims: {}", e)))?;
            
            // Reshape to [batch * seq_len, hidden] then take first hidden vectors for each batch
            let reshaped = embeddings.reshape((batch_size * seq_len, hidden_size))
                .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to reshape: {}", e)))?;
            
            // Extract CLS tokens (first token of each sequence)
            let mut cls_vecs = Vec::new();
            for i in 0..batch_size {
                let start_idx = i * seq_len;
                let cls_vec = reshaped.narrow(0, start_idx, 1)
                    .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to extract CLS: {}", e)))?;
                cls_vecs.push(cls_vec);
            }
            
            // Stack the CLS vectors
            Tensor::cat(&cls_vecs, 0)
                .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to cat CLS tokens: {}", e)))?
        } else {
            embeddings.i((.., 0))
                .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to extract CLS token: {}", e)))?
        };
        
        // Ensure tensor is contiguous for downstream operations
        cls_embeddings.contiguous()
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to make CLS embeddings contiguous: {}", e)))
    }
    
    pub fn debug_tokenization(&self, query: String, document: String) -> std::result::Result<magnus::RHash, Error> {
        // Create query-document pair for cross-encoder
        let query_doc_pair: EncodeInput = (query.clone(), document.clone()).into();
        
        // Tokenize using the inner tokenizer for detailed info
        let encoding = self.tokenizer.inner().encode(query_doc_pair, true)
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
    
    pub fn rerank_with_options(&self, query: String, documents: RArray, pooling_method: String, apply_sigmoid: bool) -> std::result::Result<RArray, Error> {
        let documents: Vec<String> = documents.to_vec()?;
        
        // Create query-document pairs for cross-encoder
        let query_and_docs: Vec<EncodeInput> = documents
            .iter()
            .map(|d| (query.clone(), d.clone()).into())
            .collect();

        // Tokenize batch using inner tokenizer for access to token type IDs
        let encodings = self.tokenizer.inner().encode_batch(query_and_docs, true)
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
                let cls_embeddings = self.extract_cls_embeddings(&embeddings)?;
                let pooled = self.pooler.forward(&cls_embeddings)
                    .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Pooler forward failed: {}", e)))?;
                pooled.tanh()
                    .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Tanh activation failed: {}", e)))?
            },
            "cls" => {
                // Just use the [CLS] token embeddings directly (no pooler layer)
                self.extract_cls_embeddings(&embeddings)?
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

    /// Get the tokenizer used by this model
    pub fn tokenizer(&self) -> std::result::Result<crate::ruby::tokenizer::Tokenizer, Error> {
        Ok(crate::ruby::tokenizer::Tokenizer(self.tokenizer.clone()))
    }
    
    /// Get the model_id
    pub fn model_id(&self) -> String {
        self.model_id.clone()
    }
    
    /// Get the device
    pub fn device(&self) -> Device {
        Device::from_device(&self.device)
    }
    
    /// Get all options as a hash
    pub fn options(&self) -> std::result::Result<magnus::RHash, Error> {
        let hash = magnus::RHash::new();
        hash.aset("model_id", self.model_id.clone())?;
        hash.aset("device", self.device().__str__())?;
        Ok(hash)
    }
}

pub fn init(rb_candle: RModule) -> std::result::Result<(), Error> {
    let c_reranker = rb_candle.define_class("Reranker", class::object())?;
    c_reranker.define_singleton_method("_create", function!(Reranker::new, 2))?;
    c_reranker.define_method("rerank_with_options", method!(Reranker::rerank_with_options, 4))?;
    c_reranker.define_method("debug_tokenization", method!(Reranker::debug_tokenization, 2))?;
    c_reranker.define_method("tokenizer", method!(Reranker::tokenizer, 0))?;
    c_reranker.define_method("model_id", method!(Reranker::model_id, 0))?;
    c_reranker.define_method("device", method!(Reranker::device, 0))?;
    c_reranker.define_method("options", method!(Reranker::options, 0))?;
    Ok(())
}