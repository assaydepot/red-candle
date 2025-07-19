use magnus::{class, function, method, prelude::*, Error, RModule, RArray, RHash};
use candle_transformers::models::bert::{BertModel, Config};
use candle_core::{Device as CoreDevice, Tensor, DType, Module as CanModule};
use candle_nn::{VarBuilder, Linear};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::ruby::{Device, Result};
use crate::tokenizer::{TokenizerWrapper, loader::TokenizerLoader};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NERConfig {
    pub id2label: HashMap<i64, String>,
    pub label2id: HashMap<String, i64>,
}

#[derive(Debug, Clone)]
pub struct EntitySpan {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub token_start: usize,
    pub token_end: usize,
    pub confidence: f32,
}

#[magnus::wrap(class = "Candle::NER", free_immediately, size)]
pub struct NER {
    model: BertModel,
    tokenizer: TokenizerWrapper,
    classifier: Linear,
    config: NERConfig,
    device: CoreDevice,
    model_id: String,
}

impl NER {
    pub fn new(model_id: String, device: Option<Device>, tokenizer_id: Option<String>) -> Result<Self> {
        let device = device.unwrap_or(Device::Cpu).as_device()?;
        
        // Load model in a separate thread to avoid blocking
        let device_clone = device.clone();
        let model_id_clone = model_id.clone();
        
        let handle = std::thread::spawn(move || -> std::result::Result<(BertModel, TokenizerWrapper, Linear, NERConfig), Box<dyn std::error::Error + Send + Sync>> {
            let api = Api::new()?;
            let repo = api.repo(Repo::new(model_id_clone.clone(), RepoType::Model));
            
            // Download model files
            let config_filename = repo.get("config.json")?;
            
            // Handle tokenizer loading with optional tokenizer_id
            let tokenizer = if let Some(tok_id) = tokenizer_id {
                // Use the specified tokenizer
                let tok_repo = api.repo(Repo::new(tok_id, RepoType::Model));
                let tokenizer_filename = tok_repo.get("tokenizer.json")?;
                let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_filename)?;
                TokenizerLoader::with_padding(tokenizer, None)
            } else {
                // Try to load tokenizer from model repo
                let tokenizer_filename = repo.get("tokenizer.json")?;
                let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_filename)?;
                TokenizerLoader::with_padding(tokenizer, None)
            };
            let weights_filename = repo.get("pytorch_model.safetensors")
                .or_else(|_| repo.get("model.safetensors"))?;
            
            // Load BERT config
            let config_str = std::fs::read_to_string(&config_filename)?;
            let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
            let bert_config: Config = serde_json::from_value(config_json.clone())?;
            
            // Extract NER label configuration
            let id2label = config_json["id2label"]
                .as_object()
                .ok_or("Missing id2label in config")?
                .iter()
                .map(|(k, v)| {
                    let id = k.parse::<i64>().unwrap_or(0);
                    let label = v.as_str().unwrap_or("O").to_string();
                    (id, label)
                })
                .collect::<HashMap<_, _>>();
            
            let label2id = id2label.iter()
                .map(|(id, label)| (label.clone(), *id))
                .collect::<HashMap<_, _>>();
            
            let num_labels = id2label.len();
            let ner_config = NERConfig { id2label, label2id };
            
            // Load model weights
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device_clone)?
            };
            
            // Load BERT model
            let model = BertModel::load(vb.pp("bert"), &bert_config)?;
            
            // Load classification head for token classification
            let classifier = candle_nn::linear(
                bert_config.hidden_size,
                num_labels,
                vb.pp("classifier")
            )?;
            
            Ok((model, TokenizerWrapper::new(tokenizer), classifier, ner_config))
        });
        
        match handle.join() {
            Ok(Ok((model, tokenizer, classifier, config))) => {
                Ok(Self {
                    model,
                    tokenizer,
                    classifier,
                    config,
                    device,
                    model_id,
                })
            }
            Ok(Err(e)) => Err(Error::new(
                magnus::exception::runtime_error(),
                format!("Failed to load NER model: {}", e)
            )),
            Err(_) => Err(Error::new(
                magnus::exception::runtime_error(),
                "Thread panicked while loading NER model"
            )),
        }
    }
    
    /// Extract entities from text with confidence scores
    pub fn extract_entities(&self, text: String, confidence_threshold: Option<f64>) -> Result<RArray> {
        let threshold = confidence_threshold.unwrap_or(0.9) as f32;
        
        // Tokenize the text
        let encoding = self.tokenizer.inner().encode(text.as_str(), true)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Tokenization failed: {}", e)))?;
        
        let token_ids = encoding.get_ids();
        let tokens = encoding.get_tokens();
        let offsets = encoding.get_offsets();
        
        // Convert to tensors
        let input_ids = Tensor::new(token_ids, &self.device)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?; // Add batch dimension
        
        let attention_mask = Tensor::ones_like(&input_ids)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        let token_type_ids = Tensor::zeros_like(&input_ids)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        // Forward pass through BERT
        let output = self.model.forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        // Apply classifier to get logits for each token
        let logits = self.classifier.forward(&output)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        // Apply softmax to get probabilities
        let probs = candle_nn::ops::softmax(&logits, 2)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        // Get predictions and confidence scores
        let probs_vec: Vec<Vec<f32>> = probs.squeeze(0)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?
            .to_vec2()
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        // Extract entities with BIO decoding
        let entities = self.decode_entities(
            &text,
            &tokens.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            offsets,
            &probs_vec,
            threshold
        )?;
        
        // Convert to Ruby array
        let result = RArray::new();
        for entity in entities {
            let hash = RHash::new();
            hash.aset("text", entity.text)?;
            hash.aset("label", entity.label)?;
            hash.aset("start", entity.start)?;
            hash.aset("end", entity.end)?;
            hash.aset("confidence", entity.confidence)?;
            hash.aset("token_start", entity.token_start)?;
            hash.aset("token_end", entity.token_end)?;
            result.push(hash)?;
        }
        
        Ok(result)
    }
    
    /// Get token-level predictions with labels and confidence scores
    pub fn predict_tokens(&self, text: String) -> Result<RArray> {
        // Tokenize the text
        let encoding = self.tokenizer.inner().encode(text.as_str(), true)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Tokenization failed: {}", e)))?;
        
        let token_ids = encoding.get_ids();
        let tokens = encoding.get_tokens();
        
        // Convert to tensors
        let input_ids = Tensor::new(token_ids, &self.device)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        let attention_mask = Tensor::ones_like(&input_ids)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        let token_type_ids = Tensor::zeros_like(&input_ids)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        // Forward pass
        let output = self.model.forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        let logits = self.classifier.forward(&output)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        let probs = candle_nn::ops::softmax(&logits, 2)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        // Get predictions
        let probs_vec: Vec<Vec<f32>> = probs.squeeze(0)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?
            .to_vec2()
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        // Build result array
        let result = RArray::new();
        for (i, (token, probs)) in tokens.iter().zip(probs_vec.iter()).enumerate() {
            // Find best label
            let (label_id, confidence) = probs.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, conf)| (idx as i64, *conf))
                .unwrap_or((0, 0.0));
            
            let label = self.config.id2label.get(&label_id)
                .unwrap_or(&"O".to_string())
                .clone();
            
            let token_info = RHash::new();
            token_info.aset("token", token.to_string())?;
            token_info.aset("label", label)?;
            token_info.aset("confidence", confidence)?;
            token_info.aset("index", i)?;
            
            // Add probability distribution if needed
            let probs_hash = RHash::new();
            for (id, label) in &self.config.id2label {
                if let Some(prob) = probs.get(*id as usize) {
                    probs_hash.aset(label.as_str(), *prob)?;
                }
            }
            token_info.aset("probabilities", probs_hash)?;
            
            result.push(token_info)?;
        }
        
        Ok(result)
    }
    
    /// Decode BIO-tagged sequences into entity spans
    fn decode_entities(
        &self,
        text: &str,
        tokens: &[&str],
        offsets: &[(usize, usize)],
        probs: &[Vec<f32>],
        threshold: f32,
    ) -> Result<Vec<EntitySpan>> {
        let mut entities = Vec::new();
        let mut current_entity: Option<(String, usize, usize, Vec<f32>)> = None;
        
        for (i, (token, probs_vec)) in tokens.iter().zip(probs).enumerate() {
            // Skip special tokens
            if token.starts_with("[") && token.ends_with("]") {
                continue;
            }
            
            // Get predicted label
            let (label_id, confidence) = probs_vec.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, conf)| (idx as i64, *conf))
                .unwrap_or((0, 0.0));
            
            let label = self.config.id2label.get(&label_id)
                .unwrap_or(&"O".to_string())
                .clone();
            
            // BIO decoding logic
            if label == "O" || confidence < threshold {
                // End current entity if exists
                if let Some((entity_type, start_idx, end_idx, confidences)) = current_entity.take() {
                    if let (Some(start_offset), Some(end_offset)) = 
                        (offsets.get(start_idx), offsets.get(end_idx - 1)) {
                        let entity_text = text[start_offset.0..end_offset.1].to_string();
                        let avg_confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;
                        
                        entities.push(EntitySpan {
                            text: entity_text,
                            label: entity_type,
                            start: start_offset.0,
                            end: end_offset.1,
                            token_start: start_idx,
                            token_end: end_idx,
                            confidence: avg_confidence,
                        });
                    }
                }
            } else if label.starts_with("B-") {
                // Begin new entity
                if let Some((entity_type, start_idx, end_idx, confidences)) = current_entity.take() {
                    if let (Some(start_offset), Some(end_offset)) = 
                        (offsets.get(start_idx), offsets.get(end_idx - 1)) {
                        let entity_text = text[start_offset.0..end_offset.1].to_string();
                        let avg_confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;
                        
                        entities.push(EntitySpan {
                            text: entity_text,
                            label: entity_type,
                            start: start_offset.0,
                            end: end_offset.1,
                            token_start: start_idx,
                            token_end: end_idx,
                            confidence: avg_confidence,
                        });
                    }
                }
                
                let entity_type = label[2..].to_string();
                current_entity = Some((entity_type, i, i + 1, vec![confidence]));
            } else if label.starts_with("I-") {
                // Continue entity
                if let Some((ref mut entity_type, _, ref mut end_idx, ref mut confidences)) = current_entity {
                    let new_type = label[2..].to_string();
                    if *entity_type == new_type {
                        *end_idx = i + 1;
                        confidences.push(confidence);
                    } else {
                        // Type mismatch, start new entity
                        current_entity = Some((new_type, i, i + 1, vec![confidence]));
                    }
                } else {
                    // I- tag without B- tag, treat as beginning
                    let entity_type = label[2..].to_string();
                    current_entity = Some((entity_type, i, i + 1, vec![confidence]));
                }
            }
        }
        
        // Handle final entity
        if let Some((entity_type, start_idx, end_idx, confidences)) = current_entity {
            if let (Some(start_offset), Some(end_offset)) = 
                (offsets.get(start_idx), offsets.get(end_idx - 1)) {
                let entity_text = text[start_offset.0..end_offset.1].to_string();
                let avg_confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;
                
                entities.push(EntitySpan {
                    text: entity_text,
                    label: entity_type,
                    start: start_offset.0,
                    end: end_offset.1,
                    token_start: start_idx,
                    token_end: end_idx,
                    confidence: avg_confidence,
                });
            }
        }
        
        Ok(entities)
    }
    
    /// Get the label configuration
    pub fn labels(&self) -> Result<RHash> {
        let hash = RHash::new();
        
        let id2label = RHash::new();
        for (id, label) in &self.config.id2label {
            id2label.aset(*id, label.as_str())?;
        }
        
        let label2id = RHash::new();
        for (label, id) in &self.config.label2id {
            label2id.aset(label.as_str(), *id)?;
        }
        
        hash.aset("id2label", id2label)?;
        hash.aset("label2id", label2id)?;
        hash.aset("num_labels", self.config.id2label.len())?;
        
        Ok(hash)
    }
    
    /// Get the tokenizer
    pub fn tokenizer(&self) -> Result<crate::ruby::tokenizer::Tokenizer> {
        Ok(crate::ruby::tokenizer::Tokenizer(self.tokenizer.clone()))
    }
    
    /// Get model info
    pub fn model_info(&self) -> String {
        format!("NER model: {}, labels: {}", self.model_id, self.config.id2label.len())
    }
}

pub fn init(rb_candle: RModule) -> Result<()> {
    let ner_class = rb_candle.define_class("NER", class::object())?;
    ner_class.define_singleton_method("new", function!(NER::new, 3))?;
    ner_class.define_method("extract_entities", method!(NER::extract_entities, 2))?;
    ner_class.define_method("predict_tokens", method!(NER::predict_tokens, 1))?;
    ner_class.define_method("labels", method!(NER::labels, 0))?;
    ner_class.define_method("tokenizer", method!(NER::tokenizer, 0))?;
    ner_class.define_method("model_info", method!(NER::model_info, 0))?;
    
    Ok(())
}