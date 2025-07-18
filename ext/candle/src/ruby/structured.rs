use magnus::{Error, Module, RModule, TryConvert, function, class, Object};
use std::sync::Arc;

use crate::structured::{SchemaProcessor, VocabularyAdapter, Index};
use crate::ruby::{Result, tokenizer::Tokenizer};

/// Ruby wrapper for structured generation constraints
#[derive(Clone, Debug)]
#[magnus::wrap(class = "Candle::StructuredConstraint", mark, free_immediately)]
pub struct StructuredConstraint {
    pub(crate) index: Arc<Index>,
}

impl StructuredConstraint {
    /// Create a constraint from a JSON schema
    pub fn from_schema(schema: String, tokenizer: &Tokenizer) -> Result<Self> {
        let vocabulary = VocabularyAdapter::from_tokenizer(&tokenizer.0)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to create vocabulary: {}", e)))?;
        
        let processor = SchemaProcessor::new();
        let index = processor.process_schema(&schema, &vocabulary)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to process schema: {}", e)))?;
        
        Ok(Self { index })
    }
    
    /// Create a constraint from a regex pattern
    pub fn from_regex(pattern: String, tokenizer: &Tokenizer) -> Result<Self> {
        let vocabulary = VocabularyAdapter::from_tokenizer(&tokenizer.0)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to create vocabulary: {}", e)))?;
        
        let processor = SchemaProcessor::new();
        let index = processor.process_regex(&pattern, &vocabulary)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to process regex: {}", e)))?;
        
        Ok(Self { index })
    }
}

pub fn init_structured(rb_candle: RModule) -> Result<()> {
    let class = rb_candle.define_class("StructuredConstraint", magnus::class::object())?;
    
    class.define_singleton_method("from_schema", function!(StructuredConstraint::from_schema, 2))?;
    class.define_singleton_method("from_regex", function!(StructuredConstraint::from_regex, 2))?;
    
    Ok(())
}