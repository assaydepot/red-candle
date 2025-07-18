use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use candle_core::Result as CandleResult;
use candle_core::Error as CandleError;
use outlines_core::prelude::Index;
use outlines_core::vocabulary::Vocabulary;
use serde_json::Value as JsonValue;
use outlines_core::json_schema;

/// Processes JSON schemas into compiled Index objects for structured generation
pub struct SchemaProcessor {
    /// Cache of compiled Index objects keyed by schema hash
    cache: Arc<Mutex<HashMap<u64, Arc<Index>>>>,
}

impl SchemaProcessor {
    /// Create a new schema processor with an empty cache
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// Process a JSON schema into a compiled Index
    /// 
    /// # Arguments
    /// * `schema` - JSON schema as a string
    /// * `vocabulary` - The tokenizer's vocabulary
    /// 
    /// # Returns
    /// A compiled Index ready for constrained generation
    pub fn process_schema(
        &self,
        schema: &str,
        vocabulary: &Vocabulary,
    ) -> CandleResult<Arc<Index>> {
        // Calculate hash of the schema for caching
        let schema_hash = self.calculate_hash(schema);
        
        // Check cache first
        if let Ok(cache) = self.cache.lock() {
            if let Some(cached_index) = cache.get(&schema_hash) {
                return Ok(Arc::clone(cached_index));
            }
        }
        
        // Parse the JSON schema
        let schema_value: JsonValue = serde_json::from_str(schema)
            .map_err(|e| CandleError::Msg(format!("Invalid JSON schema: {}", e)))?;
        
        // Convert schema to regex using Outlines
        let regex = self.schema_to_regex(&schema_value)?;
        
        // Compile regex into Index
        let index = self.compile_regex(&regex, vocabulary)?;
        let index_arc = Arc::new(index);
        
        // Cache the compiled Index
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(schema_hash, Arc::clone(&index_arc));
        }
        
        Ok(index_arc)
    }
    
    /// Process a regex pattern directly into an Index
    /// 
    /// # Arguments
    /// * `regex` - Regular expression pattern
    /// * `vocabulary` - The tokenizer's vocabulary
    /// 
    /// # Returns
    /// A compiled Index for the regex pattern
    pub fn process_regex(
        &self,
        regex: &str,
        vocabulary: &Vocabulary,
    ) -> CandleResult<Arc<Index>> {
        // Calculate hash for caching
        let regex_hash = self.calculate_hash(regex);
        
        // Check cache
        if let Ok(cache) = self.cache.lock() {
            if let Some(cached_index) = cache.get(&regex_hash) {
                return Ok(Arc::clone(cached_index));
            }
        }
        
        // Compile the regex
        let index = self.compile_regex(regex, vocabulary)?;
        let index_arc = Arc::new(index);
        
        // Cache it
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(regex_hash, Arc::clone(&index_arc));
        }
        
        Ok(index_arc)
    }
    
    /// Convert a JSON schema to a regex pattern
    fn schema_to_regex(&self, schema: &JsonValue) -> CandleResult<String> {
        // Use Outlines' built-in JSON schema to regex conversion
        json_schema::regex_from_value(schema, None)
            .map_err(|e| CandleError::Msg(format!("Failed to convert schema to regex: {:?}", e)))
    }
    
    /// Compile a regex pattern into an Index
    fn compile_regex(&self, regex: &str, vocabulary: &Vocabulary) -> CandleResult<Index> {
        // Use Outlines to build the Index from regex
        Index::new(regex, vocabulary)
            .map_err(|e| CandleError::Msg(format!("Failed to build index from regex: {:?}", e)))
    }
    
    /// Calculate a hash for caching
    fn calculate_hash(&self, input: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        input.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Clear the cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        if let Ok(cache) = self.cache.lock() {
            let size = cache.len();
            let capacity = cache.capacity();
            (size, capacity)
        } else {
            (0, 0)
        }
    }
}

impl Default for SchemaProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_schema_processor_creation() {
        let processor = SchemaProcessor::new();
        let (size, _) = processor.cache_stats();
        assert_eq!(size, 0, "Cache should start empty");
    }
    
    #[test]
    fn test_cache_operations() {
        let processor = SchemaProcessor::new();
        
        // Initially empty
        let (size, _) = processor.cache_stats();
        assert_eq!(size, 0);
        
        // After clear (should still be empty)
        processor.clear_cache();
        let (size, _) = processor.cache_stats();
        assert_eq!(size, 0);
    }
    
    #[test]
    fn test_schema_to_regex_basic_types() {
        let processor = SchemaProcessor::new();
        
        // Test string type
        let string_schema = serde_json::json!({
            "type": "string"
        });
        let regex = processor.schema_to_regex(&string_schema).unwrap();
        // Just verify it produces a regex, exact format depends on Outlines
        assert!(!regex.is_empty(), "String schema should produce a regex");
        
        // Test number type
        let number_schema = serde_json::json!({
            "type": "number"
        });
        let regex = processor.schema_to_regex(&number_schema).unwrap();
        assert!(!regex.is_empty(), "Number schema should produce a regex");
        
        // Test boolean type
        let bool_schema = serde_json::json!({
            "type": "boolean"
        });
        let regex = processor.schema_to_regex(&bool_schema).unwrap();
        assert!(regex.contains("true") && regex.contains("false"), "Boolean regex should contain true/false");
    }
    
    #[test]
    fn test_schema_with_pattern() {
        let processor = SchemaProcessor::new();
        
        let schema = serde_json::json!({
            "type": "string",
            "pattern": r"^\d{3}-\d{3}-\d{4}$"
        });
        
        let regex = processor.schema_to_regex(&schema).unwrap();
        // Pattern should be included in the generated regex
        assert!(regex.contains("\\d{3}"), "Should contain digit pattern");
    }
}