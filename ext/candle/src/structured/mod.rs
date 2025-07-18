/// Structured generation support using Outlines
/// 
/// This module provides functionality to constrain language model generation
/// to follow specific patterns, such as JSON schemas or regular expressions.

pub mod vocabulary_adapter;
pub mod schema_processor;

pub use vocabulary_adapter::VocabularyAdapter;
pub use schema_processor::SchemaProcessor;

// Re-export commonly used types from outlines-core
pub use outlines_core::prelude::Index;
pub use outlines_core::vocabulary::Vocabulary;

#[cfg(test)]
mod vocabulary_adapter_simple_test;

#[cfg(test)]
mod integration_test;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_module_imports() {
        // Ensure all exports are available
        let _ = VocabularyAdapter;
    }
}