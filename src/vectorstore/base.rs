//! VectorStore Base Abstraction Layer
//!
//! Provides common functionality and helper methods for VectorStore implementations.

use std::sync::Arc;

use crate::embedding::embedder_trait::Embedder;
use crate::schemas::Document;
use crate::vectorstore::{VecStoreOptions, VectorStore, VectorStoreError};

/// VectorStore Base Configuration
///
/// Contains shared configuration items for all VectorStore implementations.
#[derive(Clone)]
pub struct VectorStoreBaseConfig {
    /// Embedder used to generate vectors
    pub embedder: Arc<dyn Embedder>,
    /// Collection/Table name
    pub collection_name: String,
    /// Vector dimensions (if known)
    pub vector_dimensions: Option<usize>,
}

impl std::fmt::Debug for VectorStoreBaseConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorStoreBaseConfig")
            .field("embedder", &"<dyn Embedder>")
            .field("collection_name", &self.collection_name)
            .field("vector_dimensions", &self.vector_dimensions)
            .finish()
    }
}

impl VectorStoreBaseConfig {
    /// Create a new configuration
    pub fn new(embedder: Arc<dyn Embedder>, collection_name: String) -> Self {
        Self {
            embedder,
            collection_name,
            vector_dimensions: None,
        }
    }

    /// Set the vector dimensions
    pub fn with_vector_dimensions(mut self, dimensions: usize) -> Self {
        self.vector_dimensions = Some(dimensions);
        self
    }

    /// Get or calculate vector dimensions
    pub async fn get_vector_dimensions(&self) -> Result<usize, VectorStoreError> {
        if let Some(dims) = self.vector_dimensions {
            Ok(dims)
        } else {
            // Get dimensions by embedding a test text
            let test_embedding =
                self.embedder.embed_query("test").await.map_err(|e| {
                    VectorStoreError::InternalError(format!("Embedding error: {}", e))
                })?;
            Ok(test_embedding.len())
        }
    }
}

/// VectorStore Helper Functions
pub struct VectorStoreHelpers;

impl VectorStoreHelpers {
    /// Extract text content from documents
    pub fn extract_texts(docs: &[Document]) -> Vec<String> {
        docs.iter().map(|d| d.page_content.clone()).collect()
    }

    /// Validate that the number of documents and vectors match
    pub fn validate_documents_vectors(
        docs: &[Document],
        vectors: &[Vec<f64>],
    ) -> Result<(), VectorStoreError> {
        if docs.len() != vectors.len() {
            return Err(VectorStoreError::InternalError(format!(
                "Number of documents ({}) and vectors ({}) do not match",
                docs.len(),
                vectors.len()
            )));
        }
        Ok(())
    }

    /// Get embedder from options or use default
    pub fn get_embedder<F>(
        opt: &VecStoreOptions<F>,
        default: &Arc<dyn Embedder>,
    ) -> Arc<dyn Embedder> {
        opt.embedder.as_ref().unwrap_or(default).clone()
    }

    /// Apply score threshold filtering
    pub fn apply_score_threshold(mut docs: Vec<Document>, threshold: Option<f32>) -> Vec<Document> {
        if let Some(threshold) = threshold {
            docs.retain(|doc| doc.score >= threshold as f64);
        }
        docs
    }

    /// Sort documents by score (descending)
    pub fn sort_by_score(mut docs: Vec<Document>) -> Vec<Document> {
        docs.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        docs
    }
}

/// VectorStore Initialization Trait
///
/// Provides a unified interface for VectorStore implementations that require initialization.
#[async_trait::async_trait]
pub trait VectorStoreInitializable: VectorStore {
    /// Initialize VectorStore (create tables, collections, etc.)
    async fn initialize(&self) -> Result<(), VectorStoreError>;
}

/// VectorStore Batch Operations Trait
///
/// Provides an optimized interface for VectorStore that support batch operations.
#[async_trait::async_trait]
pub trait VectorStoreBatch: VectorStore {
    /// Batch add documents (may be more efficient than adding one by one)
    async fn add_documents_batch(
        &self,
        docs: &[Document],
        batch_size: usize,
        opt: &Self::Options,
    ) -> Result<Vec<String>, VectorStoreError> {
        // Default implementation: call add_documents in batches
        let mut all_ids = Vec::new();
        for chunk in docs.chunks(batch_size) {
            let ids = self.add_documents(chunk, opt).await?;
            all_ids.extend(ids);
        }
        Ok(all_ids)
    }

    /// Batch delete documents
    async fn delete_batch(
        &self,
        ids: &[String],
        batch_size: usize,
        opt: &Self::Options,
    ) -> Result<(), VectorStoreError> {
        // Default implementation: call delete in batches
        for chunk in ids.chunks(batch_size) {
            self.delete(chunk, opt).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_texts() {
        let docs = vec![Document::new("text1"), Document::new("text2")];
        let texts = VectorStoreHelpers::extract_texts(&docs);
        assert_eq!(texts, vec!["text1", "text2"]);
    }

    #[test]
    fn test_validate_documents_vectors() {
        let docs = vec![Document::new("text1"), Document::new("text2")];
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!(VectorStoreHelpers::validate_documents_vectors(&docs, &vectors).is_ok());
    }

    #[test]
    fn test_validate_documents_vectors_mismatch() {
        let docs = vec![Document::new("text1")];
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!(VectorStoreHelpers::validate_documents_vectors(&docs, &vectors).is_err());
    }

    #[test]
    fn test_apply_score_threshold() {
        let mut docs = vec![
            Document {
                page_content: "text1".to_string(),
                metadata: Default::default(),
                score: 0.8,
            },
            Document {
                page_content: "text2".to_string(),
                metadata: Default::default(),
                score: 0.3,
            },
        ];
        let filtered = VectorStoreHelpers::apply_score_threshold(docs, Some(0.5));
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].score, 0.8);
    }

    #[test]
    fn test_sort_by_score() {
        let docs = vec![
            Document {
                page_content: "text1".to_string(),
                metadata: Default::default(),
                score: 0.3,
            },
            Document {
                page_content: "text2".to_string(),
                metadata: Default::default(),
                score: 0.8,
            },
        ];
        let sorted = VectorStoreHelpers::sort_by_score(docs);
        assert_eq!(sorted[0].score, 0.8);
        assert_eq!(sorted[1].score, 0.3);
    }
}
