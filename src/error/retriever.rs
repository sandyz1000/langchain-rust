use thiserror::Error;

use crate::error::VectorStoreError;

/// Retriever-related error types
#[derive(Error, Debug)]
pub enum RetrieverError {
    // ============ Base Retrieval Errors ============
    #[error("Query failed: {0}")]
    QueryError(String),

    #[error("Document processing error: {0}")]
    DocumentProcessingError(String),

    #[error("Vector store error: {0}")]
    VectorStoreError(#[from] VectorStoreError),

    // ============ Configuration Errors ============
    #[error("Retriever configuration error: {0}")]
    ConfigurationError(String),

    #[error("Missing required configuration: {0}")]
    MissingConfiguration(String),

    // ============ External API Errors ============
    #[error("Wikipedia API error: {0}")]
    WikipediaError(String),

    #[error("arXiv API error: {0}")]
    ArxivError(String),

    #[error("Tavily API error: {0}")]
    TavilyError(String),

    #[error("Remote API error: {0}")]
    RemoteAPIError(String),

    // ============ Algorithm Errors ============
    #[error("BM25 indexing error: {0}")]
    BM25Error(String),

    #[error("TF-IDF calculation error: {0}")]
    TFIDFError(String),

    #[error("SVM error: {0}")]
    SVMError(String),

    #[error("Reranker error: {0}")]
    RerankerError(String),

    // ============ Collection and Index Errors ============
    #[error("Index not found: {0}")]
    IndexNotFoundError(String),

    #[error("Collection not found: {0}")]
    CollectionNotFound(String),

    // ============ Rate Limiting and Timeout Errors ============
    #[error("Rate limit exceeded")]
    RateLimitError,

    #[error("Timeout: {0}")]
    TimeoutError(String),

    // ============ Parameter Validation Errors ============
    #[error("Invalid query: {0}")]
    InvalidQuery(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    // ============ Internal Errors ============
    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl From<String> for RetrieverError {
    fn from(s: String) -> Self {
        RetrieverError::Unknown(s)
    }
}

impl From<crate::error::VectorStoreError> for RetrieverError {
    fn from(e: crate::error::VectorStoreError) -> Self {
        RetrieverError::InternalError(format!("Vector store error: {}", e))
    }
}
