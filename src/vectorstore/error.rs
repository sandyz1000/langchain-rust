use thiserror::Error;

#[derive(Error, Debug)]
pub enum VectorStoreError {
    // ============ Base Operation Errors ============
    #[error("This vector store does not support delete")]
    DeleteNotSupported,

    #[error("Connection failed: {0}")]
    ConnectionError(String),

    #[error("Query failed: {0}")]
    QueryError(String),

    #[error("Document not found: {0}")]
    NotFoundError(String),

    #[error("Index error: {0}")]
    IndexError(String),

    // ============ Authentication and Permission Errors ============
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),

    #[error("Permission denied: {0}")]
    PermissionError(String),

    // ============ Rate Limiting and Timeout Errors ============
    #[error("Rate limit exceeded")]
    RateLimitError,

    #[error("Timeout: {0}")]
    TimeoutError(String),

    // ============ Data Conversion Errors ============
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    #[error("Embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    // ============ Collection/Index Management Errors ============
    #[error("Collection not found: {0}")]
    CollectionNotFound(String),

    #[error("Collection already exists: {0}")]
    CollectionAlreadyExists(String),

    #[error("Index not found: {0}")]
    IndexNotFound(String),

    // ============ Parameter Validation Errors ============
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Missing required parameter: {0}")]
    MissingParameter(String),

    // ============ Internal Errors ============
    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl From<String> for VectorStoreError {
    fn from(s: String) -> Self {
        VectorStoreError::Unknown(s)
    }
}

impl From<&str> for VectorStoreError {
    fn from(s: &str) -> Self {
        VectorStoreError::Unknown(s.to_string())
    }
}

// Generic conversion for embedding errors
impl From<crate::embedding::EmbedderError> for VectorStoreError {
    fn from(e: crate::embedding::EmbedderError) -> Self {
        VectorStoreError::InternalError(format!("Embedding error: {}", e))
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for VectorStoreError {
    fn from(e: Box<dyn std::error::Error + Send + Sync>) -> Self {
        VectorStoreError::Unknown(e.to_string())
    }
}
