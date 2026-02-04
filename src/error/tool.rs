use thiserror::Error;

/// Tool-related error types
#[derive(Error, Debug)]
pub enum ToolError {
    // ============ Execution Errors ============
    #[error("Execution failed: {0}")]
    ExecutionError(String),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    // ============ Input Validation Errors ============
    #[error("Invalid input: {0}")]
    InvalidInputError(String),

    #[error("Input parsing failed: {0}")]
    ParsingError(String),

    #[error("Missing required input: {0}")]
    MissingInput(String),

    // ============ Resource Errors ============
    #[error("Timeout: {0}")]
    TimeoutError(String),

    #[error("Rate limit exceeded")]
    RateLimitError,

    #[error("Resource not found: {0}")]
    ResourceNotFound(String),

    // ============ Permission and Security Errors ============
    #[error("Permission denied: {0}")]
    PermissionError(String),

    #[error("Security error: {0}")]
    SecurityError(String),

    // ============ Configuration Errors ============
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Missing configuration: {0}")]
    MissingConfiguration(String),

    // ============ External Service Errors ============
    #[error("External service error: {0}")]
    ExternalServiceError(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    // ============ Internal Errors ============
    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl From<String> for ToolError {
    fn from(s: String) -> Self {
        ToolError::Unknown(s)
    }
}
