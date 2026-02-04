//! Unified error handling module
//! Provides error type definitions for all modules in the langchain-ai-rust project.
//! Uses the `thiserror` library to provide type-safe and easy-to-understand error types.

pub use crate::chain::ChainError;
pub use crate::language_models::LLMError;
pub use crate::retrievers::RetrieverError;
pub use crate::tools::ToolError;
pub use crate::vectorstore::VectorStoreError;

// Re-export agent-related errors for convenience
pub use crate::agent::AgentError;
pub use crate::rag::RAGError;
// Note: MultiAgentError is re-exported through agent module

// Re-export error utilities
pub mod utils;
pub use utils::{error_context, error_info, ErrorCode, ErrorContext};

/// Unified error enum that combines error types from all submodules.
///
/// This enum serves as the top-level error type for the entire project,
/// allowing errors from all submodules to propagate upward.
///
/// # Usage Example
///
/// ```rust,ignore
/// use langchain_ai_rust::error::LangChainError;
///
/// async fn example() -> Result<(), LangChainError> {
///     // Errors from all submodules can be automatically converted
///     some_agent_operation().await?;
///     some_rag_operation().await?;
///     Ok(())
/// }
/// ```
#[derive(thiserror::Error, Debug)]
pub enum LangChainError {
    #[error("LLM error: {0}")]
    LLMError(#[from] LLMError),

    #[error("Chain error: {0}")]
    ChainError(#[from] ChainError),

    #[error("Vector store error: {0}")]
    VectorStoreError(#[from] VectorStoreError),

    #[error("Retriever error: {0}")]
    RetrieverError(#[from] RetrieverError),

    #[error("Tool error: {0}")]
    ToolError(#[from] ToolError),

    #[error("Agent error: {0}")]
    AgentError(#[from] crate::agent::AgentError),

    #[error("RAG error: {0}")]
    RAGError(#[from] crate::rag::RAGError),

    #[error("Multi-agent error: {0}")]
    MultiAgentError(#[from] crate::agent::MultiAgentError),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

// Convenience type alias
pub type Result<T> = std::result::Result<T, LangChainError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_langchain_error_creation() {
        let chain_error = ChainError::OtherError("test".to_string());
        let langchain_error: LangChainError = chain_error.into();

        match langchain_error {
            LangChainError::ChainError(_) => {}
            _ => panic!("Expected ChainError variant"),
        }
    }

    #[test]
    fn test_tool_error_creation() {
        let tool_error = ToolError::ExecutionError("test execution".to_string());
        let langchain_error: LangChainError = tool_error.into();

        match langchain_error {
            LangChainError::ToolError(_) => {}
            _ => panic!("Expected ToolError variant"),
        }
    }

    #[test]
    fn test_vectorstore_error_creation() {
        let vectorstore_error = VectorStoreError::DeleteNotSupported;
        let langchain_error: LangChainError = vectorstore_error.into();

        match langchain_error {
            LangChainError::VectorStoreError(_) => {}
            _ => panic!("Expected VectorStoreError variant"),
        }
    }

    #[test]
    fn test_retriever_error_creation() {
        let retriever_error = RetrieverError::WikipediaError("test wikipedia".to_string());
        let langchain_error: LangChainError = retriever_error.into();

        match langchain_error {
            LangChainError::RetrieverError(_) => {}
            _ => panic!("Expected RetrieverError variant"),
        }
    }

    #[test]
    fn test_agent_error_creation() {
        let agent_error = crate::agent::AgentError::OtherError("test agent".to_string());
        let langchain_error: LangChainError = agent_error.into();

        match langchain_error {
            LangChainError::AgentError(_) => {}
            _ => panic!("Expected AgentError variant"),
        }
    }

    #[test]
    fn test_rag_error_creation() {
        let rag_error = crate::rag::RAGError::InvalidConfiguration("test config".to_string());
        let langchain_error: LangChainError = rag_error.into();

        match langchain_error {
            LangChainError::RAGError(_) => {}
            _ => panic!("Expected RAGError variant"),
        }
    }

    #[test]
    fn test_multi_agent_error_creation() {
        let multi_agent_error =
            crate::agent::multi_agent::MultiAgentError::AgentNotFound("test_agent".to_string());
        let langchain_error: LangChainError = multi_agent_error.into();

        match langchain_error {
            LangChainError::MultiAgentError(_) => {}
            _ => panic!("Expected MultiAgentError variant"),
        }
    }
}
