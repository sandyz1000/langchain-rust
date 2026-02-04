//! langchain-ai-rust error handling module
//!
//! Provides unified error type definitions and handling patterns.
//!
//! # Usage Example
//!
//! ```rust
//! use langchain_ai_rust::error::LangChainError;
//!
//! async fn example() -> Result<(), LangChainError> {
//!     // Use the ? operator to propagate errors
//!     some_operation().await?;
//!     Ok(())
//! }
// ! }
// ! ```

pub mod chain;
pub mod llm;
pub mod retriever;
pub mod tool;
pub mod vectorstore;

pub use crate::error::chain::ChainError;
pub use crate::error::llm::LLMError;
pub use crate::error::retriever::RetrieverError;
pub use crate::error::tool::ToolError;
pub use crate::error::vectorstore::VectorStoreError;
pub use LangChainError;
