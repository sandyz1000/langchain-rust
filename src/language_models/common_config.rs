//! Common Configuration and Interface for LLM Clients
//!
//! Provides shared configuration patterns and helper functions for all LLM clients.

use crate::language_models::{llm::LLM, options::CallOptions};

/// Common Configuration Trait for LLM Clients
///
/// Provides a unified configuration interface for all LLM clients.
pub trait LLMConfig: Send + Sync {
    /// Get the model name
    fn model(&self) -> &str;

    /// Get the call options
    fn options(&self) -> &CallOptions;

    /// Set the model
    fn set_model(&mut self, model: String);

    /// Set the call options
    fn set_options(&mut self, options: CallOptions);
}

/// Builder Trait for LLM Clients
///
/// Provides a unified builder pattern for all LLM clients.
///
/// Note: This is a marker trait; specific implementations are provided by each LLM client.
/// This trait is primarily used for documentation and type constraints.
pub trait LLMBuilder: Sized {
    /// Create a new client instance
    fn new() -> Self;

    /// Set the model
    fn with_model<S: Into<String>>(self, model: S) -> Self;

    /// Set the call options
    fn with_options(self, options: CallOptions) -> Self;
}

/// Helper Functions for LLM Clients
pub struct LLMHelpers;

impl LLMHelpers {
    /// Validate the model name format
    ///
    /// Check if the model name meets basic format requirements.
    pub fn validate_model_name(model: &str) -> Result<(), String> {
        if model.is_empty() {
            return Err("Model name cannot be empty".to_string());
        }
        if model.len() > 256 {
            return Err("Model name too long (max 256 characters)".to_string());
        }
        Ok(())
    }

    /// Get API Key from Environment Variables
    ///
    /// # Parameters
    /// - `env_var`: Environment variable name
    /// - `default`: Default value if the environment variable does not exist
    ///
    /// # Returns
    /// API key string
    pub fn get_api_key_from_env(env_var: &str, default: &str) -> String {
        std::env::var(env_var).unwrap_or_else(|_| default.to_string())
    }

    /// Merge Call Options
    ///
    /// Merge two CallOptions, where the second option's fields override the first.
    pub fn merge_options(_base: CallOptions, override_opts: CallOptions) -> CallOptions {
        // Note: This needs to be implemented based on the actual structure of CallOptions
        // Currently returns override_opts; actual implementation should merge fields
        override_opts
    }

    /// Create default call options
    pub fn default_options() -> CallOptions {
        CallOptions::default()
    }
}

/// LLM Client Initialization Configuration
///
/// Contains shared initialization parameters for all LLM clients.
#[derive(Debug, Clone)]
pub struct LLMInitConfig {
    /// Model name
    pub model: Option<String>,
    /// API key (if applicable)
    pub api_key: Option<String>,
    /// Base URL (if applicable)
    pub base_url: Option<String>,
    /// Call options
    pub options: Option<CallOptions>,
}

impl LLMInitConfig {
    /// Create a new configuration
    pub fn new() -> Self {
        Self {
            model: None,
            api_key: None,
            base_url: None,
            options: None,
        }
    }

    /// Set the model
    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set API key
    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set base URL
    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set call options
    pub fn with_options(mut self, options: CallOptions) -> Self {
        self.options = Some(options);
        self
    }
}

impl Default for LLMInitConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for Streaming Response Handling
///
/// Provides a unified interface for all LLM clients that support streaming responses.
pub trait StreamingLLM: LLM {
    /// Check if streaming responses are supported
    fn supports_streaming(&self) -> bool {
        true // Most modern LLMs support streaming responses
    }

    /// Get the default configuration for streaming responses
    fn default_streaming_config(&self) -> CallOptions {
        CallOptions::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_model_name() {
        assert!(LLMHelpers::validate_model_name("gpt-4").is_ok());
        assert!(LLMHelpers::validate_model_name("").is_err());
    }

    #[test]
    fn test_llm_init_config() {
        let config = LLMInitConfig::new()
            .with_model("gpt-4")
            .with_api_key("test-key");

        assert_eq!(config.model, Some("gpt-4".to_string()));
        assert_eq!(config.api_key, Some("test-key".to_string()));
    }
}
