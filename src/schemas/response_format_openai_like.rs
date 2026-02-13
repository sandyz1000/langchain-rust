use crate::schemas::convert::OpenAIFromLangchain;

#[derive(Clone, Debug)]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        description: Option<String>,
        name: String,
        schema: Option<serde_json::Value>,
        strict: Option<bool>,
    },
}

impl OpenAIFromLangchain<ResponseFormat> for async_openai::types::chat::ResponseFormat {
    fn from_langchain(langchain: ResponseFormat) -> Self {
        match langchain {
            ResponseFormat::Text => async_openai::types::chat::ResponseFormat::Text,
            ResponseFormat::JsonObject => async_openai::types::chat::ResponseFormat::JsonObject,
            ResponseFormat::JsonSchema {
                name,
                description,
                schema,
                strict,
            } => async_openai::types::chat::ResponseFormat::JsonSchema {
                json_schema: async_openai::types::chat::ResponseFormatJsonSchema {
                    name,
                    description,
                    schema,
                    strict,
                },
            },
        }
    }
}
