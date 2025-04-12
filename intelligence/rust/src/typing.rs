use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum FailureCode {
    LocalEngineChatError,
    InvalidRemoteConfigError,
    InvalidArgumentsError,
    EngineSpecificError,
    AuthenticationError,
    UnavailableError,
    TimeoutError,
    RemoteError,
    EncryptionError,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Failure {
    pub code: FailureCode,
    pub description: String,
}

pub type FIResult<T> = std::result::Result<T, Failure>;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

pub struct Progress {
    pub description: String,
}

pub struct StreamEvent {
    pub chunk: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tool {
    pub func: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    pub func: String,
}

#[derive(Clone, Default)]
pub struct ChatOptions {
    pub model: Option<String>,
    pub temperature: Option<f64>,
    pub max_completion_tokens: Option<u32>,
    pub stream: Option<bool>,
    pub on_stream_event: Option<Arc<dyn Fn(StreamEvent) + Send + Sync>>,
    pub tools: Option<Vec<Tool>>,
    pub force_remote: Option<bool>,
    pub force_local: Option<bool>,
    pub encrypt: Option<bool>,
}

pub type ChatResponseResult = FIResult<Message>;
