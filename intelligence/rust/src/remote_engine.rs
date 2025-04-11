use async_trait::async_trait;
use std::sync::Arc;

use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Response;
use serde::{Deserialize, Serialize};

use crate::{
    constants::REMOTE_URL,
    engine::Engine,
    typing::{
        ChatResponseResult, FIResult, Failure, FailureCode, Message, Progress, StreamEvent, Tool,
        ToolCall,
    },
};

#[derive(Debug)]
pub struct RemoteEngine {
    base_url: String,
    api_key: String,
    // skipping crypto fields
}

impl RemoteEngine {
    pub fn new(api_key: &str) -> Self {
        Self {
            base_url: REMOTE_URL.to_string(),
            api_key: api_key.to_string(),
        }
    }

    /// Creates the request payload.
    fn create_request_data(
        &self,
        messages: Vec<Message>,
        model: String,
        temperature: Option<f64>,
        max_completion_tokens: Option<u32>,
        stream: Option<bool>,
        tools: Option<Vec<Tool>>,
        // ignore encryption details – simply set encrypt to false.
        _encrypt: Option<bool>,
    ) -> ChatCompletionsRequest {
        ChatCompletionsRequest {
            model,
            messages,
            temperature,
            max_completion_tokens,
            stream,
            tools,
            // We ignore encryption_id and related fields.
            encrypt: Some(false),
        }
    }

    /// Prepares headers.
    fn get_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let bearer = format!("Bearer {}", self.api_key);
        headers.insert(AUTHORIZATION, HeaderValue::from_str(&bearer).unwrap());
        headers
    }

    /// The streaming chat implementation.
    async fn chat_stream(
        &self,
        messages: Vec<Message>,
        model: String,
        _encrypt: bool, // encryption ignored
        temperature: Option<f64>,
        max_completion_tokens: Option<u32>,
        on_stream_event: Option<Arc<dyn Fn(StreamEvent) + Send + Sync>>,
    ) -> FIResult<String> {
        // Create request data with streaming enabled.
        let request_data = self.create_request_data(
            messages,
            model,
            temperature,
            max_completion_tokens,
            Some(true),
            None,
            Some(false),
        );
        let response = send_request(
            &request_data,
            "/v1/chat/completions",
            &self.base_url,
            self.get_headers(),
        )
        .await?;
        let mut accumulated_response = String::new();
        let mut stream = response.bytes_stream();

        // Process each chunk from the stream.
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| Failure {
                code: FailureCode::RemoteError,
                description: e.to_string(),
            })?;
            let text_chunk = String::from_utf8_lossy(&chunk).to_string();

            // Invoke the stream callback, if provided.
            if let Some(callback) = &on_stream_event {
                callback(StreamEvent {
                    chunk: text_chunk.clone(),
                });
            }
            accumulated_response.push_str(&text_chunk);
        }
        Ok(accumulated_response)
    }

    /// Extracts the final output from a JSON response.
    async fn extract_output(
        &self,
        response: ChatCompletionsResponse,
        _encrypt: bool, // ignore encryption
    ) -> ChatResponseResult {
        // Take the first choice and build a Message.
        let choice = response.choices.into_iter().next().ok_or(Failure {
            code: FailureCode::RemoteError,
            description: "No choices in response.".to_string(),
        })?;
        let msg = choice.message;
        Ok(Message {
            role: msg.role,
            content: msg.content.unwrap_or_default(),
            tool_calls: msg.tool_calls,
        })
    }
}

#[async_trait]
impl Engine for RemoteEngine {
    async fn chat(
        &mut self,
        messages: Vec<Message>,
        model: String,
        temperature: Option<f64>,
        max_completion_tokens: Option<u32>,
        stream: Option<bool>,
        on_stream_event: Option<Arc<dyn Fn(StreamEvent) + Send + Sync>>,
        tools: Option<Vec<Tool>>,
        encrypt: Option<bool>,
    ) -> ChatResponseResult {
        // For this implementation we ignore encryption (set false).
        let encrypt_flag = encrypt.unwrap_or(false);
        if stream.unwrap_or(false) {
            let stream_result = self
                .chat_stream(
                    messages,
                    model,
                    encrypt_flag,
                    temperature,
                    max_completion_tokens,
                    on_stream_event,
                )
                .await?;
            Ok(Message {
                role: "assistant".to_string(),
                content: stream_result,
                ..Default::default()
            })
        } else {
            let request_data = self.create_request_data(
                messages,
                model.clone(),
                temperature,
                max_completion_tokens,
                Some(false),
                tools,
                Some(false),
            );
            let response = send_request(
                &request_data,
                "/v1/chat/completions",
                &self.base_url,
                self.get_headers(),
            )
            .await?;

            if !response.status().is_success() {
                let status = response.status();
                let description = format!(
                    "{}: {}",
                    status.as_u16(),
                    status.canonical_reason().unwrap_or("Error")
                );
                let code = match status.as_u16() {
                    401 | 403 | 407 => FailureCode::AuthenticationError,
                    404 | 502 | 503 => FailureCode::UnavailableError,
                    408 | 504 => FailureCode::TimeoutError,
                    _ => FailureCode::RemoteError,
                };
                return Err(Failure { code, description });
            }

            let chat_response: ChatCompletionsResponse =
                response.json().await.map_err(|e| Failure {
                    code: FailureCode::RemoteError,
                    description: e.to_string(),
                })?;
            self.extract_output(chat_response, encrypt_flag).await
        }
    }

    async fn fetch_model(
        &mut self,
        _model: String,
        _callback: Arc<dyn Fn(Progress) + Send + Sync>,
    ) -> FIResult<()> {
        Err(Failure {
            code: FailureCode::EngineSpecificError,
            description: "Cannot fetch model with remote inference engine.".to_string(),
        })
    }

    async fn is_supported(&self, _model: String) -> FIResult<()> {
        Ok(())
    }
}

async fn send_request(
    request_data: &ChatCompletionsRequest,
    endpoint: &str,
    base_url: &str,
    headers: HeaderMap,
) -> FIResult<Response> {
    let client = reqwest::Client::new();
    let url = format!("{}{}", base_url, endpoint);

    let response = client
        .post(&url)
        .headers(headers)
        .json(request_data)
        .send()
        .await
        .map_err(|e| Failure {
            code: FailureCode::RemoteError,
            description: e.to_string(),
        })?;

    if !response.status().is_success() {
        let status = response.status();
        let description = format!(
            "{}: {}",
            status.as_u16(),
            status.canonical_reason().unwrap_or("Error")
        );
        let code = match status.as_u16() {
            401 | 403 | 407 => FailureCode::AuthenticationError,
            404 | 502 | 503 => FailureCode::UnavailableError,
            408 | 504 => FailureCode::TimeoutError,
            _ => FailureCode::RemoteError,
        };
        Err(Failure { code, description })
    } else {
        Ok(response)
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionsRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encrypt: Option<bool>,
    // Ignoring encryption_id and other crypto-related fields.
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsResponse {
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct Choice {
    index: u32,
    message: ChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct ChoiceMessage {
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}
