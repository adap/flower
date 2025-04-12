use crate::typing;
use async_trait::async_trait;
use std::sync::Arc;

#[async_trait]
pub trait Engine: Send + Sync {
    async fn chat(
        &mut self,
        messages: Vec<typing::Message>,
        model: String,
        temperature: Option<f64>,
        max_completion_tokens: Option<u32>,
        stream: Option<bool>,
        on_stream_event: Option<Arc<dyn Fn(typing::StreamEvent) + Send + Sync>>,
        tools: Option<Vec<typing::Tool>>,
        encrypt: Option<bool>,
    ) -> typing::ChatResponseResult;

    async fn fetch_model(
        &mut self,
        model: String,
        callback: Arc<dyn Fn(typing::Progress) + Send + Sync>,
    ) -> typing::FIResult<()>;

    async fn is_supported(&self, model: String) -> typing::FIResult<()>;
}

pub struct SimpleEngine;

#[async_trait]
impl Engine for SimpleEngine {
    async fn chat(
        &mut self,
        _messages: Vec<typing::Message>,
        model: String,
        _temperature: Option<f64>,
        _max_completion_tokens: Option<u32>,
        _stream: Option<bool>,
        _on_stream_event: Option<Arc<dyn Fn(typing::StreamEvent) + Send + Sync>>,
        _tools: Option<Vec<typing::Tool>>,
        _encrypt: Option<bool>,
    ) -> typing::ChatResponseResult {
        Ok(typing::Message {
            role: "assistant".into(),
            content: format!("SimpleEngine response using model '{}'", model),
            tool_calls: None,
        })
    }

    async fn fetch_model(
        &mut self,
        _model: String,
        callback: Arc<dyn Fn(typing::Progress) + Send + Sync>,
    ) -> typing::FIResult<()> {
        callback(typing::Progress {
            description: "test".into(),
        });
        Ok(())
    }

    async fn is_supported(&self, _model: String) -> typing::FIResult<()> {
        Ok(())
    }
}
