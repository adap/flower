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
        messages: Vec<typing::Message>,
        model: String,
        temperature: Option<f64>,
        max_completion_tokens: Option<u32>,
        stream: Option<bool>,
        on_stream_event: Option<Arc<dyn Fn(typing::StreamEvent) + Send + Sync>>,
        tools: Option<Vec<typing::Tool>>,
        encrypt: Option<bool>,
    ) -> typing::ChatResponseResult {
        Ok(typing::Message {
            role: "assistant".into(),
            content: format!("SimpleEngine response using model '{}'", model),
            tool_calls: None,
        })
    }

    async fn fetch_model(
        &mut self,
        model: String,
        callback: Arc<dyn Fn(typing::Progress) + Send + Sync>,
    ) -> typing::FIResult<()> {
        callback(typing::Progress {
            description: "test".into(),
        });
        Ok(())
    }

    async fn is_supported(&self, model: String) -> typing::FIResult<()> {
        // For this dummy engine, assume it supports models containing "transformers".
        if model.contains("transformers") {
            Ok(())
        } else {
            Err(typing::Failure {
                code: typing::FailureCode::LocalEngineChatError,
                description: "Model not supported by SimpleEngine".into(),
            })
        }
    }
}
