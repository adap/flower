mod constants;
mod engine;
mod remote_engine;

use constants::DEFAULT_MODEL;
use engine::{Engine, SimpleEngine};
use once_cell::sync::Lazy;
use remote_engine::RemoteEngine;
use std::sync::Arc;
use tokio::sync::Mutex;
use typing::{ChatOptions, ChatResponseResult, FIResult, Failure, FailureCode, Message, Progress};

pub struct FlowerIntelligence {
    remote_engine: Option<RemoteEngine>,
    available_local_engines: Vec<Box<dyn Engine>>,
    remote_handoff: bool,
    api_key: Option<String>,
}
pub mod typing;

impl FlowerIntelligence {
    pub fn new() -> Self {
        let available_local_engines: Vec<Box<dyn Engine>> = vec![Box::new(SimpleEngine)];
        FlowerIntelligence {
            remote_engine: None,
            available_local_engines,
            remote_handoff: false,
            api_key: None,
        }
    }

    pub fn instance() -> &'static Mutex<FlowerIntelligence> {
        static INSTANCE: Lazy<Mutex<FlowerIntelligence>> =
            Lazy::new(|| Mutex::new(FlowerIntelligence::new()));
        &INSTANCE
    }

    pub fn set_remote_handoff(&mut self, value: bool) {
        self.remote_handoff = value;
    }

    pub fn remote_handoff(&self) -> bool {
        self.remote_handoff
    }

    pub fn set_api_key(&mut self, key: String) {
        self.api_key = Some(key);
    }

    pub async fn fetch_model<F>(&mut self, model: &str, callback: F) -> FIResult<()>
    where
        F: Fn(Progress) + Send + Sync + 'static,
    {
        let engine = self.get_engine(model.to_string(), false, false).await?;
        engine
            .fetch_model(model.to_string(), Arc::new(callback))
            .await
    }

    pub async fn chat(&mut self, input: &str, options: Option<ChatOptions>) -> ChatResponseResult {
        let opts = options.unwrap_or_default();
        let messages = vec![Message {
            role: "user".to_string(),
            content: input.to_string(),
            tool_calls: None,
        }];
        self.internal_chat(messages, opts).await
    }

    pub async fn chat_with_messages(
        &mut self,
        messages: Vec<Message>,
        options: Option<ChatOptions>,
    ) -> ChatResponseResult {
        let opts = options.unwrap_or_default();
        self.internal_chat(messages, opts).await
    }

    async fn internal_chat(
        &mut self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> ChatResponseResult {
        let model = options.model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
        let force_remote = options.force_remote.unwrap_or(false);
        let force_local = options.force_local.unwrap_or(false);
        let engine = self
            .get_engine(model.clone(), force_remote, force_local)
            .await?;
        engine
            .chat(
                messages,
                model.clone(),
                options.temperature,
                options.max_completion_tokens,
                options.stream,
                options.on_stream_event,
                options.tools,
                options.encrypt,
            )
            .await
    }

    async fn get_engine(
        &mut self,
        model: String,
        force_remote: bool,
        force_local: bool,
    ) -> FIResult<&mut dyn Engine> {
        self.validate_args(force_remote, force_local)?;
        if force_remote {
            return self.get_or_create_remote_engine(None).await;
        }
        match self.choose_local_engine_index(model).await {
            Ok(index) => {
                // Create a fresh mutable borrow to return the engine.
                Ok(self.available_local_engines[index].as_mut())
            }
            Err(local_failure) => self.get_or_create_remote_engine(Some(local_failure)).await,
        }
    }

    async fn get_or_create_remote_engine(
        &mut self,
        local_failure: Option<Failure>,
    ) -> FIResult<&mut dyn Engine> {
        if let Some(failure) = local_failure {
            if !self.remote_handoff || self.api_key.is_none() {
                let mut description = failure.description;
                if self.remote_handoff {
                    description.push_str(
                        "\nAdditionally, a valid API key for Remote Handoff was not provided.",
                    );
                } else {
                    description.push_str("\nAdditionally, Remote Handoff was not enabled.");
                }
                return Err(Failure {
                    code: failure.code,
                    description,
                });
            }
        }
        if !self.remote_handoff {
            return Err(Failure {
                description: "To use remote inference, remote handoff must be allowed.".to_string(),
                code: FailureCode::InvalidRemoteConfigError,
            });
        }
        if self.api_key.is_none() {
            return Err(Failure {
                description: "To use remote inference, a valid API key must be set.".to_string(),
                code: FailureCode::InvalidRemoteConfigError,
            });
        }
        if self.remote_engine.is_none() {
            self.remote_engine = Some(RemoteEngine::new(self.api_key.as_ref().unwrap().as_str()));
        }
        Ok(self.remote_engine.as_mut().unwrap())
    }

    async fn choose_local_engine_index(&mut self, model: String) -> FIResult<usize> {
        let mut failures = Vec::new();
        for (i, engine) in self.available_local_engines.iter_mut().enumerate() {
            match engine.is_supported(model.to_string()).await {
                Ok(()) => return Ok(i),
                Err(e) => failures.push(e),
            }
        }
        if failures.is_empty() {
            Err(Failure {
                code: FailureCode::LocalEngineChatError,
                description: "No available local engines.".into(),
            })
        } else {
            let highest_failure = failures.into_iter().max().unwrap();
            Err(highest_failure)
        }
    }

    fn validate_args(&self, force_remote: bool, force_local: bool) -> FIResult<()> {
        if force_remote && force_local {
            Err(Failure {
                description:
                    "The `forceLocal` and `forceRemote` options cannot be true at the same time."
                        .to_string(),
                code: FailureCode::InvalidArgumentsError,
            })
        } else {
            Ok(())
        }
    }
}
