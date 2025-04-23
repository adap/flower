use dotenv::dotenv;
use intelligence::{
    typing::{ChatOptions, StreamEvent},
    FlowerIntelligence,
};
use std::{
    env,
    io::{self, Write},
    sync::Arc,
};
use tokio::sync::Mutex;

#[tokio::main]
async fn main() {
    let fi_mutex: &'static Mutex<FlowerIntelligence> = FlowerIntelligence::instance();
    let mut fi = fi_mutex.lock().await;

    fi.set_remote_handoff(true);

    dotenv().ok();
    let api_key = env::var("FI_API_KEY").ok();
    if let Some(key) = api_key {
        fi.set_api_key(key);
    }

    let chat_result = fi
        .chat(
            "Why is the sky blue?",
            Some(ChatOptions {
                model: Some("meta/llama3.2-3b/instruct-q4".to_string()),
                temperature: Some(0.7),
                max_completion_tokens: Some(1000),
                stream: Some(true),
                on_stream_event: Some(Arc::new(|event: StreamEvent| {
                    // Print the chunk without adding a newline.
                    print!("{}", event.chunk);
                    // Flush stdout to ensure immediate output.
                    io::stdout().flush().unwrap();
                })),
                tools: None,
                force_remote: Some(true),
                force_local: Some(false),
                encrypt: Some(false),
            }),
        )
        .await;

    match chat_result {
        Ok(response) => {
            println!("{}", response.content);
            assert_eq!(response.role, "assistant");
        }
        Err(e) => {
            println!("chat test failed: {}", e.description);
            std::process::exit(1);
        }
    }
}
