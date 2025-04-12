use intelligence::{
    typing::{ChatOptions, StreamEvent},
    FlowerIntelligence,
};
use std::io::{self, Write};
use std::{env, sync::Arc};
use tokio::sync::Mutex;

#[tokio::main]
async fn main() {
    let fi_mutex: &'static Mutex<FlowerIntelligence> = FlowerIntelligence::instance();
    let mut fi = fi_mutex.lock().await;

    fi.set_remote_handoff(true);

    let api_key = env::var("FI_API_KEY").ok();
    if let Some(key) = api_key {
        fi.set_api_key(key);
    }

    // let options = ChatOptions {
    //     model: Some("meta/llama3.2-3b/instruct-q4".to_string()),
    //     temperature: Some(0.7),
    //     max_completion_tokens: Some(100),
    //     stream: Some(false),
    //     on_stream_event: None,
    //     tools: None,
    //     force_remote: Some(true),
    //     force_local: Some(false),
    //     encrypt: Some(true),
    // };
    let options = ChatOptions {
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
        encrypt: Some(true),
    };

    let chat_result = fi.chat("Why is the sky blue?", Some(options)).await;
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
