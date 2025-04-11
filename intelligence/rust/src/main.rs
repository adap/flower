use intelligence::FlowerIntelligence;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() {
    let fi_mutex: &'static Mutex<FlowerIntelligence> = FlowerIntelligence::instance();
    let mut fi = fi_mutex.lock().await;

    fi.set_remote_handoff(true);
    fi.set_api_key("API_KEY".to_string());

    let chat_result = fi.chat("Why is the sky blue?", None).await;
    match chat_result {
        Ok(response) => {
            println!("chat test passed. Response: {}", response.content);
            assert_eq!(response.role, "assistant");
        }
        Err(e) => {
            println!("chat test failed: {}", e.description);
            std::process::exit(1);
        }
    }
    println!("All tests in main passed successfully.");
}
