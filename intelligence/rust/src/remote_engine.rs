use crate::{
    constants::REMOTE_URL,
    engine::Engine,
    typing::{
        ChatResponseResult, FIResult, Failure, FailureCode, Message, Progress, StreamEvent, Tool,
        ToolCall,
    },
};
use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::Aes256Gcm;
use async_trait::async_trait;
use base64::{engine::general_purpose, Engine as _};
use chrono::DateTime;
use chrono::Utc;
use fancy_regex::Regex;
use futures_util::StreamExt;
use rand::RngCore;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Response;
use ring::{agreement, hkdf, rand as ring_rand};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use x509_parser::prelude::*;
use yasna::models::ObjectIdentifier;

const AES_KEY_LENGTH: usize = 32; // 32 bytes => 256-bit AES key
const GCM_IV_LENGTH: usize = 12; // 12 bytes for AES-GCM IV
const HKDF_INFO: &[u8] = b"ecdh key exchange";

pub struct RemoteEngine {
    base_url: String,
    api_key: String,
    crypto_handler: CryptographyHandler,
}

impl RemoteEngine {
    pub fn new(api_key: &str) -> Self {
        Self {
            base_url: REMOTE_URL.to_string(),
            api_key: api_key.to_string(),
            crypto_handler: CryptographyHandler::new(REMOTE_URL, api_key),
        }
    }

    fn create_request_data(
        &self,
        messages: Vec<Message>,
        model: String,
        temperature: Option<f64>,
        max_completion_tokens: Option<u32>,
        stream: Option<bool>,
        tools: Option<Vec<Tool>>,
        encrypt: Option<bool>,
    ) -> ChatCompletionsRequest {
        let encryption_id = if encrypt.unwrap_or(false) {
            self.crypto_handler.encryption_id().cloned()
        } else {
            None
        };
        ChatCompletionsRequest {
            model,
            messages,
            temperature,
            max_completion_tokens,
            stream,
            tools,
            encrypt,
            encryption_id,
        }
    }

    fn get_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let bearer = format!("Bearer {}", self.api_key);
        headers.insert(AUTHORIZATION, HeaderValue::from_str(&bearer).unwrap());
        headers
    }

    async fn chat_stream(
        &self,
        messages: Vec<Message>,
        model: String,
        encrypt: bool,
        temperature: Option<f64>,
        max_completion_tokens: Option<u32>,
        on_stream_event: Option<Arc<dyn Fn(StreamEvent) + Send + Sync>>,
    ) -> FIResult<String> {
        let request_data = self.create_request_data(
            messages,
            model,
            temperature,
            max_completion_tokens,
            Some(true),
            None,
            Some(encrypt),
        );
        let response = send_request(
            &request_data,
            "/v1/chat/completions",
            &self.base_url,
            self.get_headers(),
        )
        .await?;

        // Prepare a regex to split JSON objects that are concatenated in the stream.
        let re = Regex::new(r"(?<=})\s*(?={)").unwrap();

        let mut accumulated_response = String::new();
        let mut stream = response.bytes_stream();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| Failure {
                code: FailureCode::RemoteError,
                description: e.to_string(),
            })?;
            // Convert the byte chunk to a String.
            let chunk_text = String::from_utf8_lossy(&chunk).to_string();
            // Split the chunk into potential JSON segments.
            let data_array: Vec<&str> = re
                .split(&chunk_text)
                .map(|res| res.map_err(|e| format!("Regex split error: {:?}", e)))
                .collect::<Result<Vec<_>, String>>()
                .map_err(|s| Failure {
                    code: FailureCode::RemoteError,
                    description: s,
                })?;

            for data in data_array {
                // Try to parse the segment as a StreamResponse.
                let parsed: Result<StreamResponse, _> = serde_json::from_str(data);
                if let Ok(stream_response) = parsed {
                    for choice in stream_response.choices {
                        if let Some(delta_content) = choice.delta.content {
                            // If encryption is enabled, attempt to decrypt the content.
                            let content = if encrypt {
                                match self.crypto_handler.decrypt_message(&delta_content).await {
                                    Ok(decrypted) => decrypted,
                                    Err(err_msg) => {
                                        return Err(Failure {
                                            code: FailureCode::EncryptionError,
                                            description: err_msg,
                                        });
                                    }
                                }
                            } else {
                                delta_content
                            };
                            // Invoke the stream callback if provided.
                            if let Some(callback) = &on_stream_event {
                                callback(StreamEvent {
                                    chunk: content.clone(),
                                });
                            }
                            // Append the resulting chunk to the accumulated response.
                            accumulated_response.push_str(&content);
                        }
                    }
                } else {
                    eprintln!("Error parsing JSON chunk: {}", data);
                }
            }
        }
        Ok(accumulated_response)
    }

    async fn extract_output(
        &self,
        response: ChatCompletionsResponse,
        encrypt: bool,
    ) -> ChatResponseResult {
        // Take the first choice and build a Message.
        let choice = response.choices.into_iter().next().ok_or(Failure {
            code: FailureCode::RemoteError,
            description: "No choices in response.".to_string(),
        })?;
        let msg = choice.message;
        let content = if encrypt {
            self.crypto_handler
                .decrypt_message(msg.content.as_deref().unwrap_or(""))
                .await
                .map_err(|e| Failure {
                    code: FailureCode::EncryptionError,
                    description: e,
                })?
        } else {
            msg.content.unwrap_or_default()
        };
        Ok(Message {
            role: msg.role,
            content,
            tool_calls: msg.tool_calls,
        })
    }
}

#[async_trait]
impl Engine for RemoteEngine {
    async fn chat(
        &mut self,
        mut messages: Vec<Message>,
        model: String,
        temperature: Option<f64>,
        max_completion_tokens: Option<u32>,
        stream: Option<bool>,
        on_stream_event: Option<Arc<dyn Fn(StreamEvent) + Send + Sync>>,
        tools: Option<Vec<Tool>>,
        encrypt: Option<bool>,
    ) -> ChatResponseResult {
        let encrypt_flag = encrypt.unwrap_or(false);
        if encrypt_flag {
            self.crypto_handler
                .initialize_keys_and_exchange()
                .await
                .map_err(|e| Failure {
                    code: FailureCode::EncryptionError,
                    description: e,
                })?;
            self.crypto_handler
                .encrypt_messages(&mut messages)
                .await
                .map_err(|e| Failure {
                    code: FailureCode::EncryptionError,
                    description: e,
                })?;
        }
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
                Some(encrypt_flag),
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

struct AesKeyLen;
impl hkdf::KeyType for AesKeyLen {
    fn len(&self) -> usize {
        AES_KEY_LENGTH
    }
}

pub struct KeyManager {
    private_key: Option<agreement::EphemeralPrivateKey>,
    public_key: Option<Vec<u8>>,
    shared_secret_key: Option<Vec<u8>>,
}

impl KeyManager {
    pub fn new() -> Self {
        Self {
            private_key: None,
            public_key: None,
            shared_secret_key: None,
        }
    }

    pub async fn generate_key_pair(&mut self) -> Result<(), String> {
        let rng = ring_rand::SystemRandom::new();
        let private = agreement::EphemeralPrivateKey::generate(&agreement::ECDH_P384, &rng)
            .map_err(|e| format!("Error generating key pair: {:?}", e))?;
        let public_key = private
            .compute_public_key()
            .map_err(|e| format!("Error computing public key: {:?}", e))?;
        let public_bytes = public_key.as_ref().to_vec();

        self.private_key = Some(private);
        self.public_key = Some(public_bytes);
        Ok(())
    }

    pub async fn export_public_key(&self) -> Result<String, String> {
        if let Some(raw_pub) = &self.public_key {
            // Build a DER-encoded SubjectPublicKeyInfo using yasna.
            let der = yasna::construct_der(|writer| {
                writer.write_sequence(|writer| {
                    // Write the AlgorithmIdentifier sequence.
                    writer.next().write_sequence(|writer| {
                        // Write the ecPublicKey OID: 1.2.840.10045.2.1
                        writer
                            .next()
                            .write_oid(&ObjectIdentifier::from_slice(&[1, 2, 840, 10045, 2, 1]));
                        // Write the named curve OID for secp384r1: 1.3.132.0.34
                        writer
                            .next()
                            .write_oid(&ObjectIdentifier::from_slice(&[1, 3, 132, 0, 34]));
                    });
                    // Write the public key as a BIT STRING using the BitVec.
                    writer.next().write_bitvec_bytes(raw_pub, raw_pub.len() * 8);
                });
            });

            // Base64-encode the DER output.
            Ok(general_purpose::STANDARD.encode(der))
        } else {
            Err("Public key not generated.".to_string())
        }
    }

    pub async fn derive_shared_secret(
        &mut self,
        server_public_key_base64: &str,
    ) -> Result<Vec<u8>, String> {
        let private = self
            .private_key
            .take()
            .ok_or("Private key is not initialized.".to_string())?;

        let server_pub_bytes = general_purpose::STANDARD
            .decode(server_public_key_base64)
            .map_err(|e| format!("Base64 decode error: {:?}", e))?;

        // Determine if the key is already in raw uncompressed format.
        // For P-384, raw keys should be 97 bytes long and start with 0x04.
        let raw_server_pub = if server_pub_bytes.len() == 97 && server_pub_bytes[0] == 0x04 {
            server_pub_bytes
        } else {
            // Otherwise, assume the key is in DER-encoded SPKI format.
            // Parse the DER structure and extract the subjectPublicKey bytes.
            let (_, spki) = SubjectPublicKeyInfo::from_der(&server_pub_bytes)
                .map_err(|_| "Failed to parse SPKI.".to_string())?;
            spki.subject_public_key.data.to_vec()
        };

        let peer_public_key =
            agreement::UnparsedPublicKey::new(&agreement::ECDH_P384, &raw_server_pub);

        let raw_shared_secret =
            agreement::agree_ephemeral(private, &peer_public_key, (), |key_material| {
                Ok(key_material.to_vec())
            })
            .map_err(|_| "Failed to derive shared secret.".to_string())?;

        // Apply HKDF to produce a 32-byte key.
        let salt_bytes = [0u8; AES_KEY_LENGTH]; // all-zero salt
        let salt = hkdf::Salt::new(hkdf::HKDF_SHA256, &salt_bytes);
        let prk = salt.extract(&raw_shared_secret);
        let okm = prk
            .expand(&[HKDF_INFO], AesKeyLen)
            .map_err(|_| "HKDF expand error.".to_string())?;
        let mut derived_key = vec![0u8; AES_KEY_LENGTH];
        okm.fill(&mut derived_key)
            .map_err(|_| "HKDF fill error.".to_string())?;

        self.shared_secret_key = Some(derived_key.clone());
        Ok(derived_key)
    }
}

#[derive(Debug)]
pub struct NetworkService {
    server_url: String,
    api_key: String,
    server_public_key: Option<String>,
    server_public_key_expires_at: Option<u64>,
    client_public_key_expires_at: Option<u64>,
    client: reqwest::Client,
}

impl NetworkService {
    pub fn new(server_url: &str, api_key: &str) -> Self {
        Self {
            server_url: server_url.to_string(),
            api_key: api_key.to_string(),
            server_public_key: None,
            server_public_key_expires_at: None,
            client_public_key_expires_at: None,
            client: reqwest::Client::new(),
        }
    }

    fn is_server_key_expired(&self) -> bool {
        match self.server_public_key_expires_at {
            Some(exp) => {
                let current = Utc::now().timestamp_millis() as u64;
                current >= exp * 1000 // assuming exp is in seconds
            }
            None => true,
        }
    }

    pub fn is_client_key_expired(&self) -> bool {
        match self.client_public_key_expires_at {
            Some(exp) => {
                let current = Utc::now().timestamp_millis() as u64;
                current >= exp * 1000
            }
            None => true,
        }
    }

    pub async fn submit_client_public_key(
        &mut self,
        client_public_key: &str,
    ) -> Result<String, String> {
        let url = format!("{}/v1/encryption/public-key", self.server_url);
        let body = serde_json::json!({ "public_key_base64": client_public_key });
        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Network error: {:?}", e))?;
        if !response.status().is_success() {
            return Err(format!("Failed to send public key: {}", response.status()));
        }
        // Expected JSON: { "expires_at": "<date>", "encryption_id": "<id>" }
        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| format!("JSON error: {:?}", e))?;
        let expires_at = data
            .get("expires_at")
            .and_then(|v| v.as_str())
            .ok_or("Missing expires_at")?;
        let encryption_id = data
            .get("encryption_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing encryption_id")?;
        self.client_public_key_expires_at = Some(get_timestamp(expires_at));
        Ok(encryption_id.to_string())
    }

    pub async fn get_server_public_key(&mut self) -> Result<String, String> {
        if self.is_server_key_expired() || self.server_public_key.is_none() {
            self.fetch_new_server_public_key().await?;
        }
        self.server_public_key
            .clone()
            .ok_or_else(|| "Server public key not set.".to_string())
    }

    async fn fetch_new_server_public_key(&mut self) -> Result<(), String> {
        let url = format!("{}/v1/encryption/server-public-key", self.server_url);
        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .map_err(|e| format!("Network error: {:?}", e))?;
        if !response.status().is_success() {
            return Err(format!(
                "Failed to fetch server public key: {}",
                response.status()
            ));
        }
        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| format!("JSON error: {:?}", e))?;
        let public_key_base64 = data
            .get("public_key_base64")
            .and_then(|v| v.as_str())
            .ok_or("Missing public_key_base64")?;
        let expires_at = data
            .get("expires_at")
            .and_then(|v| v.as_str())
            .ok_or("Missing expires_at")?;
        self.server_public_key = Some(public_key_base64.to_string());
        self.server_public_key_expires_at = Some(get_timestamp(expires_at));
        Ok(())
    }
}

pub struct CryptographyHandler {
    key_manager: KeyManager,
    network_service: NetworkService,
    shared_secret_key: Option<Vec<u8>>,
    encryption_id: Option<String>,
}

impl CryptographyHandler {
    pub fn new(server_url: &str, api_key: &str) -> Self {
        Self {
            key_manager: KeyManager::new(),
            network_service: NetworkService::new(server_url, api_key),
            shared_secret_key: None,
            encryption_id: None,
        }
    }

    pub fn encryption_id(&self) -> Option<&String> {
        self.encryption_id.as_ref()
    }

    pub async fn initialize_keys_and_exchange(&mut self) -> Result<(), String> {
        if self.network_service.is_client_key_expired() {
            self.key_manager.generate_key_pair().await?;
            let client_public_key = self.key_manager.export_public_key().await?;
            let encryption_id = self
                .network_service
                .submit_client_public_key(&client_public_key)
                .await?;
            self.encryption_id = Some(encryption_id);
        }
        let server_public_key = self.network_service.get_server_public_key().await?;
        let shared = self
            .key_manager
            .derive_shared_secret(&server_public_key)
            .await?;
        self.shared_secret_key = Some(shared);
        Ok(())
    }

    pub async fn encrypt_message(&self, message: &str) -> Result<String, String> {
        if self.shared_secret_key.is_none() {
            return Err("Shared secret is not derived.".to_string());
        }
        let key_bytes = self.shared_secret_key.as_ref().unwrap();
        // Create an AES-GCM cipher instance.
        let key = aes_gcm::Key::<Aes256Gcm>::from_slice(key_bytes);
        let cipher = Aes256Gcm::new(key);
        // Generate a random IV of length GCM_IV_LENGTH.
        let mut iv = [0u8; GCM_IV_LENGTH];
        rand::thread_rng().fill_bytes(&mut iv);
        let nonce = aes_gcm::Nonce::from_slice(&iv);
        let plaintext = message.as_bytes();
        let ciphertext = cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| format!("Encryption error: {:?}", e))?;
        // Prepend IV to the ciphertext.
        let mut combined = Vec::new();
        combined.extend_from_slice(&iv);
        combined.extend_from_slice(&ciphertext);
        // Return as Base64 string.
        Ok(general_purpose::STANDARD.encode(&combined))
    }

    pub async fn encrypt_messages(&mut self, messages: &mut [Message]) -> Result<(), String> {
        for msg in messages.iter_mut() {
            let encrypted = self.encrypt_message(&msg.content).await?;
            msg.content = encrypted;
        }
        Ok(())
    }

    pub async fn decrypt_message(&self, encrypted_message: &str) -> Result<String, String> {
        if self.shared_secret_key.is_none() {
            return Err("Shared secret is not derived.".to_string());
        }
        let key_bytes = self.shared_secret_key.as_ref().unwrap();
        let key = aes_gcm::Key::<Aes256Gcm>::from_slice(key_bytes);
        let cipher = Aes256Gcm::new(key);
        let combined = general_purpose::STANDARD
            .decode(encrypted_message)
            .map_err(|e| format!("Base64 decode error: {:?}", e))?;
        if combined.len() < GCM_IV_LENGTH {
            return Err("Invalid encrypted message.".to_string());
        }
        let (iv, ciphertext) = combined.split_at(GCM_IV_LENGTH);
        let nonce = aes_gcm::Nonce::from_slice(iv);
        let plaintext = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| format!("Decryption error: {:?}", e))?;
        String::from_utf8(plaintext).map_err(|e| format!("UTF8 conversion error: {:?}", e))
    }
}

pub fn get_timestamp(date_string: &str) -> u64 {
    let trimmed = if date_string.len() >= 23 {
        &date_string[..23]
    } else {
        date_string
    };
    let iso = format!("{}Z", trimmed);
    match DateTime::parse_from_rfc3339(&iso) {
        Ok(dt) => dt.timestamp_millis() as u64,
        Err(_) => 0,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    encryption_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChatCompletionsResponse {
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
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
#[allow(dead_code)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct StreamDelta {
    content: Option<String>,
    role: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct StreamChoice {
    index: u32,
    delta: StreamDelta,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct StreamResponse {
    object: String,
    choices: Vec<StreamChoice>,
}
