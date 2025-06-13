// Copyright 2025 Flower Labs GmbH. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

import nodeCrypto from 'crypto';
import getRandomValues from 'get-random-values';
import { REMOTE_URL } from '../constants';
import {
  ChatResponseResult,
  FailureCode,
  Message,
  Progress,
  Result,
  StreamEvent,
  Tool,
  ToolCall,
} from '../typing';
import { BaseEngine } from './engine';

let crypto: Crypto | typeof nodeCrypto = nodeCrypto;

try {
  crypto = window.crypto;
} catch (_) {
  // fall back to nodeCrypto
}

const KEY_TYPE = 'ECDH';
const CURVE = 'P-384'; // secp384r1 in Web Crypto
const KEY_FORMAT = 'spki';

const AES_KEY_LENGTH = 32;
const GCM_IV_LENGTH = 12;
const BIT_TAG_LENGTH = 8 * 16; // 128 bits or 16 bytes
const SIGN_ALG = 'HMAC';
const CRYPTO_ALG = 'AES-GCM';
const HASH_ALG = 'SHA-256';
const HKDF_INFO = new TextEncoder().encode('ecdh key exchange');

export class RemoteEngine extends BaseEngine {
  private baseUrl: string;
  private apiKey: string;
  private cryptoHandler: CryptographyHandler;

  constructor(apiKey: string) {
    super();
    this.baseUrl = REMOTE_URL;
    this.apiKey = apiKey;
    this.cryptoHandler = new CryptographyHandler(this.baseUrl, this.apiKey);
  }

  async chat(
    messages: Message[],
    model: string,
    temperature?: number,
    maxCompletionTokens?: number,
    stream?: boolean,
    onStreamEvent?: (event: StreamEvent) => void,
    tools?: Tool[],
    encrypt = false
  ): Promise<ChatResponseResult> {
    if (encrypt) {
      const keyRes = await this.cryptoHandler.initializeKeysAndExchange();
      if (!keyRes.ok) {
        return keyRes;
      }
      const encryptRes = await this.cryptoHandler.encryptMessages(messages);
      if (!encryptRes.ok) {
        return encryptRes;
      }
    }
    if (stream) {
      const response = await this.chatStream(
        messages,
        model,
        encrypt,
        temperature,
        maxCompletionTokens,
        onStreamEvent
      );
      if (!response.ok) return response;
      return { ok: true, message: { role: 'assistant', content: response.value } };
    } else {
      const requestData = this.createRequestData(
        messages,
        model,
        temperature,
        maxCompletionTokens,
        false,
        tools,
        encrypt
      );
      const response = await sendRequest(
        requestData,
        '/v1/chat/completions',
        this.baseUrl,
        this.getHeaders()
      );
      if (!response.ok) {
        return response;
      }
      const chatResponse = (await response.value.json()) as ChatCompletionsResponse;
      return await this.extractOutput(chatResponse, encrypt);
    }
  }

  async fetchModel(_model: string, _callback: (progress: Progress) => void): Promise<Result<void>> {
    await Promise.resolve();
    return {
      ok: false,
      failure: {
        code: FailureCode.EngineSpecificError,
        description: 'Cannot fetch model with remote inference engine.',
      },
    };
  }

  async isSupported(_model: string): Promise<Result<void>> {
    await Promise.resolve();
    return {
      ok: true,
      value: undefined,
    };
  }

  private createRequestData(
    messages: Message[],
    model: string,
    temperature?: number,
    maxCompletionTokens?: number,
    stream?: boolean,
    tools?: Tool[],
    encrypt?: boolean
  ): ChatCompletionsRequest {
    return {
      model,
      messages,
      ...(temperature && { temperature }),
      ...(maxCompletionTokens && {
        max_completion_tokens: maxCompletionTokens,
      }),
      ...(stream && { stream }),
      ...(tools && { tools }),
      ...(encrypt && { encrypt, encryption_id: this.cryptoHandler.encryptionId }),
    };
  }

  private getHeaders() {
    return {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${this.apiKey}`,
    };
  }

  async chatStream(
    messages: Message[],
    model: string,
    encrypt: boolean,
    temperature?: number,
    maxCompletionTokens?: number,
    onStreamEvent?: (event: StreamEvent) => void
  ): Promise<Result<string>> {
    const requestData = this.createRequestData(
      messages,
      model,
      temperature,
      maxCompletionTokens,
      true,
      undefined,
      encrypt
    );
    const response = await sendRequest(
      requestData,
      '/v1/chat/completions',
      this.baseUrl,
      this.getHeaders()
    );

    if (!response.ok) return response;

    const reader = response.value.body?.getReader();
    const decoder = new TextDecoder('utf-8');
    let accumulatedResponse = '';

    while (reader) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const dataArray = chunk.split(/(?<=})\s*(?={)/g);

      for (const data of dataArray) {
        try {
          const { object: _, choices } = JSON.parse(data) as {
            object: string;
            choices: StreamChoice[];
          };
          for (const choice of choices) {
            const deltaContent = choice.delta.content;

            if (deltaContent) {
              let content: string;
              if (encrypt) {
                const decryptedResult = await this.cryptoHandler.decryptMessage(deltaContent);
                if (!decryptedResult.ok) {
                  return decryptedResult;
                }
                content = decryptedResult.value;
              } else {
                content = deltaContent;
              }
              onStreamEvent?.({ chunk: content });
              accumulatedResponse += content;
            }
          }
        } catch (error) {
          console.error('Error parsing JSON chunk:', error);
        }
      }
    }
    return { ok: true, value: accumulatedResponse };
  }

  async extractOutput(
    response: ChatCompletionsResponse,
    encrypt: boolean
  ): Promise<ChatResponseResult> {
    const message = response.choices[0].message;
    let content: string;
    if (encrypt) {
      const decryptedResult = await this.cryptoHandler.decryptMessage(message.content ?? '');
      if (!decryptedResult.ok) {
        return decryptedResult;
      }
      content = decryptedResult.value;
    } else {
      content = message.content ?? '';
    }
    const toolCalls = message.tool_calls;

    return {
      ok: true,
      message: {
        role: message.role as Message['role'],
        content: content,
        ...(toolCalls && { tool_calls: toolCalls }),
      },
    };
  }
}

async function sendRequest(
  requestData: ChatCompletionsRequest,
  endpoint: string,
  baseUrl: string,
  headers: Record<string, string>
): Promise<Result<Response>> {
  const response = await fetch(`${baseUrl}${endpoint}`, {
    method: 'POST',
    headers,
    body: JSON.stringify(requestData),
  });

  if (!response.ok) {
    let code = FailureCode.RemoteError;
    switch (response.status) {
      case 401:
      case 403:
      case 407:
        code = FailureCode.AuthenticationError;
        break;
      case 404:
      case 502:
      case 503:
        code = FailureCode.UnavailableError;
        break;
      case 408:
      case 504:
        code = FailureCode.TimeoutError;
        break;

      default:
        break;
    }
    return {
      ok: false,
      failure: { code, description: `${String(response.status)}: ${response.statusText}` },
    };
  }

  return { ok: true, value: response };
}

/** Handles key generation and ECDH shared secret derivation */
export class KeyManager {
  private privateKey: CryptoKey | null = null;
  private publicKey: CryptoKey | null = null;
  private sharedSecretKey: ArrayBuffer | null = null;

  /**
   * Generate a new ECDH key pair.
   */
  async generateKeyPair() {
    const keyPair = await crypto.subtle.generateKey(
      {
        name: KEY_TYPE,
        namedCurve: CURVE,
      },
      true,
      ['deriveKey', 'deriveBits']
    );

    this.privateKey = keyPair.privateKey;
    this.publicKey = keyPair.publicKey;
  }

  /**
   * Export the public key as a Base64 string.
   */
  async exportPublicKey(): Promise<Result<string>> {
    if (!this.publicKey) {
      return {
        ok: false,
        failure: { code: FailureCode.EncryptionError, description: 'Public key not generated.' },
      };
    }
    const exportedKey = await crypto.subtle.exportKey(KEY_FORMAT, this.publicKey);
    return { ok: true, value: btoa(String.fromCharCode(...new Uint8Array(exportedKey))) };
  }

  /**
   * Derive a shared secret using the server's public key.
   */
  async deriveSharedSecret(serverPublicKeyBase64: string): Promise<Result<ArrayBuffer>> {
    if (!this.privateKey) {
      return {
        ok: false,
        failure: {
          code: FailureCode.EncryptionError,
          description: 'Private key is not initialized.',
        },
      };
    }

    const serverPublicKeyBuffer = Uint8Array.from(atob(serverPublicKeyBase64), (char) =>
      char.charCodeAt(0)
    );

    // Import server's public key
    const serverPublicKey = await crypto.subtle.importKey(
      KEY_FORMAT,
      serverPublicKeyBuffer,
      { name: KEY_TYPE, namedCurve: CURVE },
      false,
      []
    );

    // Compute shared secret
    const sharedSecret = await crypto.subtle.deriveBits(
      { name: KEY_TYPE, public: serverPublicKey },
      this.privateKey,
      384
    );

    // Apply HKDF to derive final encryption key
    this.sharedSecretKey = await hkdf(HASH_ALG, sharedSecret, HKDF_INFO, AES_KEY_LENGTH);
    return { ok: true, value: this.sharedSecretKey };
  }

  getSharedSecretKey(): ArrayBuffer | null {
    return this.sharedSecretKey;
  }
}

/** Handles server communication for key exchange */
export class NetworkService {
  private serverUrl: string;
  private apiKey: string;
  private serverPublicKey: string | null = null;
  private serverPublicKeyExpiresAt: number | null = null;
  private clientPublicKeyExpiresAt: number | null = null;

  constructor(serverUrl: string, apiKey: string) {
    this.serverUrl = serverUrl;
    this.apiKey = apiKey;
  }

  private isServerKeyExpired(): boolean {
    if (!this.serverPublicKeyExpiresAt) return true;
    // Conversion from milliseconds to microseconds
    return Date.now() >= this.serverPublicKeyExpiresAt * 1000;
  }

  isClientKeyExpired(): boolean {
    if (!this.clientPublicKeyExpiresAt) return true;
    // Conversion from milliseconds to microseconds
    return Date.now() >= this.clientPublicKeyExpiresAt * 1000;
  }

  async submitClientPublicKey(clientPublicKey: string): Promise<Result<string>> {
    const response = await fetch(`${this.serverUrl}/v1/encryption/public-key`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${this.apiKey}` },
      body: JSON.stringify({ public_key_base64: clientPublicKey }),
    });

    if (!response.ok) {
      return {
        ok: false,
        failure: {
          code: FailureCode.EncryptionError,
          description: `Failed to send public key: ${response.statusText}`,
        },
      };
    }

    const data = (await response.json()) as { expires_at: string; encryption_id: string };
    this.clientPublicKeyExpiresAt = getTimestamp(data.expires_at);
    return { ok: true, value: data.encryption_id };
  }

  async getServerPublicKey(): Promise<Result<string>> {
    if (this.isServerKeyExpired() || !this.serverPublicKey) await this.fetchNewServerPublicKey();
    if (!this.serverPublicKey) {
      return {
        ok: false,
        failure: { code: FailureCode.EncryptionError, description: 'Public key is not set.' },
      };
    }
    return { ok: true, value: this.serverPublicKey };
  }

  private async fetchNewServerPublicKey(): Promise<Result<void>> {
    const response = await fetch(`${this.serverUrl}/v1/encryption/server-public-key`, {
      headers: { Authorization: `Bearer ${this.apiKey}` },
    });

    if (!response.ok) {
      return {
        ok: false,
        failure: {
          code: FailureCode.EncryptionError,
          description: `Failed to fetch server public key: ${response.statusText}`,
        },
      };
    }

    const data = (await response.json()) as { public_key_base64: string; expires_at: string };
    this.serverPublicKey = data.public_key_base64;
    this.serverPublicKeyExpiresAt = getTimestamp(data.expires_at);
    return { ok: true, value: undefined };
  }
}

/** Orchestrates key management, key exchange, and message encryption/decryption */
export class CryptographyHandler {
  private keyManager: KeyManager;
  private networkService: NetworkService;
  private sharedSecretKey: ArrayBuffer | null = null;
  #encryptionId: string | null = null;

  constructor(serverUrl: string, apiKey: string) {
    this.keyManager = new KeyManager();
    this.networkService = new NetworkService(serverUrl, apiKey);
  }

  get encryptionId() {
    return this.#encryptionId;
  }

  async initializeKeysAndExchange(): Promise<Result<void>> {
    if (this.networkService.isClientKeyExpired()) {
      await this.keyManager.generateKeyPair();
      const clientPublicKey = await this.keyManager.exportPublicKey();
      if (!clientPublicKey.ok) {
        return clientPublicKey;
      }
      const encryptionId = await this.networkService.submitClientPublicKey(clientPublicKey.value);
      if (!encryptionId.ok) {
        return encryptionId;
      }
      this.#encryptionId = encryptionId.value;
    }
    const serverPublicKey = await this.networkService.getServerPublicKey();
    if (!serverPublicKey.ok) {
      return serverPublicKey;
    }
    const sharedSecretKey = await this.keyManager.deriveSharedSecret(serverPublicKey.value);
    if (!sharedSecretKey.ok) {
      return sharedSecretKey;
    }
    this.sharedSecretKey = sharedSecretKey.value;
    return { ok: true, value: undefined };
  }

  async encryptMessage(message: string): Promise<Result<string>> {
    if (!this.sharedSecretKey) {
      return {
        ok: false,
        failure: {
          code: FailureCode.EncryptionError,
          description: 'Shared secret is not derived.',
        },
      };
    }

    try {
      const iv = getRandomValues(new Uint8Array(GCM_IV_LENGTH));
      const aesKey = await crypto.subtle.importKey(
        'raw',
        this.sharedSecretKey,
        { name: CRYPTO_ALG },
        false,
        ['encrypt']
      );

      const encodedMessage = new TextEncoder().encode(message);
      const encryptedData = await crypto.subtle.encrypt(
        { name: CRYPTO_ALG, iv, tagLength: BIT_TAG_LENGTH },
        aesKey,
        encodedMessage
      );

      const encryptedBytes = new Uint8Array(encryptedData);
      const combined = new Uint8Array(iv.length + encryptedBytes.length);
      combined.set(iv, 0);
      combined.set(encryptedBytes, iv.length);

      return { ok: true, value: btoa(String.fromCharCode(...combined)) };
    } catch (error) {
      return {
        ok: false,
        failure: { code: FailureCode.EncryptionError, description: String(error) },
      };
    }
  }

  async encryptMessages(messages: Message[]): Promise<Result<void>> {
    for (const message of messages) {
      const encryptedContent = await this.encryptMessage(message.content);
      if (!encryptedContent.ok) {
        return encryptedContent;
      }
      message.content = encryptedContent.value;
    }
    return { ok: true, value: undefined };
  }

  async decryptMessage(encryptedMessage: string): Promise<Result<string>> {
    if (!this.sharedSecretKey) {
      return {
        ok: false,
        failure: {
          code: FailureCode.EncryptionError,
          description: 'Shared secret is not derived.',
        },
      };
    }

    try {
      const data = Uint8Array.from(atob(encryptedMessage), (char) => char.charCodeAt(0));
      const iv = data.slice(0, GCM_IV_LENGTH);
      const ciphertext = data.slice(GCM_IV_LENGTH);

      const aesKey = await crypto.subtle.importKey(
        'raw',
        this.sharedSecretKey,
        { name: CRYPTO_ALG },
        false,
        ['decrypt']
      );

      const plaintext = await crypto.subtle.decrypt({ name: CRYPTO_ALG, iv }, aesKey, ciphertext);
      return { ok: true, value: new TextDecoder().decode(plaintext) };
    } catch (error: unknown) {
      return {
        ok: false,
        failure: { code: FailureCode.EncryptionError, description: String(error) },
      };
    }
  }
}

async function hkdf(
  hash: string,
  ikm: ArrayBuffer, // Input Keying Material (shared secret)
  info: Uint8Array, // Contextual information (e.g., 'ecdh key exchange')
  length: number
): Promise<ArrayBuffer> {
  const salt = new Uint8Array(AES_KEY_LENGTH); // All-zero salt
  const saltKey = await crypto.subtle.importKey('raw', salt, { name: SIGN_ALG, hash }, false, [
    'sign',
  ]);
  const prk = await crypto.subtle.sign(SIGN_ALG, saltKey, ikm);

  const prkKey = await crypto.subtle.importKey('raw', prk, { name: SIGN_ALG, hash }, false, [
    'sign',
  ]);
  const hashLength = AES_KEY_LENGTH;
  const numBlocks = Math.ceil(length / hashLength);

  let previousBlock = new Uint8Array(0);
  const output = new Uint8Array(length);
  let offset = 0;

  for (let i = 0; i < numBlocks; i++) {
    const input = new Uint8Array([...previousBlock, ...info, i + 1]);
    previousBlock = new Uint8Array(await crypto.subtle.sign(SIGN_ALG, prkKey, input));

    output.set(previousBlock.slice(0, Math.min(hashLength, length - offset)), offset);
    offset += hashLength;
  }

  return output.buffer;
}

/**
 * Convert date formatted as "2025-03-06T13:19:47.353034" to numerical timestamp
 * (in this example, 1741267187353)
 */
export function getTimestamp(dateString: string) {
  return new Date(dateString.slice(0, 23) + 'Z').valueOf();
}

interface ChoiceMessage {
  role: string;
  content?: string;
  tool_calls?: ToolCall[];
}

interface Choice {
  index: number;
  message: ChoiceMessage;
}

interface StreamChoice {
  index: number;
  delta: {
    content: string;
    role: string;
  };
}

interface Usage {
  total_duration: number; // time spent generating the response
  load_duration: number; // time spent in nanoseconds loading the model
  prompt_eval_count: number; // number of tokens in the prompt
  prompt_eval_duration: number; // time spent in nanoseconds evaluating the prompt
  eval_count: number; // number of tokens in the response
  eval_duration: number; // time in nanoseconds spent generating the response
}

interface ChatCompletionsRequest {
  model: string;
  messages: Message[];
  temperature?: number;
  max_completion_tokens?: number;
  tools?: Tool[];
  encrypt?: boolean;
}

interface ChatCompletionsResponse {
  object: string;
  created: number;
  model: string;
  choices: Choice[];
  usage: Usage;
}
