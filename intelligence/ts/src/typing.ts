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

/**
 * Represents a message in a chat session.
 */
export interface Message {
  /**
   * The role of the sender (e.g., "user", "system", "assistant").
   */
  role: string;

  /**
   * The content of the message.
   */
  content: string;

  /**
   * An optional list of calls to specific tools
   */
  toolCalls?: ToolCall[];
}

/**
 * Represents a call to a specific tool with its name and arguments.
 */
export type ToolCall = Record<
  string,
  {
    /**
     * The name of the tool being called.
     */
    name: string;

    /**
     * The arguments passed to the tool as key-value pairs.
     */
    arguments: Record<string, string>;
  }
>;

/**
 * Represents a property of a tool's function parameter.
 */
export interface ToolParameterProperty {
  /**
   * The data type of the property (e.g., "string", "number").
   */
  type: string;

  /**
   * A description of the property.
   */
  description: string;

  /**
   * An optional list of allowed values for the property.
   */
  enum?: string[];
}

/**
 * Represents the parameters required for a tool's function.
 */
export interface ToolFunctionParameters {
  /**
   * The data type of the parameters (e.g., "object").
   */
  type: string;

  /**
   * A record defining the properties of each parameter.
   */
  properties: Record<string, ToolParameterProperty>;

  /**
   * A list of parameter names that are required.
   */
  required: string[];
}

/**
 * Represents the function provided by a tool.
 */
export interface ToolFunction {
  /**
   * The name of the function provided by the tool.
   */
  name: string;

  /**
   * A brief description of what the function does.
   */
  description: string;

  /**
   * The parameters required for invoking the function.
   */
  parameters: ToolFunctionParameters;
}

/**
 * Represents a tool with details about its type, function, and parameters.
 */
export interface Tool {
  /**
   * The type of the tool (e.g., "function" or "plugin").
   */
  type: string;

  /**
   * Details about the function provided by the tool.
   */
  function: ToolFunction;
}

/**
 * Represents a single event in a streaming response.
 */
export interface StreamEvent {
  /**
   * The chunk of text data received in the stream event.
   */
  chunk: string;
}

/**
 * Enum representing failure codes for different error scenarios.
 */
export enum FailureCode {
  /**
   * Indicates a local error (e.g., client-side issues).
   */
  LocalError = 100,

  /**
   * Indicates a chat error coming from a local engine.
   */
  LocalEngineChatError,

  /**
   * Indicates a fetch error coming from a local engine.
   */
  LocalEngineFetchError,

  /**
   * Indicates an missing provider for a local model.
   */
  NoLocalProviderError,

  /**
   * Indicates that a model requires more RAM than currently available to be loaded.
   */
  InsufficientRAMError,

  /**
   * Indicates a remote error (e.g., server-side issues).
   */
  RemoteError = 200,

  /**
   * Indicates an authentication error (e.g., HTTP 401, 403, 407).
   */
  AuthenticationError,

  /**
   * Indicates that the service is unavailable (e.g., HTTP 404, 502, 503).
   */
  UnavailableError,

  /**
   * Indicates a timeout error (e.g., HTTP 408, 504).
   */
  TimeoutError,

  /**
   * Indicates a connection error (e.g., network issues).
   */
  ConnectionError,

  /**
   * Indicates an engine-specific error.
   */
  EngineSpecificError = 300,

  /**
   * Indicates that a model is not supported by a given engine.
   */
  UnsupportedModelError,

  /**
   * Indicates a error related to the encryption protocol for remote inference.
   */
  EncryptionError,

  /**
   * Indicates an error caused by a misconfigured state.
   */
  ConfigError = 400,

  /**
   * Indicates that invalid arguments were provided.
   */
  InvalidArgumentsError,

  /**
   * Indicates misconfigured config options for remote inference.
   */
  InvalidRemoteConfigError,

  /**
   * Indicates an unknown model error (e.g., unavailable or invalid model).
   */
  UnknownModelError,

  /**
   * Indicates that the requested feature is not implemented.
   */
  NotImplementedError,
}

/**
 * Represents a failure response with a code and description.
 */
export interface Failure {
  /**
   * The failure code indicating the type of error.
   */
  code: FailureCode;

  /**
   * A description of the failure.
   */
  description: string;
}

export interface ProviderMapping {
  onnx?: string;
  webllm?: string;
}

export interface Progress {
  totalBytes?: number;
  loadedBytes?: number;
  percentage?: number;
  description?: string;
}

/**
 * Options to configure the chat interaction.
 */
export interface ChatOptions {
  /**
   * The model name to use for the chat. Defaults to a predefined model if not specified.
   */
  model?: string;

  /**
   * Controls the creativity of the response. Typically a value between 0 and 1.
   */
  temperature?: number;

  /**
   * Maximum number of tokens to generate in the response.
   */
  maxCompletionTokens?: number;

  /**
   * If true, the response will be streamed.
   */
  stream?: boolean;

  /**
   * Optional callback invoked when a stream event occurs.
   * @param event The {@link StreamEvent} data from the stream.
   */
  onStreamEvent?: (event: StreamEvent) => void;

  /**
   * Optional array of tools available for the chat.
   */
  tools?: Tool[];

  /**
   * If true and remote handoff is enabled, forces the use of a remote engine.
   */
  forceRemote?: boolean;

  /**
   * If true, forces the use of a local engine.
   */
  forceLocal?: boolean;

  /**
   * If true, enables end-to-end encryption for processing the request.
   */
  encrypt?: boolean;
}

export type Result<T> = { ok: true; value: T } | { ok: false; failure: Failure };

/**
 * Represents the result of a chat operation.
 *
 * This union type encapsulates both successful and failed chat outcomes.
 *
 * - **Success:**
 *   When the operation is successful, the result is an object with:
 *   - `ok: true` indicating success.
 *   - `message: {@link Message}` containing the chat response.
 *
 * - **Failure:**
 *   When the operation fails, the result is an object with:
 *   - `ok: false` indicating failure.
 *   - `failure: {@link Failure}` providing details about the error.
 */
export type ChatResponseResult = { ok: true; message: Message } | { ok: false; failure: Failure };
