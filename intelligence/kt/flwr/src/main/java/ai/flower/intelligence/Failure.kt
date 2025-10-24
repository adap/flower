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
// ==============================================================================

package ai.flower.intelligence

/** Enum representing failure codes for different error scenarios. */
enum class FailureCode(val code: Int) {
  /** Indicates a local error (e.g., client-side issues). */
  LocalError(100),

  /** Indicates a chat error coming from a local engine. */
  LocalEngineChatError(101),

  /** Indicates a fetch error coming from a local engine. */
  LocalEngineFetchError(102),

  /** Indicates a missing provider for a local model. */
  NoLocalProviderError(103),

  /** Indicates a remote error (e.g., server-side issues). */
  RemoteError(200),

  /** Indicates an authentication error (e.g., HTTP 401, 403, 407). */
  AuthenticationError(201),

  /** Indicates that the service is unavailable (e.g., HTTP 404, 502, 503). */
  UnavailableError(202),

  /** Indicates a timeout error (e.g., HTTP 408, 504). */
  TimeoutError(203),

  /** Indicates a connection error (e.g., network issues). */
  ConnectionError(204),

  /** Indicates an engine-specific error. */
  EngineSpecificError(300),

  /** Indicates an error related to the encryption protocol for remote inference. */
  EncryptionError(301),

  /** Indicates an error caused by a misconfigured state. */
  ConfigError(400),

  /** Indicates that invalid arguments were provided. */
  InvalidArgumentsError(401),

  /** Indicates misconfigured config options for remote inference. */
  InvalidRemoteConfigError(402),

  /** Indicates an unknown model error (e.g., unavailable or invalid model). */
  UnknownModelError(403),

  /** Indicates that the requested feature is not implemented. */
  NotImplementedError(404),
}

/**
 * Represents a failure response with a code and description.
 *
 * @property code The failure code indicating the type of error.
 * @property message A description of the failure.
 */
class Failure(val code: FailureCode, override val message: String) : Exception(message)
