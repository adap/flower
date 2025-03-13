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

import Foundation

/// Represents a failure response with a code and description.
public struct Failure: Error {
  /// Enum representing failure codes for different error scenarios.
  public enum FailureCode: Int, Sendable {
    /// Indicates a local error (e.g., client-side issues).
    case localError = 100

    /// Indicates a chat error coming from a local engine.
    case localEngineChatError = 101

    /// Indicates a fetch error coming from a local engine.
    case localEngineFetchError = 102

    /// Indicates a missing provider for a local model.
    case noLocalProviderError = 103

    /// Indicates a remote error (e.g., server-side issues).
    case remoteError = 200

    /// Indicates an authentication error (e.g., HTTP 401, 403, 407).
    case authenticationError = 201

    /// Indicates that the service is unavailable (e.g., HTTP 404, 502, 503).
    case unavailableError = 202

    /// Indicates a timeout error (e.g., HTTP 408, 504).
    case timeoutError = 203

    /// Indicates a connection error (e.g., network issues).
    case connectionError = 204

    /// Indicates an engine-specific error.
    case engineSpecificError = 300

    /// Indicates an error related to the encryption protocol for remote inference.
    case encryptionError = 301

    /// Indicates an error caused by a misconfigured state.
    case configError = 400

    /// Indicates that invalid arguments were provided.
    case invalidArgumentsError = 401

    /// Indicates misconfigured config options for remote inference.
    case invalidRemoteConfigError = 402

    /// Indicates an unknown model error (e.g., unavailable or invalid model).
    case unknownModelError = 403

    /// Indicates that the requested feature is not implemented.
    case notImplementedError = 404
  }
  /// The failure code indicating the type of error.
  public let code: FailureCode
  /// A description of the failure.
  public let message: String
}
