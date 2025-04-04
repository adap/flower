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
import MLX

protocol Engine {
  func chat(
    _ messages: [Message],
    model: String?,
    temperature: Float?,
    maxCompletionTokens: Int?,
    stream: Bool,
    onStreamEvent: (@Sendable (StreamEvent) -> Void)?,
    tools: [Tool]?
  ) async throws -> Message

  func fetchModel(model: String, callback: @escaping (Progress) -> Void) async throws
}

extension Engine {
  func chat(
    _ messages: [Message],
    model: String?,
    temperature: Float? = nil,
    maxCompletionTokens: Int? = nil,
    stream: Bool = false,
    onStreamEvent: (@Sendable (StreamEvent) -> Void)? = nil,
    tools: [Tool]? = nil
  ) async throws -> Message {
    throw Failure(
      code: .notImplementedError,
      message: "Chat function is not implemented yet"
    )
  }

  func fetchModel(model: String, callback: @escaping (Progress) -> Void) async throws {
    throw Failure(
      code: .notImplementedError,
      message: "FetchModel function is not implemented yet"
    )
  }

  func fetchModelConfig(model: String, engine: String) async throws -> ModelConfigResponse {
    let requestBody = ModelConfigRequest(
      model: model, engineName: engine, sdk: sdk, sdkVersion: version)
    guard let url = URL(string: "\(remoteUrl)\(fetchModelConfigPath)") else {
      throw Failure(
        code: .connectionError,
        message: URLError(.badURL).localizedDescription
      )
    }
    return try await NetworkService.getElement(on: url)
  }
}

struct ModelConfigRequest: Codable {
  let model: String
  let engineName: String
  let sdk: String
  let sdkVersion: String

  enum CodingKeys: String, CodingKey {
    case model
    case engineName = "engine_name"
    case sdk
    case sdkVersion = "sdk_version"
  }
}

struct ModelConfigResponse: Codable {
  let isSupported: Bool
  let engineModel: String
  let model: String

  enum CodingKeys: String, CodingKey {
    case isSupported = "is_supported"
    case engineModel = "engine_model"
    case model
  }
}
