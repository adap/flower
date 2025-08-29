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
import MLXLLM
import MLXNN

/// Class representing the core intelligence service for Flower Labs.
/// It facilitates chat, generation, and summarization tasks, with the option of using a
/// local or remote engine based on configurations and availability.
public class FlowerIntelligence {
  /// Sets the remote handoff boolean.
  public var remoteHandoff: Bool = false
  /// Set apiKey for FlowerIntelligence.
  public var apiKey = "" {
    didSet {
      remoteEngine.apiKey = apiKey
    }
  }
  private let mlxEngine: Engine
  private var remoteEngine: RemoteEngineProtocol
  private var engine: Engine {
    return remoteHandoff ? remoteEngine : mlxEngine
  }

  private init() {
    self.mlxEngine = MlxEngine()
    self.remoteEngine = RemoteEngine()
  }

  internal init(mlxEngine: Engine, remoteEngine: RemoteEngineProtocol) {
    self.mlxEngine = mlxEngine
    self.remoteEngine = remoteEngine
  }

  /// Singleton instance of FlowerIntelligence.
  @MainActor public static let instance = FlowerIntelligence()

  /// Conducts a chat interaction using a string input.
  ///
  /// This method takes a string as input, which is automatically wrapped as a single message
  /// with the role `"user"`. An optional `ChatOptions` object can be provided to configure
  /// additional parameters like temperature or model.
  ///
  /// Example:
  /// ```swift
  /// let result = await chat("Why is the sky blue?", maybeOptions: ChatOptions(temperature: 0.7))
  /// ```
  ///
  /// - Parameters:
  ///   - input: A `String` representing the user message.
  ///   - maybeOptions: An optional `ChatOptions` object for customization.
  /// - Returns: An `async` `Result<Message, Failure>`. On success, the result contains the message reply;
  ///            on failure, it includes an error code and description.
  public func chat(
    _ input: String,
    maybeOptions: ChatOptions? = nil
  ) async -> Result<Message, Failure> {
    var engine = self.engine
    if let options = maybeOptions {
      if options.forceLocal && options.forceRemote {
        return .failure(
          Failure(
            code: .configError,
            message: "Cannot set both forceRemote and forceLocal to true"
          )
        )
      }
      if options.forceRemote {
        engine = remoteEngine
      } else if options.forceLocal {
        engine = mlxEngine
      }
    }
    do {
      let messages: [Message] = [try Message(role: "user", content: input)]
      let result = try await engine.chat(
        messages,
        model: maybeOptions?.model,
        temperature: maybeOptions?.temperature,
        maxCompletionTokens: maybeOptions?.maxCompletionTokens,
        stream: maybeOptions?.stream ?? false,
        onStreamEvent: maybeOptions?.onStreamEvent,
        tools: maybeOptions?.tools
      )
      return .success(result)
    } catch {
      return .failure(
        error as? Failure ?? Failure(code: .unavailableError, message: error.localizedDescription))
    }
  }

  /// Conducts a chat interaction using an array of messages and options.
  ///
  /// This method allows for multi-message conversations by accepting an array of `Message` objects,
  /// along with `ChatOptions` for configuration.
  ///
  /// Example:
  /// ```swift
  /// let messages: [Message] = [Message(role: "user", content: "Why is the sky blue?")]
  /// let result = await chat(options: (messages, ChatOptions(model: "meta/llama3.2-1b")))
  /// ```
  ///
  /// - Parameters:
  ///   - options: A tuple containing a `[Message]` array and a `ChatOptions` object.
  /// - Returns: An `async` `Result<Message, Failure>`. On success, the result contains the message reply;
  ///            on failure, it includes an error code and description.
  public func chat(
    options: ([Message], ChatOptions)
  ) async -> Result<Message, Failure> {
    let messages = options.0
    let chatOptions = options.1
    let engine: Engine
    if chatOptions.forceRemote && chatOptions.forceLocal {
      return .failure(
        Failure(
          code: .configError,
          message: "Cannot set both forceRemote and forceLocal to true"
        )
      )
    }
    if chatOptions.forceRemote {
      engine = remoteEngine
    } else if chatOptions.forceLocal {
      engine = mlxEngine
    } else {
      engine = self.engine
    }
    do {
      let result = try await engine.chat(
        messages,
        model: chatOptions.model,
        temperature: chatOptions.temperature,
        maxCompletionTokens: chatOptions.maxCompletionTokens,
        stream: chatOptions.stream,
        onStreamEvent: chatOptions.onStreamEvent,
        tools: chatOptions.tools
      )
      return .success(result)
    } catch {
      return .failure(
        error as? Failure ?? Failure(code: .unavailableError, message: error.localizedDescription))
    }
  }
}
