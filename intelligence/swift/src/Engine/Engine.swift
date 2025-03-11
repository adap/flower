// Copyright 2024 Flower Labs GmbH. All Rights Reserved.

import Foundation
import MLX

protocol Engine {
  func chat(
    _ messages: [Message],
    model: String?,
    temperature: Float?,
    maxCompletionTokens: Int?,
    stream: Bool,
    onStreamEvent: ((StreamEvent) -> Void)?,
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
    onStreamEvent: ((StreamEvent) -> Void)? = nil,
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

}
