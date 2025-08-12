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

protocol RemoteEngineProtocol: Engine {
  var apiKey: String { get set }
}

class RemoteEngine: RemoteEngineProtocol {
  private let baseURL: String = remoteUrl
  var apiKey: String = ""
  var authorization: String {
    "Bearer \(apiKey)"
  }

  func chat(
    _ messages: [Message],
    model: String?,
    temperature: Float?,
    maxCompletionTokens: Int?,
    stream: Bool,
    onStreamEvent: (@Sendable (StreamEvent) -> Void)?,
    tools: [Tool]?
  ) async throws -> Message {
    let model = model ?? "meta/llama3.2-1b"
    guard modelMapping[model] != nil else {
      throw Failure(
        code: .configError,
        message: "Model \(model) is not supported"
      )
    }
    let payload = ChatCompletionsRequest(
      model: model,
      messages: messages,
      temperature: temperature,
      maxCompletionTokens: maxCompletionTokens,
      stream: nil,
      tools: tools,
      encrypt: nil
    )

    guard let url = URL(string: "\(baseURL)\(chatCompletionPath)") else {
      throw Failure(
        code: .connectionError,
        message: URLError(.badURL).localizedDescription
      )
    }
    if stream {
      var accumulatedResponse = ""
      try await NetworkService.streamElement(payload, authorization: authorization, on: url) {
        (streamElement: ServerSentEvent) in
        guard let json = streamElement.data.data(using: .utf8) else { return }
        guard let chunk = try? NetworkService.parseJson(from: json, as: StreamChunk.self) else {
          return
        }
        for choice in chunk.choices {
          let deltaContent = choice.delta.content
          onStreamEvent?(StreamEvent(chunk: deltaContent))
          accumulatedResponse += deltaContent
        }
      }
      return try Message(
        role: "assistant",
        content: accumulatedResponse
      )
    }
    let response: ChatCompletionsResponse = try await NetworkService.postElement(
      payload,
      authorization: authorization,
      on: url
    )
    guard let message = response.choices.first?.message else {
      throw Failure(code: .remoteError, message: "No message found in response")
    }
    return try Message(
      role: message.role,
      content: message.content ?? "",
      toolCalls: message.toolCalls
    )
  }
}

enum NetworkService {
  static func parseJson<T: Decodable>(
    from data: Data,
    as type: T.Type
  ) throws -> T {
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    return try decoder.decode(T.self, from: data)
  }

  static func urlRequest(
    _ method: String,
    url: URL,
    authorization: String? = nil,
    body: Data? = nil
  ) -> URLRequest {
    var urlRequest = URLRequest(url: url)
    urlRequest.httpMethod = method

    urlRequest.addValue("application/json", forHTTPHeaderField: "Content-Type")
    if let authorization = authorization {
      urlRequest.addValue(authorization, forHTTPHeaderField: "Authorization")
    }

    urlRequest.addValue(sdk, forHTTPHeaderField: "FI-SDK-Type")
    urlRequest.addValue(version, forHTTPHeaderField: "FI-SDK-Version")

    urlRequest.httpBody = body

    return urlRequest
  }

  static func sendRequest<Element: Decodable>(_ urlRequest: URLRequest) async throws -> Element {
    let (data, response) = try await URLSession.shared.data(for: urlRequest)
    guard let httpResponse = response as? HTTPURLResponse else {
      throw Failure(
        code: .connectionError,
        message: "Invalid response from server"
      )
    }
    try checkStatusCode(httpResponse)
    return try parseJson(from: data, as: Element.self)
  }

  static func checkStatusCode(_ httpResponse: HTTPURLResponse) throws {
    switch httpResponse.statusCode {
    case 200...299:
      return

    case 401, 403, 407:
      throw Failure(
        code: .authenticationError,
        message: "Authentication error: \(httpResponse.statusCode)"
      )

    case 404, 502, 503:
      throw Failure(
        code: .unavailableError,
        message: "Service unavailable: \(httpResponse.statusCode)"
      )

    case 408, 504:
      throw Failure(
        code: .timeoutError,
        message: "Request timed out: \(httpResponse.statusCode)"
      )

    case 500...599:
      throw Failure(
        code: .remoteError,
        message: "Server error: \(httpResponse.statusCode)"
      )
    default:
      throw Failure(
        code: .connectionError,
        message: "Unexpected error: \(httpResponse.statusCode)"
      )
    }
  }

  static func getElement<Element: Decodable>(
    on route: URL,
    authorization: String? = nil
  ) async throws -> Element {
    try await sendRequest(urlRequest("GET", url: route, authorization: authorization))
  }

  static func getElements<Element: Decodable>(
    on route: URL,
    authorization: String? = nil
  ) async throws -> [Element] {
    try await getElement(on: route, authorization: authorization)
  }

  static func postElement<RequestElement: Encodable, ResponseElement: Decodable>(
    _ element: RequestElement,
    authorization: String? = nil,
    on route: URL
  ) async throws -> ResponseElement {
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    return try await sendRequest(
      urlRequest(
        "POST",
        url: route,
        authorization: authorization,
        body: try? encoder.encode(element)
      )
    )
  }

  static func streamElement<RequestElement: Encodable, StreamElement: Decodable>(
    _ element: RequestElement,
    authorization: String? = nil,
    on route: URL,
    onStreamEvent: @escaping (StreamElement) -> Void
  ) async throws {
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    let request = urlRequest(
      "POST",
      url: route,
      authorization: authorization,
      body: try? encoder.encode(element)
    )
    let (stream, response) = try await URLSession.shared.bytes(for: request)
    guard let httpResponse = response as? HTTPURLResponse else {
      throw Failure(
        code: .connectionError,
        message: "Invalid response from server"
      )
    }
    try checkStatusCode(httpResponse)
    for try await line in stream.lines {
      guard let json = line.data(using: .utf8) else { continue }
      let streamElement = try parseJson(from: json, as: StreamElement.self)
      onStreamEvent(streamElement)
    }
  }
}

struct ChoiceMessage: Codable {
  let role: String
  let content: String?
  let toolCalls: [ToolCall]?

  enum CodingKeys: String, CodingKey {
    case role, content
    case toolCalls = "tool_calls"
  }
}

struct Choice: Codable {
  let index: Int
  let message: ChoiceMessage
}

struct StreamChoice: Codable {
  let index: Int
  let delta: DeltaMessage
}

struct DeltaMessage: Codable {
  let content: String
  let role: String
}

struct ServerSentEvent: Codable {
  let data: String
}

struct StreamChunk: Codable {
  let object: String
  let model: String
  let choices: [StreamChoice]
}

struct Usage: Codable {
  let completionTokens: Int
  let promptTokens: Int
  let totalTokens: Int

  enum CodingKeys: String, CodingKey {
    case completionTokens = "completion_tokens"
    case promptTokens = "prompt_tokens"
    case totalTokens = "total_tokens"
  }
}

struct ChatCompletionsRequest: Codable {
  let model: String
  let messages: [Message]
  let temperature: Float?
  let maxCompletionTokens: Int?
  let stream: Bool?
  let tools: [Tool]?
  let encrypt: Bool?

  enum CodingKeys: String, CodingKey {
    case model, messages, temperature, tools, encrypt, stream
    case maxCompletionTokens = "max_completion_tokens"
  }
}

struct ChatCompletionsResponse: Codable {
  let object: String
  let created: Int
  let model: String
  let choices: [Choice]
  let usage: Usage
}

struct ModelListResponse: Codable {
  let object: String
  let data: [ModelData]
}

struct ModelData: Codable {
  let id: String
  let object: String
  let created: Int
  let ownedBy: String

  enum CodingKeys: String, CodingKey {
    case id, object, created
    case ownedBy = "owned_by"
  }
}
