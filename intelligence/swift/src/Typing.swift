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

/// Represents the progress of an operation.
public struct Progress: Codable {
  public let totalBytes: Int?
  public let loadedBytes: Int?
  public let percentage: Double?
  public let description: String?
}

enum Role: String, Codable, Sendable, CaseIterable {
  case user
  case system
  case assistant
}

/// Represents a message in a chat session.
public struct Message: Codable, Sendable {
  private let _role: Role
  /// The role of the sender (e.g., "user", "system", "assistant").
  public let role: String

  /// The content of the message.
  public let content: String

  /// An optional list of calls to specific tools
  public let toolCalls: [ToolCall]?

  enum CodingKeys: String, CodingKey {
    case role, content, toolCalls
  }

  public init(role: String, content: String, toolCalls: [ToolCall]? = nil) throws {
    guard let parsed = Role(rawValue: role) else {
      let validRoles = Role.allCases.map(\.rawValue).joined(separator: ", ")
      throw Failure(
        code: .invalidArgumentsError,
        message: "Invalid message role: \(role). Available roles are: \(validRoles)."
      )
    }
    self._role = parsed
    self.role = role
    self.content = content
    self.toolCalls = toolCalls
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let roleString = try container.decode(String.self, forKey: .role)

    guard let parsed = Role(rawValue: roleString) else {
      let validRoles = Role.allCases.map(\.rawValue).joined(separator: ", ")
      throw Failure(
        code: .invalidArgumentsError,
        message: "Invalid message role: \(roleString). Available roles are: \(validRoles)."
      )
    }

    self._role = parsed
    self.role = roleString
    self.content = try container.decode(String.self, forKey: .content)
    self.toolCalls = try container.decodeIfPresent([ToolCall].self, forKey: .toolCalls)
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(role, forKey: .role)
    try container.encode(content, forKey: .content)
    try container.encodeIfPresent(toolCalls, forKey: .toolCalls)
  }
}

/// Represents a call to a specific tool with its name and arguments.
public typealias ToolCall = [String: ToolCallDetails]

/// Represents the details of a tool call.
public struct ToolCallDetails: Codable, Sendable {
  /// The name of the tool being called.
  public let name: String

  /// The arguments passed to the tool as key-value pairs.
  public let arguments: [String: String]
}

/// Represents a property of a tool's function parameter.
public struct ToolParameterProperty: Codable, Sendable {
  /// The data type of the property (e.g., "string", "number").
  public let type: String

  /// A description of the property.
  public let description: String

  /// An optional list of allowed values for the property.
  public let `enum`: [String]?
}

/// Represents the parameters required for a tool's function.
public struct ToolFunctionParameters: Codable, Sendable {
  /// The data type of the parameters (e.g., "object").
  public let type: String

  /// A dictionary defining the properties of each parameter.
  public let properties: [String: ToolParameterProperty]

  /// A list of parameter names that are required.
  public let required: [String]
}

/// Represents the function provided by a tool.
public struct ToolFunction: Codable, Sendable {
  /// The name of the function provided by the tool.
  public let name: String

  /// A brief description of what the function does.
  public let description: String

  /// The parameters required for invoking the function.
  public let parameters: ToolFunctionParameters
}

/// Represents a tool with details about its type, function, and parameters.
public struct Tool: Codable, Sendable {
  /// The type of the tool (e.g., "function" or "plugin").
  public let type: String

  /// Details about the function provided by the tool.
  public let function: ToolFunction
}

/// Represents a single event in a streaming response.
public struct StreamEvent: Codable, Sendable {
  /// The chunk of text data received in the stream event.
  public let chunk: String
}

/// Represents the options available for a chat interaction.
public struct ChatOptions {
  public var model: String?
  public var temperature: Float?
  public var maxCompletionTokens: Int?
  public var stream: Bool
  public var onStreamEvent: (@Sendable (StreamEvent) -> Void)?
  public var tools: [Tool]?
  public var forceRemote: Bool
  public var forceLocal: Bool
  public var encrypt: Bool

  public init(
    model: String? = nil,
    temperature: Float? = nil,
    maxCompletionTokens: Int? = nil,
    stream: Bool = false,
    onStreamEvent: (@Sendable (StreamEvent) -> Void)? = nil,
    tools: [Tool]? = nil,
    forceRemote: Bool = false,
    forceLocal: Bool = false,
    encrypt: Bool = false
  ) {
    self.model = model
    self.temperature = temperature
    self.maxCompletionTokens = maxCompletionTokens
    self.stream = stream
    self.onStreamEvent = onStreamEvent
    self.tools = tools
    self.forceRemote = forceRemote
    self.forceLocal = forceLocal
    self.encrypt = encrypt
  }
}
