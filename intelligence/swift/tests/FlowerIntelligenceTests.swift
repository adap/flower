import Testing

@testable import FlowerIntelligence

class MockMlxEngine: Engine {
  var lastMessages: [Message]?
  var lastModel: String?
  var lastTemperature: Float?
  var lastMaxCompletionTokens: Int?
  var lastStream: Bool?
  var lastTools: [Tool]?

  func chat(
    _ messages: [Message],
    model: String?,
    temperature: Float?,
    maxCompletionTokens: Int?,
    stream: Bool,
    onStreamEvent: ((StreamEvent) -> Void)?,
    tools: [Tool]?
  ) async throws -> Message {
    lastMessages = messages
    return try Message(role: "assistant", content: "Mock Local Engine Response")
  }
}

class MockRemoteEngine: RemoteEngineProtocol {
  var apiKey: String = ""
  var lastMessages: [Message]?
  var lastModel: String?
  var lastTemperature: Float?
  var lastMaxCompletionTokens: Int?
  var lastStream: Bool?
  var lastTools: [Tool]?

  func chat(
    _ messages: [Message],
    model: String?,
    temperature: Float?,
    maxCompletionTokens: Int?,
    stream: Bool,
    onStreamEvent: ((StreamEvent) -> Void)?,
    tools: [Tool]?
  ) async throws -> Message {
    lastMessages = messages
    return try Message(role: "assistant", content: "Mock Remote Engine Response")
  }
}

class FlowerIntelligenceTests {
  var flowerIntelligence: FlowerIntelligence
  var mockMlxEngine: MockMlxEngine
  var mockRemoteEngine: MockRemoteEngine

  init() {
    mockMlxEngine = MockMlxEngine()
    mockRemoteEngine = MockRemoteEngine()
    flowerIntelligence = FlowerIntelligence(
      mlxEngine: mockMlxEngine, remoteEngine: mockRemoteEngine)
  }

  @Test
  func testUsesCorrectMlxEngine() async throws {
    flowerIntelligence.remoteHandoff = false
    _ = await flowerIntelligence.chat("Hello")
    try #require(mockMlxEngine.lastMessages != nil)
    try #require(mockRemoteEngine.lastMessages == nil)
  }

  @Test
  func testUsesCorrectRemoteEngine() async throws {
    flowerIntelligence.remoteHandoff = true
    _ = await flowerIntelligence.chat("Hello")
    try #require(mockRemoteEngine.lastMessages != nil)
    try #require(mockMlxEngine.lastMessages == nil)
  }

  @Test
  func testForceRemoteOverridesRemoteHandoff() async throws {
    flowerIntelligence.remoteHandoff = false
    _ = await flowerIntelligence.chat("Hello", maybeOptions: ChatOptions(forceRemote: true))
    try #require(mockRemoteEngine.lastMessages != nil)
    try #require(mockMlxEngine.lastMessages == nil)
  }

  @Test
  func testForceLocalOverridesRemoteHandoff() async throws {
    flowerIntelligence.remoteHandoff = true
    _ = await flowerIntelligence.chat("Hello", maybeOptions: ChatOptions(forceLocal: true))
    try #require(mockMlxEngine.lastMessages != nil)
    try #require(mockRemoteEngine.lastMessages == nil)
  }

  @Test
  func testForceRemoteAndForceLocalReturnFailure() async throws {
    let result = await flowerIntelligence.chat(
      "Hello", maybeOptions: ChatOptions(forceRemote: true, forceLocal: true))
    switch result {
    case .failure(let error):
      try #require(error.message == "Cannot set both forceRemote and forceLocal to true")
    default:
      Issue.record("Expected failure but got success")
    }
  }

  @Test
  func testChatFunctionWithSingleMessage() async throws {
    let result = await flowerIntelligence.chat("Test message")
    switch result {
    case .success(let response):
      let messages = try #require(mockMlxEngine.lastMessages)
      try #require(messages.first?.content == "Test message")
      try #require(response.role == "assistant")
      try #require(response.content == "Mock Local Engine Response")
    default:
      Issue.record("Expected success but got failure")
    }
  }

  @Test
  func testChatFunctionWithArrayOfMessages() async throws {
    let messages = [
      try Message(role: "system", content: "You are an AI"),
      try Message(role: "user", content: "What is Swift?"),
    ]
    let options = ChatOptions(model: "meta/llama3.2-1b")
    let result = await flowerIntelligence.chat(options: (messages, options))

    switch result {
    case .success(let response):
      let messages = try #require(mockMlxEngine.lastMessages)
      try #require(messages.count == 2)
      try #require(messages.last?.content == "What is Swift?")
      try #require(response.role == "assistant")
      try #require(response.content == "Mock Local Engine Response")
    default:
      Issue.record("Expected success but got failure")
    }
  }
}
