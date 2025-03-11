// Copyright 2024 Flower Labs GmbH. All Rights Reserved.

import Foundation
import MLX

protocol RemoteEngineProtocol: Engine {
    var apiKey: String { get set }
}

class RemoteEngine: RemoteEngineProtocol {
    private let baseURL: String = REMOTE_URL
    private let chatCompletionPath = CHAT_COMPLETION_PATH
    private let modelMapping = MODEL_MAPPING
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
        onStreamEvent: ((StreamEvent) -> Void)?,
        tools: [Tool]?
    ) async throws -> Message {
        let model = model ?? "meta/llama3.2-1b"
        guard let modelConfig = MODEL_MAPPING[model] else {
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
            try await NetworkService.streamElement(payload, authorization: authorization, on: url) { (streamElement: [StreamChoice]) in
                for choice in streamElement {
                    let deltaContent = choice.delta.content
                    onStreamEvent?(StreamEvent(chunk: deltaContent))
                    accumulatedResponse += deltaContent
                }
            }
            return Message(
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
        return Message(
            role: message.role,
            content: message.content ?? "",
            toolCalls: message.toolCalls
        )
    }
}

enum NetworkService {
    private static func parseJson<T: Decodable>(
        from data: Data,
        as type: T.Type
    ) throws -> T {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(T.self, from: data)
    }
    
    static func urlRequest(_ method: String,
                           url: URL,
                           authorization: String? = nil,
                           body: Data? = nil) -> URLRequest {
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = method
        
        urlRequest.addValue("application/json", forHTTPHeaderField: "Content-Type")
        if let authorization = authorization {
            urlRequest.addValue(authorization, forHTTPHeaderField: "Authorization")
        }
        
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
    
    static func getElement<Element: Decodable>(on route: URL,
                                               authorization: String? = nil) async throws -> Element {
        try await sendRequest(urlRequest("GET", url: route, authorization: authorization))
    }
    
    static func getElements<Element: Decodable>(on route: URL,
                                                authorization: String? = nil) async throws -> [Element] {
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
            urlRequest("POST",
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
        onStreamEvent:  @escaping (StreamElement) -> Void
    ) async throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let request = urlRequest("POST",
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

struct Usage: Codable {
    let totalDuration: Int // Time spent generating the response
    let loadDuration: Int // Time spent in nanoseconds loading the model
    let promptEvalCount: Int // Number of tokens in the prompt
    let promptEvalDuration: Int // Time spent in nanoseconds evaluating the prompt
    let evalCount: Int // Number of tokens in the response
    let evalDuration: Int // Time in nanoseconds spent generating the response

    enum CodingKeys: String, CodingKey {
        case totalDuration = "total_duration"
        case loadDuration = "load_duration"
        case promptEvalCount = "prompt_eval_count"
        case promptEvalDuration = "prompt_eval_duration"
        case evalCount = "eval_count"
        case evalDuration = "eval_duration"
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
