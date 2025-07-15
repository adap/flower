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
import MLXLMCommon
import MLXRandom

class MlxEngine: Engine {
  enum LoadState {
    case idle
    case loaded(ModelContainer)
  }

  var loadState = LoadState.idle
  var modelConfiguration = llama3_2_1B {
    didSet {
      loadState = .idle
    }
  }

  func load(_ progressHandler: (@Sendable (Progress) -> Void)? = nil) async throws -> ModelContainer
  {
    switch loadState {
    case .idle:
      MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

      let modelContainer = try await LLMModelFactory.shared.loadContainer(
        configuration: modelConfiguration
      ) { [modelConfiguration] progress in
        if let progressHandler = progressHandler {
          let ret = Progress(
            totalBytes: Int(progress.totalUnitCount), loadedBytes: Int(progress.completedUnitCount),
            percentage: progress.fractionCompleted, description: modelConfiguration.name)
          progressHandler(ret)
        }
      }

      loadState = .loaded(modelContainer)
      return modelContainer

    case .loaded(let modelContainer):
      return modelContainer
    }
  }

  private static func convertToolsToDictionaries(_ tools: [Tool]?) throws -> [[String: Any]]? {
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    if let tools = tools {
      let jsonData = try encoder.encode(tools)
      let jsonObject = try JSONSerialization.jsonObject(with: jsonData, options: [])
      if let dictionaryArray = jsonObject as? [[String: Any]] {
        return dictionaryArray
      }
    }
    return nil
  }

  func chat(
    _ messages: [Message],
    model: String? = nil,
    temperature: Float? = nil,
    maxCompletionTokens: Int? = nil,
    stream: Bool = false,
    onStreamEvent: (@Sendable (StreamEvent) -> Void)? = nil,
    tools: [Tool]? = nil
  ) async throws -> Message {
    if let model = model {
      modelConfiguration = modelMapping[model] ?? llama3_2_1B
    }

    let generateParameters = GenerateParameters(temperature: temperature ?? 0.6)
    let displayEveryNTokens = 4

    let modelContainer = try await load()
    MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))
    let result = try await modelContainer.perform { context in
      let input = try await context.processor.prepare(
        input: .init(
          messages: messages.map {
            Dictionary(uniqueKeysWithValues: [("role", $0.role), ("content", $0.content)])
          },
          tools: MlxEngine.convertToolsToDictionaries(tools))
      )
      return try MLXLMCommon.generate(
        input: input, parameters: generateParameters, context: context
      ) { tokens in
        if tokens.count % displayEveryNTokens == 0 {
          let text = context.tokenizer.decode(tokens: tokens)
          if stream, let onStreamEvent = onStreamEvent {
            onStreamEvent(StreamEvent(chunk: text))
          }
        }
        if tokens.count >= maxCompletionTokens ?? 240 {
          return .stop
        } else {
          return .more
        }
      }
    }
    return try Message(role: "assistant", content: result.output)
  }

  func fetchModel(model: String, callback: @escaping (Progress) -> Void) async throws {
    modelConfiguration = modelMapping[model] ?? ModelRegistry.llama3_2_1B_4bit
    let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    let base = documents.appending(component: "huggingface")
    if case .id(let id, _) = modelConfiguration.id {
      let modelUrl = base.appending(component: "models").appending(component: id)
      if FileManager.default.fileExists(atPath: modelUrl.path) { return }
    }
    _ = try await load()
  }
}
