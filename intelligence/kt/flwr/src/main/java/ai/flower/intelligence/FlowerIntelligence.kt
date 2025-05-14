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

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * FlowerIntelligence is the core intelligence service for Flower Labs.
 *
 * It facilitates chat, generation, and summarization tasks, with the option of using a local or
 * remote engine based on configuration and availability.
 */
object FlowerIntelligence {
  internal var remoteEngine = RemoteEngine()
  private var remoteHandoff: Boolean = true

  /**
   * API key for FlowerIntelligence.
   *
   * Setting this value also updates the remote engine's API key.
   */
  var apiKey: String = ""
    set(value) {
      field = value
      remoteEngine.apiKey = value
    }

  private val engine: Engine
    get() = remoteEngine

  /**
   * Conducts a chat interaction using a single string input.
   *
   * This method automatically wraps the input string as a message from the user. Additional
   * parameters like temperature or model can be configured via [maybeOptions].
   *
   * Example:
   * ```
   * val result = FlowerIntelligence.chat("Why is the sky blue?", ChatOptions(temperature = 0.7))
   * ```
   *
   * @param input A string representing the user message.
   * @param maybeOptions Optional [ChatOptions] to customize the chat behavior.
   * @return A [Result] containing the reply [Message] on success, or [Failure] on error.
   */
  suspend fun chat(input: String, maybeOptions: ChatOptions? = null): Result<Message> =
    withContext(Dispatchers.Default) {
      var selectedEngine = engine

      maybeOptions?.let { options ->
        if (options.forceLocal && options.forceRemote) {
          return@withContext Result.failure(
            Failure(FailureCode.ConfigError, "Cannot set both forceRemote and forceLocal to true")
          )
        }
        selectedEngine = remoteEngine
      }

      val messages = listOf(Message(role = "user", content = input))

      return@withContext try {
        val result =
          selectedEngine.chat(
            messages,
            model = maybeOptions?.model,
            temperature = maybeOptions?.temperature,
            maxCompletionTokens = maybeOptions?.maxCompletionTokens,
            stream = maybeOptions?.stream ?: false,
            onStreamEvent = maybeOptions?.onStreamEvent,
            tools = maybeOptions?.tools,
          )
        Result.success(result)
      } catch (e: Exception) {
        Result.failure(
          e as? Failure
            ?: Failure(FailureCode.UnavailableError, e.localizedMessage ?: "Unknown error")
        )
      }
    }

  /**
   * Conducts a chat interaction using a list of messages and optional options.
   *
   * This method allows for multi-message conversations by accepting a list of [Message] objects and
   * a [ChatOptions] configuration.
   *
   * Example:
   * ```
   * val messages = listOf(
   *   Message(role = "system", content = "You are a helpful assistant."),
   *   Message(role = "user", content = "Why is the sky blue?")
   * )
   * val options = ChatOptions(model = "meta/llama3.2-1b")
   * val result = FlowerIntelligence.chat(messages, options)
   * ```
   *
   * @param messages The list of [Message] objects representing the conversation.
   * @param chatOptions Optional [ChatOptions] to customize behavior.
   * @return A [Result] containing the reply [Message] on success, or [Failure] on error.
   */
  suspend fun chat(messages: List<Message>, chatOptions: ChatOptions? = null): Result<Message> =
    withContext(Dispatchers.Default) {
      val selectedEngine = remoteEngine

      return@withContext try {
        val result =
          selectedEngine.chat(
            messages,
            model = chatOptions?.model,
            temperature = chatOptions?.temperature,
            maxCompletionTokens = chatOptions?.maxCompletionTokens,
            stream = chatOptions?.stream ?: false,
            onStreamEvent = chatOptions?.onStreamEvent,
            tools = chatOptions?.tools,
          )
        Result.success(result)
      } catch (e: Exception) {
        Result.failure(
          e as? Failure
            ?: Failure(FailureCode.UnavailableError, e.localizedMessage ?: "Unknown error")
        )
      }
    }
}
