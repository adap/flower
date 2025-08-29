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

import kotlinx.datetime.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Represents the progress of an operation.
 *
 * @property totalBytes The total number of bytes expected.
 * @property loadedBytes The number of bytes loaded so far.
 * @property percentage The completion percentage of the operation.
 * @property description A textual description of the progress state.
 */
@Serializable
data class Progress(
  val totalBytes: Int? = null,
  val loadedBytes: Int? = null,
  val percentage: Double? = null,
  val description: String? = null,
)

/**
 * Represents a message in a chat session.
 *
 * @property role The role of the sender (e.g., "user", "system", "assistant").
 * @property content The content of the message.
 * @property toolCalls An optional list of tool calls associated with the message.
 */
@Serializable
data class Message(val role: String, val content: String, val toolCalls: List<ToolCall>? = null) {
  init {
    if (role !in Constants.ALLOWED_ROLES) {
      throw Failure(
        FailureCode.InvalidArgumentsError,
        "Invalid message role: $role. " +
          "Available roles are: ${Constants.ALLOWED_ROLES.joinToString(", ")}.",
      )
    }
  }
}

/** Represents a call to a specific tool with its name and arguments. */
typealias ToolCall = Map<String, ToolCallDetails>

/**
 * Represents the details of a tool call.
 *
 * @property name The name of the tool being called.
 * @property arguments The arguments passed to the tool as key-value pairs.
 */
@Serializable data class ToolCallDetails(val name: String, val arguments: Map<String, String>)

/**
 * Represents a property of a tool's function parameter.
 *
 * @property type The data type of the property (e.g., "string", "number").
 * @property description A description of the property.
 * @property enum An optional list of allowed values for the property.
 */
@Serializable
data class ToolParameterProperty(
  val type: String,
  val description: String,
  val `enum`: List<String>? = null,
)

/**
 * Represents the parameters required for a tool's function.
 *
 * @property type The data type of the parameters (e.g., "object").
 * @property properties A dictionary defining the properties of each parameter.
 * @property required A list of parameter names that are required.
 */
@Serializable
data class ToolFunctionParameters(
  val type: String,
  val properties: Map<String, ToolParameterProperty>,
  val required: List<String>,
)

/**
 * Represents the function provided by a tool.
 *
 * @property name The name of the function provided by the tool.
 * @property description A brief description of what the function does.
 * @property parameters The parameters required for invoking the function.
 */
@Serializable
data class ToolFunction(
  val name: String,
  val description: String,
  val parameters: ToolFunctionParameters,
)

/**
 * Represents a tool with details about its type, function, and parameters.
 *
 * @property type The type of the tool (e.g., "function" or "plugin").
 * @property function Details about the function provided by the tool.
 */
@Serializable data class Tool(val type: String, val function: ToolFunction)

/**
 * Represents a single event in a streaming response.
 *
 * @property chunk The chunk of text data received in the stream event.
 */
@Serializable data class StreamEvent(val chunk: String)

/**
 * Represents the options available for a chat interaction.
 *
 * @property model Optional model identifier.
 * @property temperature Optional sampling temperature for creativity.
 * @property maxCompletionTokens Optional maximum number of tokens to generate.
 * @property stream Whether to stream responses back.
 * @property onStreamEvent Callback for handling streaming events.
 * @property tools Optional list of tools available to the model.
 * @property forceRemote Force using a remote engine for the chat.
 * @property forceLocal Force using a local engine for the chat.
 * @property encrypt Whether to use encryption for remote inference.
 */
data class ChatOptions(
  var model: String? = null,
  var temperature: Float? = null,
  var maxCompletionTokens: Int? = null,
  var stream: Boolean = false,
  var onStreamEvent: ((StreamEvent) -> Unit)? = null,
  var tools: List<Tool>? = null,
  var forceRemote: Boolean = false,
  var forceLocal: Boolean = false,
  var encrypt: Boolean = false,
)

@Serializable
internal data class ChoiceMessage(
  val role: String,
  val content: String? = null,
  @SerialName("tool_calls") val toolCalls: List<ToolCall>? = null,
)

@Serializable internal data class Choice(val index: Int, val message: ChoiceMessage)

@Serializable internal data class StreamChoice(val index: Int, val delta: DeltaMessage)

@Serializable internal data class DeltaMessage(val content: String, val role: String)

@Serializable
internal data class StreamChunk(
  val `object`: String,
  val model: String,
  val choices: List<StreamChoice> = emptyList(),
)

@Serializable data class ServerSentEvent(val data: String)

@Serializable
internal data class Usage(
  @SerialName("completion_tokens") val completionTokens: Int,
  @SerialName("prompt_tokens") val promptTokens: Int,
  @SerialName("total_tokens") val totalTokens: Int,
)

@Serializable
internal data class ChatCompletionsRequest(
  val model: String,
  val messages: List<Message>,
  val temperature: Float? = null,
  @SerialName("max_completion_tokens") val maxCompletionTokens: Int? = null,
  val stream: Boolean? = null,
  val tools: List<Tool>? = null,
  val encrypt: Boolean? = null,
)

@Serializable
internal data class ChatCompletionsResponse(
  val `object`: String,
  val created: Int,
  val model: String,
  val choices: List<Choice>,
  val usage: Usage,
)

@Serializable
internal data class ModelListResponse(val `object`: String, val data: List<ModelData>)

@Serializable
internal data class ModelData(
  val id: String,
  val `object`: String,
  val created: Int,
  @SerialName("owned_by") val ownedBy: String,
)

@Serializable
data class SubmitClientPublicKeyResponse(val expiresAt: Instant, val encryptionId: String)

@Serializable
data class GetServerPublicKeyResponse(val publicKeyEncoded: String, val expiresAt: Instant)
