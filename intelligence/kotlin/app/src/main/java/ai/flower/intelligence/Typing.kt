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

import kotlinx.serialization.Serializable
import kotlinx.serialization.SerialName

@Serializable
data class Progress(
    val totalBytes: Int? = null,
    val loadedBytes: Int? = null,
    val percentage: Double? = null,
    val description: String? = null
)

@Serializable
data class Message(
    val role: String,
    val content: String,
    val toolCalls: List<ToolCall>? = null
)

typealias ToolCall = Map<String, ToolCallDetails>

@Serializable
data class ToolCallDetails(
    val name: String,
    val arguments: Map<String, String>
)

@Serializable
data class ToolParameterProperty(
    val type: String,
    val description: String,
    val `enum`: List<String>? = null
)

@Serializable
data class ToolFunctionParameters(
    val type: String,
    val properties: Map<String, ToolParameterProperty>,
    val required: List<String>
)

@Serializable
data class ToolFunction(
    val name: String,
    val description: String,
    val parameters: ToolFunctionParameters
)

@Serializable
data class Tool(
    val type: String,
    val function: ToolFunction
)

@Serializable
data class StreamEvent(
    val chunk: String
)

data class ChatOptions(
    var model: String? = null,
    var temperature: Float? = null,
    var maxCompletionTokens: Int? = null,
    var stream: Boolean = false,
    var onStreamEvent: ((StreamEvent) -> Unit)? = null,
    var tools: List<Tool>? = null,
    var forceRemote: Boolean = false,
    var forceLocal: Boolean = false,
    var encrypt: Boolean = false
)

@Serializable
data class ChoiceMessage(
    val role: String,
    val content: String? = null,
    @SerialName("tool_calls")
    val toolCalls: List<ToolCall>? = null
)

@Serializable
data class Choice(
    val index: Int,
    val message: ChoiceMessage
)

@Serializable
data class StreamChoice(
    val index: Int,
    val delta: DeltaMessage
)

@Serializable
data class DeltaMessage(
    val content: String,
    val role: String
)

@Serializable
data class Usage(
    @SerialName("completion_tokens")
    val completionTokens: Int,
    @SerialName("prompt_tokens")
    val promptTokens: Int,
    @SerialName("total_tokens")
    val totalTokens: Int
)

@Serializable
data class ChatCompletionsRequest(
    val model: String,
    val messages: List<Message>,
    val temperature: Float? = null,
    @SerialName("max_completion_tokens")
    val maxCompletionTokens: Int? = null,
    val stream: Boolean? = null,
    val tools: List<Tool>? = null,
    val encrypt: Boolean? = null
)

@Serializable
data class ChatCompletionsResponse(
    val `object`: String,
    val created: Int,
    val model: String,
    val choices: List<Choice>,
    val usage: Usage
)

@Serializable
data class ModelListResponse(
    val `object`: String,
    val data: List<ModelData>
)

@Serializable
data class ModelData(
    val id: String,
    val `object`: String,
    val created: Int,
    @SerialName("owned_by")
    val ownedBy: String
)
