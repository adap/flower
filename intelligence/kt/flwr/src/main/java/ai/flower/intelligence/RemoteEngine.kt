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

import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.plugins.websocket.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.serialization.json.Json

internal interface RemoteEngineProtocol : Engine {
  var apiKey: String
}

internal class RemoteEngine(
  private val baseURL: String = Constants.BASE_URL,
  override var apiKey: String = "",
) : RemoteEngineProtocol {

  private val client =
    HttpClient(CIO) {
      install(ContentNegotiation) { json(Json { ignoreUnknownKeys = true }) }
      defaultRequest {
        header("FI-SDK-Type", Constants.SDK)
        header("FI-SDK-Version", Constants.VERSION)
      }
    }

  private val authorization: String
    get() = "Bearer $apiKey"

  override suspend fun chat(
    messages: List<Message>,
    model: String?,
    temperature: Float?,
    maxCompletionTokens: Int?,
    stream: Boolean,
    onStreamEvent: ((StreamEvent) -> Unit)?,
    tools: List<Tool>?,
  ): Message {
    val chosenModel = model ?: "meta/llama3.2-1b"

    val payload =
      ChatCompletionsRequest(
        model = chosenModel,
        messages = messages,
        temperature = temperature,
        maxCompletionTokens = maxCompletionTokens,
        stream = null,
        tools = tools,
        encrypt = null,
      )

    val url = "$baseURL${Constants.CHAT_COMPLETION_PATH}"

    return if (stream) {
      var accumulatedResponse = ""
      NetworkService.streamElement(
        client = client,
        element = payload,
        authorization = authorization,
        url = url,
      ) { streamElements: List<ServerSentEvent> ->
        for (event in streamElements) {
          val chunk = Json.decodeFromString<StreamChunk>(sse.data)
          for (choice in chunk.choices) {
            val deltaContent = choice.delta.content
            onStreamEvent?.invoke(StreamEvent(deltaContent))
            accumulatedResponse += deltaContent
          }
        }
      }
      Message(role = "assistant", content = accumulatedResponse)
    } else {
      val response: ChatCompletionsResponse =
        NetworkService.postElement(
          client = client,
          element = payload,
          authorization = authorization,
          url = url,
        )

      val message =
        response.choices.firstOrNull()?.message
          ?: throw Failure(FailureCode.RemoteError, "No message found in response")

      Message(role = message.role, content = message.content ?: "", toolCalls = message.toolCalls)
    }
  }

  override suspend fun fetchModel(model: String, callback: (Progress) -> Unit) {
    TODO("Not yet implemented")
  }
}

internal object NetworkService {
  suspend inline fun <reified Element : Any> getElement(
    client: HttpClient,
    url: String,
    authorization: String? = null,
  ): Element {
    val response =
      client.get(url) {
        headers {
          append(HttpHeaders.ContentType, ContentType.Application.Json)
          authorization?.let { append(HttpHeaders.Authorization, it) }
        }
      }
    checkStatusCode(response)
    return response.body()
  }

  suspend inline fun <reified Element : Any> postElement(
    client: HttpClient,
    element: Any,
    authorization: String? = null,
    url: String,
  ): Element {
    val response =
      client.post(url) {
        contentType(ContentType.Application.Json)
        authorization?.let { header(HttpHeaders.Authorization, it) }
        setBody(element)
      }
    checkStatusCode(response)
    return response.body()
  }

  suspend inline fun <reified StreamElement : Any> streamElement(
    client: HttpClient,
    element: Any,
    authorization: String? = null,
    url: String,
    crossinline onStreamEvent: (List<StreamElement>) -> Unit,
  ) {
    val response =
      client.post(url) {
        contentType(ContentType.Application.Json)
        authorization?.let { header(HttpHeaders.Authorization, it) }
        setBody(element)
      }
    checkStatusCode(response)
    val bodyString = response.bodyAsText()
    val lines = bodyString.split("\n").filter { it.isNotBlank() }
    for (line in lines) {
      val streamElement = Json.decodeFromString<List<StreamElement>>(line)
      onStreamEvent(streamElement)
    }
  }

  fun checkStatusCode(response: HttpResponse) {
    when (response.status.value) {
      in 200..299 -> Unit
      401,
      403,
      407 ->
        throw Failure(
          FailureCode.AuthenticationError,
          "Authentication error: ${response.status.value}",
        )
      404,
      502,
      503 ->
        throw Failure(FailureCode.UnavailableError, "Service unavailable: ${response.status.value}")
      408,
      504 -> throw Failure(FailureCode.TimeoutError, "Request timed out: ${response.status.value}")
      in 500..599 ->
        throw Failure(FailureCode.RemoteError, "Server error: ${response.status.value}")
      else ->
        throw Failure(FailureCode.ConnectionError, "Unexpected error: ${response.status.value}")
    }
  }
}
