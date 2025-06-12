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

import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test

class FlowerIntelligenceTest {

  private lateinit var mockEngine: RemoteEngine

  @BeforeEach
  fun setup() {
    mockEngine = mockk(relaxed = true)

    var capturedKey = ""

    every { mockEngine.apiKey = any() } answers { capturedKey = arg(0) }
    every { mockEngine.apiKey } answers { capturedKey }

    FlowerIntelligence.remoteEngine = mockEngine
  }

  @Test
  fun `should update remote engine api key`() {
    val newKey = "new-api-key"
    FlowerIntelligence.apiKey = newKey
    assertEquals(newKey, mockEngine.apiKey)
  }

  @Test
  fun `should return failure when both forceRemote and forceLocal are true`() = runBlocking {
    val options = ChatOptions(forceLocal = true, forceRemote = true)
    val result = FlowerIntelligence.chat("hello", options)

    assertTrue(result.isFailure)
    val exception = result.exceptionOrNull() as Failure
    assertEquals(FailureCode.ConfigError, exception.code)
    assertTrue(exception.message.contains("Cannot set both forceRemote and forceLocal to true"))
  }

  @Test
  fun `should delegate chat call to engine with correct parameters`() = runBlocking {
    val message = Message(role = "assistant", content = "Hello, user!")
    val inputMessages = listOf(Message(role = "user", content = "Hi"))

    coEvery {
      mockEngine.chat(
        inputMessages,
        model = "model",
        temperature = 0.9f,
        maxCompletionTokens = 300,
        stream = false,
        onStreamEvent = any(),
        tools = null,
      )
    } returns message

    val options = ChatOptions(model = "model", temperature = 0.9f, maxCompletionTokens = 300)
    val result = FlowerIntelligence.chat("Hi", options)

    assertTrue(result.isSuccess)
    assertEquals("Hello, user!", result.getOrNull()?.content)
  }

  @Test
  fun `should handle exception from engine chat`() = runBlocking {
    val exception = RuntimeException("Engine down")
    coEvery { mockEngine.chat(any(), any(), any(), any(), any(), any(), any()) } throws exception

    val result = FlowerIntelligence.chat("fail test", ChatOptions())

    assertTrue(result.isFailure)
    val failure = result.exceptionOrNull() as Failure
    assertEquals(FailureCode.UnavailableError, failure.code)
    assertEquals("Engine down", failure.message)
  }

  @Test
  fun `should run second chat variant with custom message list`() = runBlocking {
    val userMessages = listOf(Message(role = "user", content = "How are you?"))
    val response = Message(role = "assistant", content = "I'm good!")

    coEvery {
      mockEngine.chat(
        userMessages,
        model = "model",
        temperature = 0.7f,
        maxCompletionTokens = 150,
        stream = true,
        onStreamEvent = null,
        tools = null,
      )
    } returns response

    val options =
      ChatOptions(model = "model", temperature = 0.7f, maxCompletionTokens = 150, stream = true)

    // Updated to match new chat signature: messages + optional options
    val result = FlowerIntelligence.chat(userMessages, options)

    assertTrue(result.isSuccess)
    assertEquals("I'm good!", result.getOrNull()?.content)
  }
}
