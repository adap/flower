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

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import io.mockk.coEvery
import io.mockk.mockk
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class FlowerIntelligenceAndroidTest {

  private lateinit var context: Context
  private lateinit var mockEngine: RemoteEngine

  @Before
  fun setup() {
    context = ApplicationProvider.getApplicationContext()
    mockEngine = mockk(relaxed = true)
    FlowerIntelligence.remoteEngine = mockEngine
    FlowerIntelligence.apiKey = "android-test-key"
  }

  @Test
  fun testChatSuccessPathOnAndroid() = runBlocking {
    val input = "Hello from Android!"
    val response = Message(role = "assistant", content = "Hi from backend")

    coEvery {
      mockEngine.chat(
        listOf(Message(role = "user", content = input)),
        model = null,
        temperature = null,
        maxCompletionTokens = null,
        stream = false,
        onStreamEvent = null,
        tools = null,
      )
    } returns response

    val result = FlowerIntelligence.chat(input)

    assertTrue(result.isSuccess)
    assertEquals("Hi from backend", result.getOrNull()?.content)
  }

  @Test
  fun testErrorHandlingOnAndroid() = runBlocking {
    val input = "Trigger failure"

    coEvery { mockEngine.chat(any(), any(), any(), any(), any(), any(), any()) } throws
      RuntimeException("Server error")

    val result = FlowerIntelligence.chat(input)

    assertTrue(result.isFailure)
    val failure = result.exceptionOrNull() as Failure
    assertEquals(FailureCode.UnavailableError, failure.code)
    assertEquals("Server error", failure.message)
  }
}
