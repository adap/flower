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

object FlowerIntelligence {
    var remoteEngine = RemoteEngine()
    var remoteHandoff: Boolean = false

    var apiKey: String = ""
        set(value) {
            field = value
            remoteEngine.apiKey = value
        }

    private val engine: Engine
        get() = remoteEngine



    suspend fun chat(
        input: String,
        maybeOptions: ChatOptions? = null
    ): Result<Message> = withContext(Dispatchers.Default) {
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
            val result = selectedEngine.chat(
                messages,
                model = maybeOptions?.model,
                temperature = maybeOptions?.temperature,
                maxCompletionTokens = maybeOptions?.maxCompletionTokens,
                stream = maybeOptions?.stream ?: false,
                onStreamEvent = maybeOptions?.onStreamEvent,
                tools = maybeOptions?.tools
            )
            Result.success(result)
        } catch (e: Exception) {
            Result.failure(e as? Failure ?: Failure(FailureCode.UnavailableError, e.localizedMessage ?: "Unknown error"))
        }
    }

    suspend fun chat(
        options: Pair<List<Message>, ChatOptions>
    ): Result<Message> = withContext(Dispatchers.Default) {
        val (messages, chatOptions) = options
        val selectedEngine = remoteEngine

        return@withContext try {
            val result = selectedEngine.chat(
                messages,
                model = chatOptions.model,
                temperature = chatOptions.temperature,
                maxCompletionTokens = chatOptions.maxCompletionTokens,
                stream = chatOptions.stream,
                onStreamEvent = chatOptions.onStreamEvent,
                tools = chatOptions.tools
            )
            Result.success(result)
        } catch (e: Exception) {
            Result.failure(e as? Failure ?: Failure(FailureCode.UnavailableError, e.localizedMessage ?: "Unknown error"))
        }
    }
}

