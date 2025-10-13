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

internal object Constants {
  const val BASE_URL = "https://api.flower.ai/"
  const val CHAT_COMPLETION_PATH = "v1/chat/completions"
  const val ENCRYPTION_PUBLIC_KEY_PATH = "encryption/public-key"
  const val ENCRYPTION_SERVER_PUBLIC_KEY_PATH = "encryption/server-public-key"
  const val SDK = "KT"
  const val VERSION = "0.2.6"
  val ALLOWED_ROLES = setOf("user", "system", "assistant")
}
