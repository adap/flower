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

import Crypto
import Foundation
import MLXLLM
import MLXLMCommon

let remoteUrl = "https://api.flower.ai"
let chatCompletionPath = "/v1/chat/completions"
let hashAlgorithm = SHA256.self
let hkdfInfo = Data("ecdh key exchange".utf8)
let aesKeyLength = 32
let llama3_2_1B: ModelConfiguration = .init(
  id: "mlx-community/Llama-3.2-1B-Instruct-bf16",
  defaultPrompt: "What is the difference between a fruit and a vegetable?"
)
let llama3_2_3B: ModelConfiguration = .init(
  id: "mlx-community/Llama-3.2-3B-Instruct",
  defaultPrompt: "What is the difference between a fruit and a vegetable?"
)
let r1_distill: ModelConfiguration = .init(
  id: "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit"
)
let r1_distill_llama: ModelConfiguration = .init(id: "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit")
let r1_4bit: ModelConfiguration = .init(id: "mlx-community/DeepSeek-R1-4bit")

let modelMapping: [String: ModelConfiguration] = [
  "meta/llama3.2-1b": llama3_2_1B,
  "meta/llama3.2-3b": llama3_2_3B,
  "deepseek/r1-distill-qwen-32b/4-bit": r1_distill,
  "deepseek/r1-distill-llama-8b/q4": r1_distill_llama,
  "deepseek/r1-685b/q4": r1_4bit
]
