// Copyright 2025 Flower Labs GmbH. All Rights Reserved.

import Crypto
import Foundation
import MLXLLM
import MLXLMCommon

let remoteUrl = "https://api.flower.ai"
let chatCompletionPath = "/v1/chat/completions"
let hashAlgorithm = SHA256.self
let hkdfInfo = Data("ecdh key exchange".utf8)
let aesKeyLength = 32
let modelMapping: [String: ModelConfiguration] = [
  "meta/llama3.2-1b": ModelRegistry.llama3_2_1B_4bit,
  "meta/llama3.2-3b": ModelRegistry.llama3_2_3B_4bit,
]
