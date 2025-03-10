// Copyright 2025 Flower Labs GmbH. All Rights Reserved.

import Crypto
import Foundation
import MLXLMCommon
import MLXLLM

let REMOTE_URL = "https://api.flower.ai"
let CHAT_COMPLETION_PATH = "/v1/chat/completions"
let HASH_ALGORITHM = SHA256.self
let HKDF_INFO = Data("ecdh key exchange".utf8)
let AES_KEY_LENGTH = 32
let MODEL_MAPPING: [String: ModelConfiguration] = [
    "meta/llama3.2-1b": ModelRegistry.llama3_2_1B_4bit,
    "meta/llama3.2-3b": ModelRegistry.llama3_2_3B_4bit
]
