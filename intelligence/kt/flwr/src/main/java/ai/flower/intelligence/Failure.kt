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

enum class FailureCode(val code: Int) {
  // Local errors
  LocalError(100),
  LocalEngineChatError(101),
  LocalEngineFetchError(102),
  NoLocalProviderError(103),

  // Remote errors
  RemoteError(200),
  AuthenticationError(201),
  UnavailableError(202),
  TimeoutError(203),
  ConnectionError(204),

  // Engine errors
  EngineSpecificError(300),
  EncryptionError(301),

  // Config/Validation errors
  ConfigError(400),
  InvalidArgumentsError(401),
  InvalidRemoteConfigError(402),
  UnknownModelError(403),

  // Not implemented
  NotImplementedError(404),
}

class Failure(val code: FailureCode, override val message: String) : Exception(message)
