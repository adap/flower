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
// =============================================================================

import {
  ChatResponseResult,
  Embedding,
  EmbeddingInput,
  FailureCode,
  Message,
  Progress,
  ResponseFormat,
  Result,
  StreamEvent,
  Tool,
  ToolChoice,
} from '../typing';

export interface Engine {
  chat(
    messages: Message[],
    model: string,
    temperature?: number,
    topP?: number,
    maxCompletionToken?: number,
    responseFormat?: ResponseFormat,
    stream?: boolean,
    onStreamEvent?: (event: StreamEvent) => void,
    tools?: Tool[],
    toolChoice?: ToolChoice,
    encrypt?: boolean,
    signal?: AbortSignal
  ): Promise<ChatResponseResult>;
  embed(model: string, input: EmbeddingInput): Promise<Result<Embedding[]>>;
  fetchModel(model: string, callback: (progress: Progress) => void): Promise<Result<void>>;
  isSupported(model: string): Promise<Result<void>>;
}

export abstract class BaseEngine implements Engine {
  async chat(
    _messages: Message[],
    _model: string,
    _temperature?: number,
    _topP?: number,
    _maxCompletionTokens?: number,
    _responseFormat?: ResponseFormat,
    _stream?: boolean,
    _onStreamEvent?: (event: StreamEvent) => void,
    _tools?: Tool[],
    _toolChoice?: ToolChoice,
    _encrypt?: boolean,
    _signal?: AbortSignal
  ): Promise<ChatResponseResult> {
    await Promise.resolve();
    return {
      ok: false,
      failure: { code: FailureCode.NotImplementedError, description: 'Method not implemented.' },
    };
  }

  async embed(_model: string, _input: EmbeddingInput): Promise<Result<Embedding[]>> {
    await Promise.resolve();
    return {
      ok: false,
      failure: { code: FailureCode.NotImplementedError, description: 'Method not implemented.' },
    };
  }

  async fetchModel(_model: string, _callback: (progress: Progress) => void): Promise<Result<void>> {
    await Promise.resolve();
    return {
      ok: false,
      failure: { code: FailureCode.NotImplementedError, description: 'Method not implemented.' },
    };
  }

  async isSupported(_model: string): Promise<Result<void>> {
    await Promise.resolve();
    return {
      ok: false,
      failure: { code: FailureCode.NotImplementedError, description: 'Method not implemented.' },
    };
  }
}
