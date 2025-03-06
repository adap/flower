// Copyright 2025 Flower Labs GmbH. All Rights Reserved.

import {
  ChatResponseResult,
  FailureCode,
  Message,
  Progress,
  Result,
  StreamEvent,
  Tool,
} from '../typing';

export interface Engine {
  chat(
    messages: Message[],
    model: string,
    temperature?: number,
    maxCompletionToken?: number,
    stream?: boolean,
    onStreamEvent?: (event: StreamEvent) => void,
    tools?: Tool[],
    encrypt?: boolean
  ): Promise<ChatResponseResult>;
  fetchModel(model: string, callback: (progress: Progress) => void): Promise<Result<void>>;
}

export abstract class BaseEngine implements Engine {
  async chat(
    _messages: Message[],
    _model: string,
    _temperature?: number,
    _maxCompletionTokens?: number,
    _stream?: boolean,
    _onStreamEvent?: (event: StreamEvent) => void,
    _tools?: Tool[],
    _encrypt?: boolean
  ): Promise<ChatResponseResult> {
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
}
