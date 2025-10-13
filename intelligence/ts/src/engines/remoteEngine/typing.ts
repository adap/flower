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

import { EmbeddingInput, Message, Tool, ToolCall, Usage } from '../../typing';

interface ChoiceMessage {
  role: string;
  content?: string;
  tool_calls?: ToolCall[];
}

interface Choice {
  index: number;
  message: ChoiceMessage;
}

interface Data {
  index: number;
  embedding: number[];
  object: string;
}

interface StreamingToolCall {
  id?: string;
  type?: 'function';
  index: number;
  function: {
    name?: string;
    arguments?: string;
  };
}

interface StreamChoice {
  index: number;
  delta: {
    content?: string;
    tool_calls?: StreamingToolCall[];
    role: string;
  };
}

export interface EmbedRequest {
  model: string;
  input: EmbeddingInput;
}

export interface EmbedResponse {
  object: string;
  model: string;
  data: Data[];
  usage: Usage;
}

export interface ChatCompletionsRequest {
  model: string;
  messages: Message[];
  temperature?: number;
  max_completion_tokens?: number;
  tools?: Tool[];
  encrypt?: boolean;
}

export interface ChatCompletionsResponse {
  object: string;
  created: number;
  model: string;
  choices: Choice[];
  usage: Usage;
}

interface StreamChunk {
  object: 'chat.completion.chunk';
  choices: StreamChoice[];
}

interface FinalChunk {
  object: 'chat.completion.chunk';
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface HTTPError {
  detail: string;
}

interface GenericError {
  error: string;
}

interface PlatformHttpError {
  detail: {
    code: string;
    message: string;
  };
}

export function isPlatformHttpError(o: unknown): o is PlatformHttpError {
  return (
    typeof o === 'object' &&
    o !== null &&
    'detail' in o &&
    typeof o.detail === 'object' &&
    o.detail !== null &&
    'code' in o.detail &&
    typeof o.detail.code === 'string' &&
    'message' in o.detail &&
    typeof o.detail.message === 'string'
  );
}

export function getServerSentEventData(o: unknown): string {
  // Skips 'data: ' prefix
  const prefix = 'data: ';
  return (o as string).slice(prefix.length).trim();
}

export function isStreamChunk(o: unknown): o is StreamChunk {
  return typeof o === 'object' && o !== null && 'choices' in o && Array.isArray(o.choices);
}

export function isFinalChunk(o: unknown): o is FinalChunk {
  return (
    typeof o === 'object' &&
    o !== null &&
    'usage' in o &&
    typeof o.usage === 'object' &&
    o.usage !== null &&
    'prompt_tokens' in o.usage &&
    typeof o.usage.prompt_tokens === 'number'
  );
}

export function isHTTPError(o: unknown): o is HTTPError {
  return typeof o === 'object' && o !== null && 'detail' in o && typeof o.detail === 'string';
}

export function isGenericError(o: unknown): o is GenericError {
  return typeof o === 'object' && o !== null && 'error' in o && typeof o.error === 'string';
}
