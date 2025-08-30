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

import { SDK, VERSION } from '../../constants';
import {
  EmbeddingInput,
  FailureCode,
  Message,
  ResponseFormat,
  Result,
  Tool,
  ToolChoice,
} from '../../typing';
import { ChatCompletionsRequest, EmbedRequest } from './typing';

export function createChatRequestData(
  messages: Message[],
  model: string,
  temperature?: number,
  topP?: number,
  maxCompletionTokens?: number,
  responseFormat?: ResponseFormat,
  stream?: boolean,
  tools?: Tool[],
  toolChoice?: ToolChoice,
  encrypt?: boolean,
  encryptionId: string | null = null
): ChatCompletionsRequest {
  return {
    model,
    messages,
    ...(temperature && { temperature }),
    ...(topP && { top_p: topP }),
    ...(maxCompletionTokens && {
      max_completion_tokens: maxCompletionTokens,
    }),
    ...(responseFormat && { response_format: responseFormat }),
    ...(stream && { stream }),
    ...(tools && { tools }),
    ...(toolChoice && { tool_choice: toolChoice }),
    ...(encrypt && { encrypt, encryption_id: encryptionId }),
  };
}

export function createEmbedRequestData(input: EmbeddingInput, model: string): EmbedRequest {
  return {
    model,
    input,
  };
}

export async function sendRequest(
  requestData: ChatCompletionsRequest | EmbedRequest,
  endpoint: string,
  baseUrl: string,
  headers: Record<string, string>,
  signal?: AbortSignal
): Promise<Result<Response>> {
  let response: Response;
  try {
    response = await fetch(`${baseUrl}${endpoint}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(requestData),
      signal,
    });
  } catch (err: unknown) {
    // Did the user abort?
    if (
      (err instanceof DOMException && err.name === 'AbortError') ||
      (err instanceof Error && err.name === 'AbortError')
    ) {
      return {
        ok: false,
        failure: {
          code: FailureCode.RequestAborted,
          description: 'Request was aborted by the user.',
        },
      };
    }
    return {
      ok: false,
      failure: {
        code: FailureCode.RemoteError,
        description: String(err),
      },
    };
  }

  if (!response.ok) {
    let code = FailureCode.RemoteError;
    switch (response.status) {
      case 401:
      case 403:
      case 407:
        code = FailureCode.AuthenticationError;
        break;
      case 404:
      case 502:
      case 503:
        code = FailureCode.UnavailableError;
        break;
      case 408:
      case 504:
        code = FailureCode.TimeoutError;
        break;

      default:
        break;
    }
    return {
      ok: false,
      failure: { code, description: `${String(response.status)}: ${response.statusText}` },
    };
  }

  return { ok: true, value: response };
}

export function getHeaders(apiKey: string) {
  return {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${apiKey}`,
    'FI-SDK-Type': SDK,
    'FI-SDK-Version': VERSION,
  };
}
