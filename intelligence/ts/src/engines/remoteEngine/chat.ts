import {
  ChatResponseResult,
  FailureCode,
  Message,
  ResponseFormat,
  Result,
  StreamEvent,
} from '../../typing';
import { CryptographyHandler } from './cryptoHandler';
import { createRequestData, getHeaders, sendRequest } from './remoteUtils';
import {
  ChatCompletionsResponse,
  isFinalChunk,
  isGenericError,
  isHTTPError,
  isStreamChunk,
} from './typing';

export async function chatStream(
  baseUrl: string,
  cryptoHandler: CryptographyHandler,
  apiKey: string,
  messages: Message[],
  model: string,
  encrypt: boolean,
  temperature?: number,
  topP?: number,
  maxCompletionTokens?: number,
  responseFormat?: ResponseFormat,
  onStreamEvent?: (event: StreamEvent) => void,
  signal?: AbortSignal
): Promise<Result<string>> {
  const requestData = createRequestData(
    messages,
    model,
    temperature,
    topP,
    maxCompletionTokens,
    responseFormat,
    true,
    undefined,
    encrypt,
    cryptoHandler.encryptionId
  );
  const response = await sendRequest(
    requestData,
    '/v1/chat/completions',
    baseUrl,
    getHeaders(apiKey),
    signal
  );

  if (!response.ok) return response;

  const reader = response.value.body?.getReader();
  const decoder = new TextDecoder('utf-8');
  let accumulatedResponse = '';

  if (signal) {
    signal.addEventListener('abort', () => {
      void reader?.cancel();
    });
  }

  while (reader) {
    if (signal?.aborted) {
      break;
    }
    let result;
    try {
      result = await reader.read();
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') {
        break;
      }
      if (err instanceof DOMException && err.name === 'AbortError') {
        break;
      }
      return {
        ok: false,
        failure: {
          code: FailureCode.RequestAborted,
          description: 'Request was aborted by the user.',
        },
      };
    }
    const { done, value } = result;
    if (done) break;

    const text = decoder.decode(value, { stream: true });
    const parts = text.split(/(?<=})\s*(?={)/g);

    for (const part of parts) {
      if (!part.trim()) continue;

      let parsed: unknown;
      try {
        parsed = JSON.parse(part);
      } catch (err) {
        console.error('Invalid JSON chunk:', part, err);
        continue;
      }

      if (isStreamChunk(parsed)) {
        for (const choice of parsed.choices) {
          const delta = choice.delta.content;
          if (!delta) continue;

          let content = delta;
          if (encrypt) {
            const decrypted = await cryptoHandler.decryptMessage(delta);
            if (!decrypted.ok) return decrypted;
            content = decrypted.value;
          }

          onStreamEvent?.({ chunk: content });
          accumulatedResponse += content;
        }
      } else if (isFinalChunk(parsed)) {
        break;
      } else if (isHTTPError(parsed)) {
        return {
          ok: false,
          failure: { code: FailureCode.ConnectionError, description: parsed.detail },
        };
      } else if (isGenericError(parsed)) {
        return {
          ok: false,
          failure: { code: FailureCode.RemoteError, description: parsed.error },
        };
      } else {
        console.warn('Unknown stream shape:', parsed);
      }
    }
  }

  return { ok: true, value: accumulatedResponse };
}

export async function extractOutput(
  response: ChatCompletionsResponse,
  encrypt: boolean,
  cryptoHandler: CryptographyHandler
): Promise<ChatResponseResult> {
  const message = response.choices[0].message;
  let content: string;
  if (encrypt) {
    const decryptedResult = await cryptoHandler.decryptMessage(message.content ?? '');
    if (!decryptedResult.ok) {
      return decryptedResult;
    }
    content = decryptedResult.value;
  } else {
    content = message.content ?? '';
  }
  const toolCalls = message.tool_calls;

  return {
    ok: true,
    message: {
      role: message.role as Message['role'],
      content: content,
      ...(toolCalls && { toolCalls: toolCalls }),
    },
  };
}
