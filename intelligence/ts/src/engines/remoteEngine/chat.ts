import {
  ChatResponseResult,
  Embedding,
  Failure,
  FailureCode,
  Message,
  ResponseFormat,
  Result,
  StreamEvent,
} from '../../typing';
import { CryptographyHandler } from './cryptoHandler';
import { createChatRequestData, getHeaders, sendRequest } from './remoteUtils';
import {
  ChatCompletionsResponse,
  EmbedResponse,
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
  const requestData = createChatRequestData(
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
  const body = response.value.body;
  if (!body) {
    return {
      ok: false,
      failure: {
        code: FailureCode.ConnectionError,
        description: 'No response body received.',
      },
    };
  }
  try {
    const resultText = await processStream(body, cryptoHandler, encrypt, onStreamEvent, signal);
    return { ok: true, value: resultText };
  } catch (error) {
    return handleStreamError(error);
  }
}

async function processStream(
  body: ReadableStream<Uint8Array>,
  cryptoHandler: CryptographyHandler,
  encrypt: boolean,
  onStreamEvent?: (event: StreamEvent) => void,
  signal?: AbortSignal
): Promise<string> {
  const decoder = new TextDecoder('utf-8');
  const reader = body.getReader();
  let accumulated = '';
  let done = false;
  const abortListener = () => void reader.cancel();

  signal?.addEventListener('abort', abortListener);

  try {
    while (!done && !signal?.aborted) {
      const { done: streamDone, value } = await reader.read();
      done = streamDone;

      const text = decoder.decode(value, { stream: true });
      for (const part of splitJsonChunks(text)) {
        accumulated += await processChunk(part, cryptoHandler, encrypt, onStreamEvent);
      }
    }
    return accumulated;
  } finally {
    signal?.removeEventListener('abort', abortListener);
  }
}

function splitJsonChunks(text: string): string[] {
  return text.split(/(?<=})\s*(?={)/g).filter((s) => s.trim());
}

async function processChunk(
  chunk: string,
  cryptoHandler: CryptographyHandler,
  encrypt: boolean,
  onStreamEvent?: (event: StreamEvent) => void
): Promise<string> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(chunk);
  } catch {
    console.error('Invalid JSON chunk:', chunk);
    return '';
  }

  if (isStreamChunk(parsed)) {
    let text = '';
    for (const choice of parsed.choices) {
      const delta = choice.delta.content;
      if (!delta) continue;

      let content = delta;
      if (encrypt) {
        const decrypted = await cryptoHandler.decryptMessage(content);
        if (!decrypted.ok) {
          throw new StreamProcessingError(decrypted.failure);
        }
        content = decrypted.value;
      }

      onStreamEvent?.({ chunk: content });
      text += content;
    }
    return text;
  }

  if (isFinalChunk(parsed)) {
    return '';
  }

  if (isHTTPError(parsed)) {
    throw new StreamProcessingError({
      code: FailureCode.ConnectionError,
      description: parsed.detail,
    });
  }

  if (isGenericError(parsed)) {
    throw new StreamProcessingError({
      code: FailureCode.RemoteError,
      description: parsed.error,
    });
  }

  console.warn('Unknown chunk type', parsed);
  return '';
}

function handleStreamError(error: unknown): Result<string> {
  if (error instanceof StreamProcessingError) {
    return { ok: false, failure: error.failure };
  }

  if (
    (error instanceof Error && error.name === 'AbortError') ||
    (error instanceof DOMException && error.name === 'AbortError')
  ) {
    return {
      ok: false,
      failure: {
        code: FailureCode.RequestAborted,
        description: 'Request was aborted by the user.',
      },
    };
  }

  return { ok: false, failure: { code: FailureCode.RemoteError, description: String(error) } };
}

class StreamProcessingError extends Error {
  constructor(public failure: Failure) {
    super();
  }
}

export async function extractChatOutput(
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

export function extractEmbedOutput(response: EmbedResponse): Result<Embedding[]> {
  const embeddings = response.data.map((value) => value.embedding);

  return {
    ok: true,
    value: embeddings,
  };
}
