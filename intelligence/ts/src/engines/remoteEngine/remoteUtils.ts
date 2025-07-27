import { FailureCode, Message, ResponseFormat, Result, Tool } from '../../typing';
import { ChatCompletionsRequest } from './typing';

export function createRequestData(
  messages: Message[],
  model: string,
  temperature?: number,
  topP?: number,
  maxCompletionTokens?: number,
  responseFormat?: ResponseFormat,
  stream?: boolean,
  tools?: Tool[],
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
    ...(encrypt && { encrypt, encryption_id: encryptionId }),
  };
}

export async function sendRequest(
  requestData: ChatCompletionsRequest,
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
  };
}
