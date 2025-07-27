import { EmbeddingInput, Message, Tool, ToolCall } from '../../typing';

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

interface StreamChoice {
  index: number;
  delta: {
    content: string;
    role: string;
  };
}

interface Usage {
  total_duration: number; // time spent generating the response
  load_duration: number; // time spent in nanoseconds loading the model
  prompt_eval_count: number; // number of tokens in the prompt
  prompt_eval_duration: number; // time spent in nanoseconds evaluating the prompt
  eval_count: number; // number of tokens in the response
  eval_duration: number; // time in nanoseconds spent generating the response
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
