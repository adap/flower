// Copyright 2025 Flower Labs GmbH. All Rights Reserved.

import {
  type ChatCompletionMessageParam,
  CreateMLCEngine,
  type InitProgressReport,
  type MLCEngineInterface,
} from '@mlc-ai/web-llm';
import {
  ChatResponseResult,
  FailureCode,
  Message,
  Progress,
  ProviderMapping,
  Result,
  StreamEvent,
  Tool,
} from '../typing';
import MODELS from '../models.json';
import { BaseEngine } from './engine';

async function runQuery(
  engine: MLCEngineInterface,
  messages: Message[],
  stream?: boolean,
  onStreamEvent?: (event: StreamEvent) => void,
  temperature?: number,
  maxTokens?: number
) {
  if (stream && onStreamEvent) {
    const reply = await engine.chat.completions.create({
      stream: true,
      messages: messages as ChatCompletionMessageParam[],
      temperature,
      max_tokens: maxTokens,
    });
    for await (const chunk of reply) {
      onStreamEvent({ chunk: chunk.choices[0]?.delta?.content ?? '' });
    }
    return await engine.getMessage();
  } else {
    const reply = await engine.chat.completions.create({
      messages: messages as ChatCompletionMessageParam[],
      temperature,
      max_tokens: maxTokens,
    });
    return reply.choices[0].message.content ?? '';
  }
}

export class WebllmEngine extends BaseEngine {
  #loadedEngines: Record<string, MLCEngineInterface> = {};
  private models = this.getModels();

  async chat(
    messages: Message[],
    model: string,
    temperature?: number,
    maxCompletionTokens?: number,
    stream?: boolean,
    onStreamEvent?: (event: StreamEvent) => void,
    _tools?: Tool[]
  ): Promise<ChatResponseResult> {
    try {
      if (!(model in this.#loadedEngines)) {
        this.#loadedEngines.model = await CreateMLCEngine(
          model,
          {},
          {
            context_window_size: 2048,
          }
        );
      }
      const result = await runQuery(
        this.#loadedEngines.model,
        messages,
        stream,
        onStreamEvent,
        temperature,
        maxCompletionTokens
      );
      return {
        ok: true,
        message: {
          role: 'assistant',
          content: result,
        },
      };
    } catch (error) {
      return {
        ok: false,
        failure: {
          code: FailureCode.LocalEngineChatError,
          description: `WebLLM engine failed with: ${String(error)}`,
        },
      };
    }
  }

  async fetchModel(model: string, callback: (progress: Progress) => void): Promise<Result<void>> {
    try {
      if (!(model in this.#loadedEngines)) {
        this.#loadedEngines.model = await CreateMLCEngine(
          model,
          {
            initProgressCallback: (report: InitProgressReport) => {
              callback({ percentage: report.progress, description: report.text });
            },
          },
          {
            context_window_size: 2048,
          }
        );
      }
      return { ok: true, value: undefined };
    } catch (error) {
      return {
        ok: false,
        failure: { code: FailureCode.LocalEngineFetchError, description: String(error) },
      };
    }
  }

  async isSupported(model: string): Promise<Result<string>> {
    if (model in this.models) {
      return { ok: true, value: this.models[model] };
    }
    return {
      ok: false,
      failure: {
        code: FailureCode.UnsupportedModelError,
        description: `Model '${model}' is not supported on the WebLLM engine.`,
      },
    };
  }

  private getModels(): Record<string, string> {
    return Object.entries(MODELS.models).reduce<Record<string, string>>(
      (acc, [modelId, modelData]) => {
        // Cast modelData to an object with a "providers" property.
        const providers = (modelData as { providers?: ProviderMapping }).providers;
        if (providers?.webllm) {
          acc[modelId] = providers.webllm;
        }
        return acc;
      },
      {}
    );
  }
}
