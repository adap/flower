// Copyright 2025 Flower Labs GmbH. All Rights Reserved.

import MODELS from './models.json';
import { Engine } from './engines/engine';
import { RemoteEngine } from './engines/remoteEngine';
import { TransformersEngine } from './engines/transformersEngine';
import { ChatOptions, ChatResponseResult, FailureCode, Message, Progress, Result } from './typing';
import { WebllmEngine } from './engines/webllmEngine';
import { DEFAULT_MODEL } from './constants';

/* eslint-disable @typescript-eslint/no-unnecessary-condition */
const isNode = typeof process !== 'undefined' && process.versions.node != null;
/* eslint-enable @typescript-eslint/no-unnecessary-condition */

/**
 * Class representing the core intelligence service for Flower Labs.
 * It facilitates chat, generation, and summarization tasks, with the option of using a
 * local or remote engine based on configurations and availability.
 */
export class FlowerIntelligence {
  static #instance: FlowerIntelligence | null = null;
  static #remoteHandoff = false;
  static #apiKey?: string;

  #remoteEngine?: RemoteEngine;
  #availableLocalEngines: Engine[] = isNode ? [new TransformersEngine()] : [new WebllmEngine()];

  /**
   * Get the initialized FlowerIntelligence instance.
   * Initializes the instance if it doesn't exist.
   * @returns The initialized FlowerIntelligence instance.
   */
  public static get instance(): FlowerIntelligence {
    if (!this.#instance) {
      this.#instance = new FlowerIntelligence();
    }
    return this.#instance;
  }
  /**
   * Sets the remote handoff boolean.
   * @param remoteHandoffValue - If true, the processing might be done on a secure
   remote server instead of locally (if resources are lacking).
   */
  public set remoteHandoff(remoteHandoffValue: boolean) {
    FlowerIntelligence.#remoteHandoff = remoteHandoffValue;
  }

  /**
   * Gets the current remote handoff status.
   * @returns boolean - the value of the remote handoff variable
   */
  public get remoteHandoff() {
    return FlowerIntelligence.#remoteHandoff;
  }

  /**
   * Set apiKey for FlowerIntelligence.
   */
  public set apiKey(apiKey: string) {
    FlowerIntelligence.#apiKey = apiKey;
  }

  /**
   * Downloads and loads a model into memory.
   * @param model Model name to use for the chat.
   * @param callback A callback to handle the progress of the download
   * @returns A {@link Result} containing either a {@link Failure} (containing `code: number` and `description: string`) if `ok` is false or a value of `void`, if `ok` is true (meaning the loading was successful).
   */
  async fetchModel(model: string, callback: (progress: Progress) => void): Promise<Result<void>> {
    const engineResult = await this.getEngine(model, false, false);
    if (!engineResult.ok) {
      return engineResult;
    } else {
      const [engine, modelId] = engineResult.value;
      return await engine.fetchModel(modelId, callback);
    }
  }

  // Overload for string input with an optional options object
  async chat(input: string, options?: ChatOptions): Promise<ChatResponseResult>;

  // Overload for a single object that includes messages along with other options
  async chat(options: ChatOptions & { messages: Message[] }): Promise<ChatResponseResult>;

  /**
   * Conducts a chat interaction using the specified model and options.
   *
   * This method can be invoked in one of two ways:
   *
   * 1. With a string input (plus an optional options object). In this case the string
   *    is automatically wrapped as a single message with role 'user'.
   *
   *    Example:
   *    ```ts
   *    fi.chat("Why is the sky blue?", { temperature: 0.7 });
   *    ```
   *
   * 2. With a single object that includes a {@link Message} array along with additional options.
   *
   *    Example:
   *    ```ts
   *    fi.chat({
   *      messages: [{ role: 'user', content: "Why is the sky blue?" }],
   *      model: "meta/llama3.2-1b"
   *    });
   *    ```
   *
   * @param inputOrOptions - Either a string input or a {@link ChatOptions} object that must include a `messages` array.
   * @param maybeOptions - An optional {@link ChatOptions} object (used only when the first parameter is a string).
   * @returns A Promise that resolves to a {@link ChatResponseResult}. On success, the result contains the
   *          message reply and optionally any tool call details; on failure, it includes an error code and description.
   */
  async chat(
    inputOrOptions: string | (ChatOptions & { messages: Message[] }),
    maybeOptions?: ChatOptions
  ): Promise<ChatResponseResult> {
    const chatResult = await this.internalChat(inputOrOptions, maybeOptions);
    if (!chatResult.ok) {
      if (
        chatResult.failure.code === FailureCode.LocalEngineChatError &&
        this.remoteHandoff &&
        this.apiKey
      ) {
        return await this.internalChat(inputOrOptions, { ...maybeOptions, forceRemote: true });
      }
    }
    return chatResult;
  }

  private async internalChat(
    inputOrOptions: string | (ChatOptions & { messages: Message[] }),
    maybeOptions?: ChatOptions
  ): Promise<ChatResponseResult> {
    let options: ChatOptions;
    let messages: Message[];

    if (typeof inputOrOptions === 'string') {
      options = maybeOptions ?? {};
      messages = [{ role: 'user', content: inputOrOptions }];
    } else {
      ({ messages, ...options } = inputOrOptions);
    }

    const engineResult = await this.getEngine(
      options.model ?? DEFAULT_MODEL,
      options.forceRemote ?? false,
      options.forceLocal ?? false
    );

    if (!engineResult.ok) {
      return engineResult;
    }

    const [engine, modelId] = engineResult.value;
    return await engine.chat(
      messages,
      modelId,
      options.temperature,
      options.maxCompletionTokens,
      options.stream,
      options.onStreamEvent,
      options.tools,
      options.encrypt
    );
  }

  private async getEngine(
    modelId: string,
    forceRemote: boolean,
    forceLocal: boolean
  ): Promise<Result<[Engine, string]>> {
    const canonicalModelId = resolveModelAlias(modelId);
    const argsResult = this.validateArgs(forceRemote, forceLocal);
    if (!argsResult.ok) {
      return argsResult;
    }

    if (forceRemote) {
      return this.getOrCreateRemoteEngine(canonicalModelId);
    }

    const localEngineResult = await this.chooseLocalEngine(canonicalModelId);
    if (localEngineResult.ok) {
      return localEngineResult;
    }

    return this.getOrCreateRemoteEngine(canonicalModelId);
  }

  private async chooseLocalEngine(modelId: string): Promise<Result<[Engine, string]>> {
    const compatibleEngines = (
      await Promise.all(
        this.#availableLocalEngines.map(async (engine) => {
          const supportedResult = await engine.isSupported(modelId);
          return supportedResult.ok ? [engine, supportedResult.value] : null;
        })
      )
    ).filter((item): item is [Engine, string] => item !== null);

    if (compatibleEngines.length > 0) {
      // Currently we just select the first compatible localEngine without further check
      return { ok: true, value: compatibleEngines[0] };
    } else {
      return {
        ok: false,
        failure: {
          code: FailureCode.NoLocalProviderError,
          description: `No available local engine for ${modelId}.`,
        },
      };
    }
  }

  private getOrCreateRemoteEngine(modelId: string): Result<[Engine, string]> {
    if (!FlowerIntelligence.#remoteHandoff) {
      return {
        ok: false,
        failure: {
          description: 'To use remote inference, remote handoff must be allowed.',
          code: FailureCode.InvalidRemoteConfigError,
        },
      };
    }
    if (!FlowerIntelligence.#apiKey) {
      return {
        ok: false,
        failure: {
          description: 'To use remote inference, a valid API key must be set.',
          code: FailureCode.InvalidRemoteConfigError,
        },
      };
    }
    this.#remoteEngine = this.#remoteEngine ?? new RemoteEngine(FlowerIntelligence.#apiKey);
    return { ok: true, value: [this.#remoteEngine, modelId] };
  }

  private validateArgs(forceRemote: boolean, forceLocal: boolean): Result<void> {
    if (forceLocal && forceRemote) {
      return {
        ok: false,
        failure: {
          description:
            'The `forceLocal` and `forceRemote` options cannot be true at the same time.',
          code: FailureCode.InvalidArgumentsError,
        },
      };
    }
    return { ok: true, value: undefined };
  }
}

/**
 * Resolves a user-provided model ID (or alias) into a canonical model ID.
 * If the alias doesn't exist and the model ID isn't found in the canonical list,
 * an error is thrown.
 */
function resolveModelAlias(inputModelId: string): string {
  // Cast aliases to a record so we can index it with an arbitrary string.
  const aliases = MODELS.aliases as Record<string, string>;

  if (inputModelId in aliases) {
    return aliases[inputModelId];
  }

  return inputModelId;
}
