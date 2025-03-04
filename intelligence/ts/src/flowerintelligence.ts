// Copyright 2024 Flower Labs GmbH. All Rights Reserved.

import MODELS from './models.json';
import { Engine } from './engines/engine';
import { RemoteEngine } from './engines/remoteEngine';
import { TransformersEngine } from './engines/transformersEngine';
import {
  ChatOptions,
  ChatResponseResult,
  FailureCode,
  Message,
  Progress,
  ProviderMapping,
  Result,
} from './typing';
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

  #remoteEngine?: Engine;
  #localEngines: Record<string, Engine[]> = {
    onnx: isNode ? [new TransformersEngine()] : [],
    webllm: isNode ? [] : [new WebllmEngine()],
  };
  #availableModels: Record<string, ProviderMapping> = getModels();

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
    const engineResult = this.getEngine(model, false, false);
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

    const engineResult = this.getEngine(
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

  private getEngine(
    modelId: string,
    forceRemote: boolean,
    forceLocal: boolean
  ): Result<[Engine, string]> {
    const canonicalModelId = resolveModelAlias(modelId);
    const argsResult = this.validateArgs(canonicalModelId, forceRemote, forceLocal);
    if (!argsResult.ok) {
      return argsResult;
    }

    if (forceRemote) {
      return this.getOrCreateRemoteEngine(canonicalModelId);
    }

    if (this.canRunLocally(canonicalModelId)) {
      return this.chooseLocalEngine(canonicalModelId);
    }

    return this.getOrCreateRemoteEngine(canonicalModelId);
  }

  private chooseLocalEngine(modelId: string): Result<[Engine, string]> {
    const localProvider = Object.keys(this.#localEngines).find(
      (provider) =>
        this.#localEngines[provider].length > 0 && // check if the local engine exists
        provider in this.#availableModels[modelId]
    );

    if (!localProvider) {
      return {
        ok: false,
        failure: {
          description: `The model "${modelId}" is not available for local inference.`,
          code: FailureCode.NoLocalProviderError,
        },
      };
    }

    const translatedModelId = this.#availableModels[modelId][localProvider as 'onnx' | 'webllm'];

    if (!translatedModelId) {
      return {
        ok: false,
        failure: {
          description: `No match for "${modelId}" with provider "${localProvider}".`,
          code: FailureCode.NoLocalProviderError,
        },
      };
    }

    // Currently we just select the first compatible localEngine without further check
    return { ok: true, value: [this.#localEngines[localProvider][0], translatedModelId] };
  }

  private canRunLocally(modelId: string): boolean {
    if (!(modelId in this.#availableModels)) {
      return false;
    }

    const modelProviders = this.#availableModels[modelId];

    // Check if the model has any listed local providers
    const hasLocalProvider = Object.keys(this.#localEngines).some(
      (provider) => provider in modelProviders
    );

    // Placeholder for extra logic, e.g., hardware checks, resource availability
    // const hardwareCompatible = this.checkHardwareCompatibility(modelId);
    // const resourcesAvailable = this.isInferenceSlotAvailable();
    const extraLogic = true;

    return hasLocalProvider && extraLogic;
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

  private validateArgs(modelId: string, forceRemote: boolean, forceLocal: boolean): Result<void> {
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
    if (!Object.keys(this.#availableModels).includes(modelId)) {
      return {
        ok: false,
        failure: {
          description: `Only the following models are currently available: ${Object.keys(this.#availableModels).join(',')}.\nYou provided ${modelId}, which is not supported.`,
          code: FailureCode.UnknownModelError,
        },
      };
    }
    return { ok: true, value: undefined };
  }
}

export function getModels(): Record<string, ProviderMapping> {
  const relevantProviders = MODELS.languages.typescript;

  return Object.entries(MODELS.models).reduce<Record<string, ProviderMapping>>(
    (relevantModels, [modelId, modelData]) => {
      // Ensure modelData has a "providers" property; default to an empty object if not.
      const providersObj = modelData.providers;

      // Filter the providers to include only those that are relevant for TypeScript.
      const filteredProviders = Object.entries(providersObj).reduce<Partial<ProviderMapping>>(
        (acc, [provider, modelValue]) => {
          if (relevantProviders.includes(provider)) {
            acc[provider as keyof ProviderMapping] = modelValue;
          }
          return acc;
        },
        {}
      );
      relevantModels[modelId] = filteredProviders as ProviderMapping;
      return relevantModels;
    },
    {}
  );
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
