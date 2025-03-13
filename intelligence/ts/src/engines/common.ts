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

import { REMOTE_URL, SDK, VERSION } from '../constants';
import { isNode } from '../env';
import { FailureCode, Result } from '../typing';
import { NodeCacheStorage } from './caching/nodeStorage';
import { WebCacheStorage } from './caching/webStorage';
import { CacheStorage } from './caching/storage';

const STALE_TIMEOUT_MS = 24 * 60 * 60 * 1000; // 24 hours.

interface ModelResponse {
  is_supported: boolean;
  engine_model: string | undefined;
  model: string | undefined;
}

export const cacheStorage: CacheStorage = isNode ? new NodeCacheStorage() : new WebCacheStorage();

async function updateModel(model: string, engine: string): Promise<Result<string>> {
  try {
    const response = await fetch(`${REMOTE_URL}/v1/fetch-model-config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        engine_name: engine,
        sdk_version: VERSION,
        sdk: SDK,
      }),
    });

    if (!response.ok) {
      return {
        ok: false,
        failure: {
          code: FailureCode.RemoteError,
          description: `Remote call failed: ${response.statusText}`,
        },
      };
    }

    const data = (await response.json()) as ModelResponse;

    if (data.is_supported && data.engine_model) {
      await cacheStorage.setItem(`${model}_${engine}`, data.engine_model);
      return { ok: true, value: data.engine_model };
    } else {
      await cacheStorage.remove(model, engine);
      return {
        ok: false,
        failure: {
          code: FailureCode.UnsupportedModelError,
          description: `Model '${model}' is not supported on the ${engine} engine.`,
        },
      };
    }
  } catch (error) {
    return {
      ok: false,
      failure: {
        code: FailureCode.ConnectionError,
        description: `Error calling remote endpoint: ${String(error)}`,
      },
    };
  }
}

/**
 * Checks if a model is supported.
 * - If the model exists in the cache and its timestamp is fresh, return it.
 * - If it exists but is stale (older than 24 hours), trigger a background update (and return the stale mapping).
 * - If it does not exist, update synchronously.
 */
export async function getEngineModelName(model: string, engine: string): Promise<Result<string>> {
  const now = Date.now();

  const cachedEntry = await cacheStorage.getItem(`${model}_${engine}`);
  if (cachedEntry) {
    // If the cached entry is stale, trigger a background update.
    if (now - cachedEntry.timestamp > STALE_TIMEOUT_MS) {
      updateModel(model, engine).catch((err: unknown) => {
        console.warn(`Background update failed for model ${model}:`, String(err));
      });
    }
    // Return the (possibly stale) cached result.
    return { ok: true, value: cachedEntry.engineModel };
  } else {
    // Not in cache, call updateModel synchronously.
    return await updateModel(model, engine);
  }
}
