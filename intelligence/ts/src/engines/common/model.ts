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

import { REMOTE_URL, SDK, VERSION } from '../../constants';
import { isNode } from '../../env';
import { FailureCode, Result } from '../../typing';
import { CacheStorage, NodeCacheStorage, WebCacheStorage } from './storage';

interface ModelConfigResponse {
  is_supported: boolean;
  engine_model?: string;
  model?: string;
  vram?: number;
}

const cacheStorage: CacheStorage = isNode ? new NodeCacheStorage() : new WebCacheStorage();

async function updateModelConfig(model: string, engine: string): Promise<Result<ModelConfig>> {
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

    const data = (await response.json()) as ModelConfigResponse;

    if (data.is_supported && data.engine_model) {
      await cacheStorage.setItem(
        `${model}_${engine}`,
        JSON.stringify({ name: data.engine_model, vram: data.vram })
      );
      return { ok: true, value: { name: data.engine_model, vram: data.vram } };
    } else {
      await cacheStorage.setItem(`${model}_${engine}`, null);
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

interface ModelConfig {
  name: string;
  vram?: number;
}

function isModelConfig(obj: unknown): obj is ModelConfig {
  return !(
    obj === null ||
    typeof obj !== 'object' ||
    !('name' in obj) ||
    typeof obj.name !== 'string' ||
    ('vram' in obj && typeof obj.vram !== 'number')
  );
}

/**
 * Checks if a model is supported.
 * - If the model exists in the cache and its timestamp is fresh, return it.
 * - If it exists but is stale (older than 24 hours), trigger a background update (and return the stale mapping).
 * - If it does not exist, update synchronously.
 */
export async function getEngineModelConfig(
  model: string,
  engine: string
): Promise<Result<ModelConfig>> {
  const now = Date.now();

  const cachedEntry = await cacheStorage.getItem(`${model}_${engine}`);
  if (cachedEntry) {
    const lastUpdateDay = new Date(cachedEntry.lastUpdate).toDateString();
    const currentDay = new Date(now).toDateString();

    // If the cached entry was updated on a different day, trigger a background update.
    if (lastUpdateDay !== currentDay) {
      updateModelConfig(model, engine).catch((err: unknown) => {
        console.warn(`Background update failed for model ${model}:`, String(err));
      });
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(cachedEntry.value);
    } catch {
      parsed = null;
    }

    // Return the (possibly stale) cached result.
    return parsed !== null && isModelConfig(parsed)
      ? { ok: true, value: parsed }
      : {
          ok: false,
          failure: {
            code: FailureCode.LocalError,
            description: 'Wrong cache format, try deleting existing cache',
          },
        };
  } else {
    // Not in cache, call updateModel synchronously.
    return await updateModelConfig(model, engine);
  }
}
