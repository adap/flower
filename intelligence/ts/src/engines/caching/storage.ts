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

export interface CachedEntry {
  engineModel: string;
  timestamp: number;
}

export interface CachedMapping {
  mapping: Record<string, CachedEntry>;
}

export class CacheStorage {
  protected async load(): Promise<CachedMapping | null> {
    await Promise.resolve();
    throw new Error('Method not implemented');
  }
  protected async save(_cache: CachedMapping): Promise<void> {
    await Promise.resolve();
    throw new Error('Method not implemented');
  }

  /**
   * Gets a model mapping if it was saved in the cache, otherwise null.
   */
  async getItem(key: string): Promise<CachedEntry | null> {
    let cache = await this.load();
    if (!cache) {
      cache = { mapping: {} };
    }

    if (key in cache.mapping) {
      return cache.mapping[key];
    }
    return null;
  }

  /**
   * Adds or updates a model mapping in the cache with the current timestamp.
   */
  async setItem(key: string, value: string): Promise<void> {
    const now = Date.now();
    let cache = await this.load();
    if (!cache) {
      cache = { mapping: {} };
    }
    cache.mapping[key] = { engineModel: value, timestamp: now };
    await this.save(cache);
  }

  /**
   * Removes a model from the cache.
   */
  async remove(model: string, engine: string): Promise<void> {
    const cache = await this.load();
    const key = `${model}_${engine}`;
    if (cache && key in cache.mapping) {
      const { [key]: removed, ...rest } = cache.mapping;
      cache.mapping = rest;
      await this.save(cache);
    }
  }
}
