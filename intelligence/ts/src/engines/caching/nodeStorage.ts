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

import { join } from 'path';
import os from 'os';
import { mkdir, readFile, writeFile } from 'fs/promises';
import { CachedEntry, CachedMapping, CacheStorage } from './storage';

export class NodeCacheStorage extends CacheStorage {
  private filePath: string | undefined;

  private async getCacheFilePath(): Promise<string> {
    if (!this.filePath) {
      const homeDir = os.homedir();
      const cacheFolder = join(homeDir, '.flwr', 'cache');
      await mkdir(cacheFolder, { recursive: true });
      this.filePath = join(cacheFolder, 'intelligence-model-names.json');
    }
    return this.filePath;
  }

  private async load(): Promise<CachedMapping | null> {
    try {
      const filePath = await this.getCacheFilePath();
      const data = await readFile(filePath, 'utf-8');
      return JSON.parse(data) as CachedMapping;
    } catch (_) {
      return null;
    }
  }

  private async save(cache: CachedMapping): Promise<void> {
    try {
      const filePath = await this.getCacheFilePath();
      await writeFile(filePath, JSON.stringify(cache), 'utf-8');
    } catch (_) {
      return undefined;
    }
  }

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

  async setItem(key: string, value: string | null): Promise<void> {
    const now = Date.now();
    let cache = await this.load();
    if (!cache) {
      cache = { mapping: {} };
    }
    if (value) {
      cache.mapping[key] = { engineModel: value, timestamp: now };
      await this.save(cache);
    } else {
      if (key in cache.mapping) {
        const { [key]: removed, ...rest } = cache.mapping;
        cache.mapping = rest;
        await this.save(cache);
      }
    }
  }
}
