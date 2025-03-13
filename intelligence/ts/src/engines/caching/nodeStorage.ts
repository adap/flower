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
import { CachedMapping, CacheStorage } from './storage';

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

  async loadCache(): Promise<CachedMapping | null> {
    try {
      const filePath = await this.getCacheFilePath();
      const data = await readFile(filePath, 'utf-8');
      return JSON.parse(data) as CachedMapping;
    } catch (_) {
      return null;
    }
  }

  async saveCache(cache: CachedMapping): Promise<void> {
    const filePath = await this.getCacheFilePath();
    await writeFile(filePath, JSON.stringify(cache), 'utf-8');
  }
}
