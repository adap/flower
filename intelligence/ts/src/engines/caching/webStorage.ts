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

import { CachedEntry, CacheStorage } from './storage';

export class WebCacheStorage extends CacheStorage {
  private readonly CACHE_KEY_PREFIX = 'flwr-mdl-cache';

  async getItem(key: string): Promise<CachedEntry | null> {
    await Promise.resolve();
    const data = localStorage.getItem(`${this.CACHE_KEY_PREFIX}-${key}`);
    if (data) {
      try {
        return JSON.parse(data) as CachedEntry;
      } catch {
        return null;
      }
    }
    return null;
  }
  async setItem(key: string, value: string | null): Promise<void> {
    await Promise.resolve();
    if (value) {
      localStorage.setItem(
        `${this.CACHE_KEY_PREFIX}-${key}`,
        JSON.stringify({ engineModel: value, timestamp: Date.now() })
      );
    } else {
      localStorage.removeItem(`${this.CACHE_KEY_PREFIX}-${key}`);
    }
  }
}
