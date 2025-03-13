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

import { CachedMapping, CacheStorage } from './storage';

export class WebCacheStorage extends CacheStorage {
  private readonly CACHE_KEY = 'flwr-mdl-cache';

  protected async load(): Promise<CachedMapping | null> {
    await Promise.resolve();
    const data = localStorage.getItem(this.CACHE_KEY);
    if (data) {
      try {
        return JSON.parse(data) as CachedMapping;
      } catch {
        return null;
      }
    }
    return null;
  }

  protected async save(cache: CachedMapping): Promise<void> {
    await Promise.resolve();
    localStorage.setItem(this.CACHE_KEY, JSON.stringify(cache));
  }
}
