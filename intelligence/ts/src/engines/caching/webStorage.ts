import { CachedMapping, CacheStorage } from './storage';

export class WebCacheStorage extends CacheStorage {
  private readonly CACHE_KEY = 'flwr-mdl-cache';

  async loadCache(): Promise<CachedMapping | null> {
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

  async saveCache(cache: CachedMapping): Promise<void> {
    localStorage.setItem(this.CACHE_KEY, JSON.stringify(cache));
  }
}
