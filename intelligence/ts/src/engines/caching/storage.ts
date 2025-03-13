export interface CachedEntry {
  engineModel: string;
  timestamp: number;
}

export interface CachedMapping {
  mapping: Record<string, CachedEntry>;
}

export abstract class CacheStorage {
  abstract loadCache(): Promise<CachedMapping | null>;
  abstract saveCache(cache: CachedMapping): Promise<void>;
}
