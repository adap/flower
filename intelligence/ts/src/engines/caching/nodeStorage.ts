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
