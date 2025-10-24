import { coverageConfigDefaults, defaultExclude } from 'vitest/config';
import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
  test: {
    coverage: {
      exclude: ['docs/**', 'create/**', 'e2e/**', 'examples/**', ...coverageConfigDefaults.exclude],
      reporter: ['lcov'],
    },
    exclude: [...defaultExclude],
  },
  build: {
    lib: {
      entry: path.resolve(__dirname, 'src/index.ts'),
      name: 'FlowerIntelligence',
      formats: ['es', 'cjs'],
      fileName: (format) => `flowerintelligence.${format}.js`,
    },
    rollupOptions: {
      external: [
        '@huggingface/transformers',
        '@mlc-ai/web-llm',
        'crypto',
        'fs',
        'fs/promises',
        'path',
        'os',
      ],
    },
  },
});
