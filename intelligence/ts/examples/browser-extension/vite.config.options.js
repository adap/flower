import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, 'src/options.ts'),
      name: 'ExtensionOptions',
      fileName: 'options',
    },
    minify: false,
    sourcemap: 'inline',
    outDir: 'dist/options',
    rollupOptions: {},
  },
});
