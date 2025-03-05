import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, 'src/background.ts'),
      name: 'ExtensionBackground',
      fileName: 'background',
    },
    minify: false,
    sourcemap: 'inline',
    outDir: 'dist/background',
    rollupOptions: {},
  },
});
