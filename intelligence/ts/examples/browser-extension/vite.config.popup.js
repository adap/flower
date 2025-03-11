import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, 'src/popup.ts'),
      name: 'ExtensionPopUp',
      fileName: 'popup',
    },
    minify: false,
    sourcemap: 'inline',
    outDir: 'dist/popup',
    rollupOptions: {},
  },
});
