import { defineConfig } from 'vite';
import path from 'path';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
  plugins: [
    wasm(),
    topLevelAwait()
  ],
  
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@lib': path.resolve(__dirname, './lib'),
      '@stores': path.resolve(__dirname, './stores'),
      '@workers': path.resolve(__dirname, './workers'),
      '@proto': path.resolve(__dirname, './proto'),
      '@components': path.resolve(__dirname, './components'),
      '@hooks': path.resolve(__dirname, './hooks'),
      '@utils': path.resolve(__dirname, './utils'),
    }
  },
  
  optimizeDeps: {
    include: [
      'protobufjs',
      'zstd-codec',
      'comlink',
      'valtio',
      'immer',
      '@solana/web3.js'
    ],
    exclude: [
      'wasm-feature-detect'
    ],
    esbuildOptions: {
      target: 'es2022'
    }
  },
  
  build: {
    target: 'es2022',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: false,
        drop_debugger: true,
        passes: 3,
        unsafe_math: true,
        unsafe_methods: true,
        unsafe_proto: true,
        unsafe_regexp: true
      },
      mangle: {
        safari10: true
      },
      format: {
        comments: false
      }
    },
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': ['valtio', 'immer', 'comlink'],
          'proto': ['protobufjs', 'zstd-codec'],
          'solana': ['@solana/web3.js']
        }
      }
    },
    chunkSizeWarningLimit: 1000,
    sourcemap: true,
    reportCompressedSize: false
  },
  
  worker: {
    format: 'es',
    plugins: [
      wasm(),
      topLevelAwait()
    ]
  },
  
  server: {
    port: 3000,
    host: true,
    cors: true,
    hmr: {
      overlay: true
    },
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin'
    }
  },
  
  preview: {
    port: 3001,
    host: true,
    cors: true,
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin'
    }
  }
});