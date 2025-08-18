import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';
import { VitePWA } from 'vite-plugin-pwa';
import viteCompression from 'vite-plugin-compression';
import replace from '@rollup/plugin-replace';

export default defineConfig({
  plugins: [
    react({
      babel: {
        plugins: [
          ['@babel/plugin-transform-react-jsx', { runtime: 'automatic' }]
        ]
      }
    }),
    wasm(),
    topLevelAwait(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'robots.txt'],
      manifest: {
        name: 'MEV Dashboard Ultra',
        short_name: 'MEV Ultra',
        theme_color: '#0a0a0a',
        background_color: '#0a0a0a',
        display: 'standalone',
        orientation: 'landscape',
        icons: [
          {
            src: 'icon-192.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: 'icon-512.png',
            sizes: '512x512',
            type: 'image/png'
          }
        ]
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,wasm}'],
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/api\./,
            handler: 'NetworkFirst',
            options: {
              cacheName: 'api-cache',
              expiration: {
                maxEntries: 100,
                maxAgeSeconds: 60
              },
              cacheableResponse: {
                statuses: [0, 200]
              }
            }
          }
        ]
      }
    }),
    viteCompression({
      algorithm: 'brotliCompress',
      threshold: 256,
      compressionOptions: {
        level: 11
      }
    }),
    replace({
      'process.env.NODE_ENV': JSON.stringify('production'),
      __DEV__: false,
      preventAssignment: true
    })
  ],
  build: {
    target: 'esnext',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
        pure_funcs: ['console.log', 'console.info', 'console.debug'],
        passes: 2,
        // Disable unsafe optimizations that break React
        unsafe: false,
        unsafe_comps: false,
        unsafe_Function: false,
        unsafe_math: false,
        unsafe_methods: false,
        unsafe_proto: false,
        unsafe_regexp: false,
        unsafe_symbols: false,
        unsafe_undefined: false,
        // Keep React internals intact
        keep_fnames: true,
        keep_classnames: true
      },
      mangle: {
        // Don't mangle properties - this breaks React internals
        properties: false,
        // Keep important names
        reserved: ['React', 'ReactDOM', 'ReactCurrentOwner', '__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED']
      },
      format: {
        comments: false
      }
    },
    rollupOptions: {
      output: {
        manualChunks: (id) => {
          // Keep React and React-DOM together to avoid circular dependencies
          if (id.includes('node_modules/react')) {
            return 'vendor-react';
          }
          // Bundle Three.js and related packages together
          if (id.includes('three') || id.includes('@react-three')) {
            return 'vendor-3d';
          }
          // Data visualization libraries
          if (id.includes('d3') || id.includes('pixi') || id.includes('@pixi')) {
            return 'vendor-viz';
          }
          // Protocol and compression libraries
          if (id.includes('protobuf') || id.includes('msgpack') || id.includes('lz4')) {
            return 'vendor-proto';
          }
          // State management and utilities
          if (id.includes('valtio') || id.includes('zustand') || id.includes('comlink') || id.includes('idb')) {
            return 'vendor-utils';
          }
          // Let Vite handle other dependencies
          return undefined;
        },
        chunkFileNames: 'assets/[name]-[hash].js',
        entryFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash].[ext]'
      }
    },
    sourcemap: false,
    reportCompressedSize: false,
    chunkSizeWarningLimit: 300,
    assetsInlineLimit: 8192
  },
  optimizeDeps: {
    include: [
      'react', 
      'react-dom', 
      'valtio', 
      'protobufjs',
      'three',
      '@react-three/fiber',
      '@react-three/drei',
      '@react-spring/three',
      'zustand'
    ],
    exclude: ['@rollup/rollup-linux-x64-gnu'],
    esbuildOptions: {
      target: 'esnext',
      define: {
        global: 'globalThis'
      }
    }
  },
  server: {
    port: 3001,
    host: true,
    cors: true,
    hmr: {
      overlay: false
    },
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY',
      'X-XSS-Protection': '1; mode=block'
    }
  },
  worker: {
    format: 'es',
    plugins: () => [wasm(), topLevelAwait()]
  },
  resolve: {
    alias: {
      '@': '/src',
      '@components': '/src/components',
      '@lib': '/src/lib',
      '@workers': '/src/workers',
      '@stores': '/src/stores',
      '@proto': '/src/proto'
    }
  }
});