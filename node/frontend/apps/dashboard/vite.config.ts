import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react-swc';
import { resolve } from 'path';
import { visualizer } from 'rollup-plugin-visualizer';
import viteCompression from 'vite-plugin-compression';
import { VitePWA } from 'vite-plugin-pwa';
import { createHash } from 'crypto';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  
  return {
    plugins: [
      react({
        jsxImportSource: '@emotion/react',
        plugins: [['@swc/plugin-emotion', {}]],
      }),
      viteCompression({
        algorithm: 'brotliCompress',
        ext: '.br',
        threshold: 1024,
      }),
      viteCompression({
        algorithm: 'gzip',
        ext: '.gz',
        threshold: 1024,
      }),
      // Module preload for critical paths
      {
        name: 'module-preload',
        transformIndexHtml(html) {
          return html.replace(
            '</head>',
            `<link rel="modulepreload" href="/src/main.tsx" />
            <link rel="modulepreload" href="/src/App.tsx" />
            <link rel="modulepreload" href="/src/providers/WebSocketProvider.tsx" />
            <link rel="modulepreload" href="/src/services/enhanced-websocket.ts" />
            </head>`
          );
        }
      },
      VitePWA({
        registerType: 'autoUpdate',
        includeAssets: ['favicon.ico', 'robots.txt', 'apple-touch-icon.png'],
        manifest: {
          name: 'Solana MEV Dashboard',
          short_name: 'MEV Dashboard',
          theme_color: '#000000',
          background_color: '#000000',
          display: 'standalone',
          scope: '/',
          start_url: '/',
          icons: [
            {
              src: 'pwa-192x192.png',
              sizes: '192x192',
              type: 'image/png',
            },
            {
              src: 'pwa-512x512.png',
              sizes: '512x512',
              type: 'image/png',
            },
          ],
        },
      }),
      mode === 'analyze' && visualizer({
        template: 'treemap',
        open: true,
        gzipSize: true,
        brotliSize: true,
        filename: 'analyze/bundle-analysis.html',
      }),
    ].filter(Boolean),
    
    resolve: {
      alias: {
        '@': resolve(__dirname, './src'),
        '@solana-mev/ui': resolve(__dirname, '../../packages/ui/src'),
        '@solana-mev/charts': resolve(__dirname, '../../packages/charts/src'),
        '@solana-mev/websocket': resolve(__dirname, '../../packages/websocket/src'),
        '@solana-mev/protobuf': resolve(__dirname, '../../packages/protobuf/src'),
        '@solana-mev/utils': resolve(__dirname, '../../packages/utils/src'),
      },
    },
    
    optimizeDeps: {
      include: [
        'react',
        'react-dom',
        '@tanstack/react-query',
        'zustand',
        'valtio',
        'immer',
        'd3',
        'recharts',
        'framer-motion',
      ],
      exclude: ['@solana-mev/websocket'],
      esbuildOptions: {
        target: 'es2022',
      },
    },
    
    build: {
      target: 'es2022',
      minify: 'terser',
      modulePreload: {
        polyfill: true
      },
      terserOptions: {
        compress: {
          drop_console: true,
          drop_debugger: true,
          pure_funcs: ['console.log', 'console.info'],
          passes: 3,
        },
        mangle: {
          safari10: true,
        },
        format: {
          comments: false,
        },
      },
      rollupOptions: {
        // Treeshaking optimizations
        treeshake: {
          moduleSideEffects: false,
          propertyReadSideEffects: false,
          tryCatchDeoptimization: false
        },
        output: {
          // Enhanced chunk splitting strategy
          manualChunks: (id) => {
            // Workers get their own chunks
            if (id.includes('worker')) {
              return 'workers';
            }
            if (id.includes('node_modules')) {
              if (id.includes('react') || id.includes('react-dom') || id.includes('react-router')) {
                return 'react-vendor';
              }
              if (id.includes('zustand') || id.includes('valtio') || id.includes('immer') || id.includes('@tanstack')) {
                return 'state-vendor';
              }
              if (id.includes('d3') || id.includes('recharts') || id.includes('victory')) {
                return 'chart-vendor';
              }
              if (id.includes('@radix-ui') || id.includes('framer-motion')) {
                return 'ui-vendor';
              }
              if (id.includes('@solana') || id.includes('viem') || id.includes('ethers')) {
                return 'blockchain-vendor';
              }
              // Separate heavy libraries
              if (id.includes('protobuf') || id.includes('msgpack')) {
                return 'serialization-vendor';
              }
              if (id.includes('lodash') || id.includes('date-fns')) {
                return 'utils-vendor';
              }
            }
            
            // Separate large components
            if (id.includes('src/components/mev')) {
              return 'mev-components';
            }
            if (id.includes('src/components/advanced')) {
              return 'advanced-components';
            }
          },
          chunkFileNames: (chunkInfo) => {
            const facadeModuleId = chunkInfo.facadeModuleId || '';
            if (facadeModuleId.includes('worker')) {
              return 'workers/[name].[hash].js';
            }
            return 'chunks/[name].[hash].js';
          },
          entryFileNames: 'entries/[name].[hash].js',
          assetFileNames: 'assets/[name].[hash].[ext]',
          // Generate consistent chunk names for better caching
          hashCharacters: 'hex' as const,
          // Compact output
          compact: true,
          // Modern JS output optimizations
          generatedCode: {
            arrowFunctions: true,
            constBindings: true,
            objectShorthand: true,
            preset: 'es2015'
          } as any
        }
      } as any,
      sourcemap: mode === 'development',
      reportCompressedSize: false,
      chunkSizeWarningLimit: 1000,
      assetsInlineLimit: 4096,
      // Optimize CSS
      cssCodeSplit: true,
      cssMinify: 'lightningcss' as any,
    },
    
    server: {
      port: 3000,
      strictPort: false,
      host: true,
      hmr: {
        overlay: true,
        protocol: 'ws',
      },
      watch: {
        usePolling: false,
      },
      cors: true,
      proxy: {
        '/api': {
          target: env.VITE_API_URL || 'http://localhost:8000',
          changeOrigin: true,
          secure: false,
        },
        '/ws': {
          target: env.VITE_WS_URL || 'ws://localhost:8001',
          ws: true,
          changeOrigin: true,
        },
      },
    },
    
    preview: {
      port: 3000,
      strictPort: false,
      host: true,
    },
    
    worker: {
      format: 'es',
      plugins: () => [react()],
    },
    
    define: {
      __DEV__: mode !== 'production',
      __PROD__: mode === 'production',
      'process.env.NODE_ENV': JSON.stringify(mode),
    },
  };
});