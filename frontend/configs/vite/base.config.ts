import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';
import { resolve } from 'path';
import { visualizer } from 'rollup-plugin-visualizer';
import viteCompression from 'vite-plugin-compression';
import { VitePWA } from 'vite-plugin-pwa';

export const baseConfig = defineConfig({
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
    visualizer({
      template: 'treemap',
      open: false,
      gzipSize: true,
      brotliSize: true,
      filename: 'analyze/bundle-analysis.html',
    }),
  ],
  
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
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'state-vendor': ['zustand', 'valtio', 'immer', '@tanstack/react-query'],
          'chart-vendor': ['d3', 'recharts', 'victory'],
          'ui-vendor': ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu', 'framer-motion'],
          'blockchain-vendor': ['ethers', 'viem', '@solana/web3.js'],
        },
        chunkFileNames: (chunkInfo) => {
          const facadeModuleId = chunkInfo.facadeModuleId ? chunkInfo.facadeModuleId.split('/').pop() : 'chunk';
          return `chunks/[name]__${facadeModuleId}__[hash].js`;
        },
      },
    },
    sourcemap: true,
    reportCompressedSize: false,
    chunkSizeWarningLimit: 1000,
    assetsInlineLimit: 4096,
  },
  
  server: {
    port: 3000,
    strictPort: false,
    hmr: {
      overlay: true,
      protocol: 'ws',
    },
    watch: {
      usePolling: false,
    },
    cors: true,
  },
  
  preview: {
    port: 3000,
    strictPort: false,
  },
  
  worker: {
    format: 'es',
    plugins: [],
  },
  
  define: {
    __DEV__: process.env.NODE_ENV !== 'production',
    __PROD__: process.env.NODE_ENV === 'production',
  },
});