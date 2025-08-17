/**
 * API Configuration
 * Central configuration for all backend service endpoints
 */

export const API_CONFIG = {
  // Base URLs
  API_BASE_URL: process.env.VITE_API_BASE_URL || 'http://localhost:8085',
  WS_BASE_URL: process.env.VITE_WS_BASE_URL || 'ws://localhost:8085',
  
  // API Endpoints
  endpoints: {
    // Node Metrics
    node: {
      metrics: '/api/node/metrics',
      health: '/api/node/health',
      status: '/api/node/status',
      config: '/api/node/config',
    },
    
    // Scrapper
    scrapper: {
      datasets: '/api/scrapper/datasets',
      collect: '/api/scrapper/collect',
      train: '/api/scrapper/train',
      models: '/api/scrapper/models',
      export: '/api/scrapper/export',
      import: '/api/scrapper/import',
      config: '/api/scrapper/config',
    },
    
    // MEV
    mev: {
      opportunities: '/api/mev/opportunities',
      bundles: '/api/mev/bundles',
      transactions: '/api/mev/transactions',
    },
  },
  
  // WebSocket Endpoints
  ws: {
    nodeMetrics: '/ws/node-metrics',
    scrapperProgress: '/ws/scrapper-progress',
    mevStream: '/ws/mev-stream',
  },
  
  // Request Configuration
  request: {
    timeout: 30000, // 30 seconds
    retries: 3,
    retryDelay: 1000, // 1 second
    headers: {
      'Content-Type': 'application/json',
    },
  },
  
  // WebSocket Configuration
  websocket: {
    reconnect: true,
    reconnectAttempts: 10,
    reconnectDelay: 1000,
    reconnectDelayMax: 5000,
    heartbeatInterval: 30000, // 30 seconds
    messageTimeout: 5000,
  },
} as const;

export type ApiConfig = typeof API_CONFIG;