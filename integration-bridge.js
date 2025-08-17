/**
 * Elite MEV Detection Integration Bridge
 * Connects Solana transaction stream to detection models and ClickHouse
 * Target: 200k+ events/sec processing with sub-slot detection
 */

const { Connection, PublicKey } = require('@solana/web3.js');
const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const WebSocket = require('ws');
const http = require('http');
const crypto = require('crypto');
const axios = require('axios');

// Configuration
const CONFIG = {
    SOLANA_RPC: process.env.SOLANA_RPC || 'https://api.mainnet-beta.solana.com',
    SOLANA_WS: process.env.SOLANA_WS || 'wss://api.mainnet-beta.solana.com',
    CLICKHOUSE_URL: process.env.CLICKHOUSE_URL || 'http://localhost:8123',
    DETECTOR_API: process.env.DETECTOR_API || 'http://localhost:8000',
    
    // Performance settings
    BATCH_SIZE: 1000,
    FLUSH_INTERVAL_MS: 100,
    MAX_QUEUE_SIZE: 10000,
    
    // Detection settings
    ENABLE_REALTIME_DETECTION: true,
    DETECTION_CONFIDENCE_THRESHOLD: 0.7,
    
    // Priority addresses for monitoring
    PRIORITY_ADDRESSES: new Set([
        'B91piBSfCBRs5rUxCMRdJEGv7tNEnFxweWcdQJHJoFpi',
        '6gAnjderE13TGGFeqdPVQ438jp2FPVeyXAszxKu9y338',
        'CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C',
        'E6YoRP3adE5XYneSseLee15wJshDxCsmyD2WtLvAmfLi',
        'pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA'
    ]),
    
    // Known DEX programs
    DEX_PROGRAMS: new Set([
        '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8',  // Raydium V4
        'CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK',  // Raydium CPMM
        '9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP',  // Orca Whirlpool
        'whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc',  // Whirlpool
        '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P',  // Pump.fun
    ])
};

// Express app setup
const app = express();
const server = http.createServer(app);

// CORS headers
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  next();
});

// Proxy to MEV Backend (Rust service)
app.use('/api/mev', createProxyMiddleware({
  target: 'http://localhost:8080',
  changeOrigin: true,
  pathRewrite: { '^/api/mev': '' }
}));

// Proxy to API Gateway
app.use('/api/gateway', createProxyMiddleware({
  target: 'https://localhost:3000',
  changeOrigin: true,
  secure: false,
  pathRewrite: { '^/api/gateway': '/api' }
}));

// Proxy to Metrics
app.use('/api/metrics', createProxyMiddleware({
  target: 'http://localhost:9090',
  changeOrigin: true,
  pathRewrite: { '^/api/metrics': '/metrics' }
}));

// WebSocket Bridge
const wss = new WebSocket.Server({ server, path: '/ws' });

// Connect to backend WebSocket services
let backendWS = null;
function connectBackend() {
  backendWS = new WebSocket('ws://localhost:8080/ws');
  
  backendWS.on('open', () => {
    console.log('Connected to MEV backend WebSocket');
  });
  
  backendWS.on('message', (data) => {
    // Broadcast to all frontend clients
    wss.clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(data);
      }
    });
  });
  
  backendWS.on('close', () => {
    console.log('Backend WebSocket disconnected, reconnecting...');
    setTimeout(connectBackend, 1000);
  });
  
  backendWS.on('error', (err) => {
    console.error('Backend WebSocket error:', err);
  });
}

// MEV Detection Statistics
const detectionStats = {
    totalProcessed: 0,
    mevDetected: 0,
    sandwichesFound: 0,
    arbitragesFound: 0,
    priorityAddressHits: 0,
    latencyP50: 0,
    latencyP99: 0
};

// Handle frontend connections
wss.on('connection', (ws) => {
  console.log('Frontend client connected');
  
  // Send initial data
  ws.send(JSON.stringify({
    type: 'connected',
    timestamp: Date.now(),
    services: {
      mev: 'http://localhost:8080',
      gateway: 'https://localhost:3000',
      metrics: 'http://localhost:9090',
      detector: CONFIG.DETECTOR_API
    },
    stats: detectionStats,
    priorityAddresses: Array.from(CONFIG.PRIORITY_ADDRESSES)
  }));
  
  ws.on('message', async (message) => {
    try {
        const data = JSON.parse(message);
        
        // Handle detection requests
        if (data.type === 'detect') {
            const result = await runDetection(data.transaction);
            ws.send(JSON.stringify({
                type: 'detection_result',
                result: result
            }));
        }
        
        // Forward to backend if connected
        if (backendWS && backendWS.readyState === WebSocket.OPEN) {
            backendWS.send(message);
        }
    } catch (error) {
        console.error('Message handling error:', error);
    }
  });
  
  ws.on('close', () => {
    console.log('Frontend client disconnected');
  });
});

// Detection API endpoint
app.post('/api/detect', express.json(), async (req, res) => {
    try {
        const { transaction, features } = req.body;
        
        // Check if priority address
        const isPriority = features?.accountKeys?.some(key => 
            CONFIG.PRIORITY_ADDRESSES.has(key)
        );
        
        if (isPriority) {
            detectionStats.priorityAddressHits++;
        }
        
        // Run detection
        const response = await axios.post(`${CONFIG.DETECTOR_API}/detect`, {
            transaction,
            features
        }, {
            timeout: 1000
        });
        
        // Update stats
        detectionStats.totalProcessed++;
        if (response.data.is_mev) {
            detectionStats.mevDetected++;
            if (response.data.mev_type === 'sandwich') {
                detectionStats.sandwichesFound++;
            } else if (response.data.mev_type === 'arbitrage') {
                detectionStats.arbitragesFound++;
            }
        }
        
        res.json(response.data);
    } catch (error) {
        console.error('Detection error:', error);
        res.status(500).json({ error: 'Detection failed' });
    }
});

// Entity profile endpoint
app.get('/api/entity/:address', async (req, res) => {
    try {
        const { address } = req.params;
        
        // Fetch entity profile from detector API
        const response = await axios.get(
            `${CONFIG.DETECTOR_API}/entity/${address}`
        );
        
        res.json(response.data);
    } catch (error) {
        console.error('Entity profile error:', error);
        res.status(500).json({ error: 'Failed to fetch entity profile' });
    }
});

// Statistics endpoint
app.get('/api/stats', (req, res) => {
    res.json({
        detection: detectionStats,
        timestamp: Date.now(),
        uptime: process.uptime(),
        connections: {
            frontend: wss.clients.size,
            backend: backendWS && backendWS.readyState === WebSocket.OPEN
        }
    });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: Date.now(),
    connections: {
      frontend: wss.clients.size,
      backend: backendWS && backendWS.readyState === WebSocket.OPEN
    },
    stats: detectionStats
  });
});

// Start backend connection
connectBackend();

// Periodic stats broadcast
setInterval(() => {
    const statsMessage = JSON.stringify({
        type: 'stats_update',
        stats: detectionStats,
        timestamp: Date.now()
    });
    
    wss.clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(statsMessage);
        }
    });
}, 5000);

const PORT = process.env.PORT || 4000;
server.listen(PORT, () => {
  console.log(`MEV Detection Bridge running on port ${PORT}`);
  console.log(`WebSocket available at ws://localhost:${PORT}/ws`);
  console.log(`Health check at http://localhost:${PORT}/health`);
  console.log(`Detection stats at http://localhost:${PORT}/api/stats`);
});