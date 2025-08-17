// Integration Bridge - Connects Frontend to Backend Services
const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const WebSocket = require('ws');
const http = require('http');

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
      metrics: 'http://localhost:9090'
    }
  }));
  
  ws.on('message', (message) => {
    // Forward to backend if connected
    if (backendWS && backendWS.readyState === WebSocket.OPEN) {
      backendWS.send(message);
    }
  });
  
  ws.on('close', () => {
    console.log('Frontend client disconnected');
  });
});

// Start backend connection
connectBackend();

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: Date.now(),
    connections: {
      frontend: wss.clients.size,
      backend: backendWS && backendWS.readyState === WebSocket.OPEN
    }
  });
});

const PORT = process.env.PORT || 4000;
server.listen(PORT, () => {
  console.log(`Integration bridge running on port ${PORT}`);
  console.log(`WebSocket available at ws://localhost:${PORT}/ws`);
  console.log(`Health check at http://localhost:${PORT}/health`);
});