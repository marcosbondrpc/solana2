const express = require('express');
const cors = require('cors');
const { createProxyMiddleware } = require('http-proxy-middleware');
const WebSocket = require('ws');
const http = require('http');

const app = express();
const server = http.createServer(app);

// Enable CORS
app.use(cors());

// Proxy node metrics API
app.use('/api/node', createProxyMiddleware({
  target: 'http://localhost:8081',
  changeOrigin: true,
}));

// Proxy scrapper API to Historical Capture API
app.use('/api/scrapper', createProxyMiddleware({
  target: 'http://localhost:8055',
  changeOrigin: true,
  pathRewrite: {
    '^/api/scrapper': '' // Remove /api/scrapper prefix when forwarding
  }
}));

// Proxy datasets endpoint to Historical Capture API
app.use('/datasets', createProxyMiddleware({
  target: 'http://localhost:8055',
  changeOrigin: true,
}));

// Proxy mission control API
app.use('/api/mission-control', createProxyMiddleware({
  target: 'http://localhost:8083',
  changeOrigin: true,
}));

// WebSocket server for node metrics
const wss = new WebSocket.Server({ noServer: true });

// Simulate real-time metrics
setInterval(() => {
  const metrics = {
    type: 'metrics_update',
    data: {
      rpc_latency: 20 + Math.sin(Date.now() / 1000) * 10,
      websocket_latency: 15 + Math.cos(Date.now() / 1000) * 8,
      geyser_latency: 25 + Math.sin(Date.now() / 1500) * 12,
      jito_latency: 18 + Math.cos(Date.now() / 2000) * 9,
      block_height: 250000000 + Math.floor(Date.now() / 500),
      slot: 280000000 + Math.floor(Date.now() / 250),
      tps: 3000 + Math.sin(Date.now() / 3000) * 500,
      peers: 1000 + Math.floor(Math.random() * 100),
      timestamp: Date.now()
    }
  };

  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify(metrics));
    }
  });
}, 100);

// WebSocket for scrapper progress
const scrapperWss = new WebSocket.Server({ noServer: true });

// Handle scrapper WebSocket connections
scrapperWss.on('connection', (ws) => {
  console.log('Scrapper WebSocket client connected');
  
  // Send initial status
  ws.send(JSON.stringify({
    type: 'connection',
    message: 'Connected to scrapper progress WebSocket'
  }));
  
  ws.on('close', () => {
    console.log('Scrapper WebSocket client disconnected');
  });
});

// WebSocket for mission control
const missionWss = new WebSocket.Server({ noServer: true, path: '/ws/mission-control' });

server.on('upgrade', (request, socket, head) => {
  const pathname = request.url;
  
  if (pathname === '/ws/node-metrics') {
    wss.handleUpgrade(request, socket, head, (ws) => {
      wss.emit('connection', ws, request);
    });
  } else if (pathname === '/ws/scrapper-progress') {
    scrapperWss.handleUpgrade(request, socket, head, (ws) => {
      scrapperWss.emit('connection', ws, request);
    });
  } else if (pathname === '/ws/mission-control') {
    // Proxy to Mission Control WebSocket
    const ws = new WebSocket('ws://localhost:8083/ws/mission-control');
    ws.on('open', () => {
      missionWss.handleUpgrade(request, socket, head, (clientWs) => {
        // Forward messages from client to backend
        clientWs.on('message', (msg) => ws.send(msg));
        // Forward messages from backend to client
        ws.on('message', (msg) => clientWs.send(msg));
        ws.on('close', () => clientWs.close());
        clientWs.on('close', () => ws.close());
      });
    });
  } else {
    socket.destroy();
  }
});

const PORT = process.env.PORT || 8085;
server.listen(PORT, () => {
  console.log(`API Proxy running on port ${PORT}`);
  console.log(`- Node metrics API: http://localhost:${PORT}/api/node/*`);
  console.log(`- Scrapper API: http://localhost:${PORT}/api/scrapper/*`);
  console.log(`- Datasets API: http://localhost:${PORT}/datasets/*`);
  console.log(`- WebSocket (node): ws://localhost:${PORT}/ws/node-metrics`);
  console.log(`- WebSocket (scrapper): ws://localhost:${PORT}/ws/scrapper-progress`);
});