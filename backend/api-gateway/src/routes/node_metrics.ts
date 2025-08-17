import { Router } from 'express';
import WebSocket from 'ws';
import axios from 'axios';

const router = Router();

// Node metrics service configuration
const NODE_METRICS_URL = process.env.NODE_METRICS_URL || 'http://localhost:8081';
const NODE_METRICS_WS = process.env.NODE_METRICS_WS || 'ws://localhost:8081';

// GET /api/node/metrics - Get current node metrics
router.get('/metrics', async (req, res) => {
  try {
    const response = await axios.get(`${NODE_METRICS_URL}/api/metrics`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching node metrics:', error);
    res.status(500).json({ error: 'Failed to fetch node metrics' });
  }
});

// GET /api/node/health - Get connection health status
router.get('/health', async (req, res) => {
  try {
    const response = await axios.get(`${NODE_METRICS_URL}/api/health`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching health status:', error);
    res.status(500).json({ error: 'Failed to fetch health status' });
  }
});

// GET /api/node/latency/:type - Get latency statistics by type
router.get('/latency/:type', async (req, res) => {
  try {
    const { type } = req.params;
    const response = await axios.get(`${NODE_METRICS_URL}/api/latency/${type}`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching latency stats:', error);
    res.status(500).json({ error: 'Failed to fetch latency statistics' });
  }
});

// WebSocket proxy for real-time metrics streaming
export function setupNodeMetricsWebSocket(server: any) {
  const wss = new WebSocket.Server({ 
    server,
    path: '/ws/node-metrics'
  });

  wss.on('connection', (clientWs, request) => {
    console.log('New WebSocket connection for node metrics');
    
    // Connect to backend WebSocket
    const backendWs = new WebSocket(NODE_METRICS_WS);
    
    backendWs.on('open', () => {
      console.log('Connected to node metrics backend');
      
      // Subscribe to all channels
      backendWs.send(JSON.stringify({
        type: 'Subscribe',
        channels: ['metrics', 'health', 'latency']
      }));
    });
    
    backendWs.on('message', (data) => {
      // Forward messages from backend to client
      if (clientWs.readyState === WebSocket.OPEN) {
        clientWs.send(data.toString());
      }
    });
    
    backendWs.on('error', (error) => {
      console.error('Backend WebSocket error:', error);
      clientWs.close();
    });
    
    backendWs.on('close', () => {
      console.log('Backend WebSocket closed');
      clientWs.close();
    });
    
    // Handle client messages
    clientWs.on('message', (message) => {
      // Forward client messages to backend
      if (backendWs.readyState === WebSocket.OPEN) {
        backendWs.send(message.toString());
      }
    });
    
    clientWs.on('close', () => {
      console.log('Client WebSocket closed');
      backendWs.close();
    });
    
    clientWs.on('error', (error) => {
      console.error('Client WebSocket error:', error);
      backendWs.close();
    });
    
    // Send ping every 30 seconds to keep connection alive
    const pingInterval = setInterval(() => {
      if (clientWs.readyState === WebSocket.OPEN) {
        clientWs.ping();
      } else {
        clearInterval(pingInterval);
      }
    }, 30000);
  });
  
  return wss;
}

export default router;