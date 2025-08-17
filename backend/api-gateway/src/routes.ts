import { FastifyInstance } from 'fastify';
import fastifyWebsocket from '@fastify/websocket';

export async function setupRoutes(server: FastifyInstance) {
  // Register WebSocket plugin
  await server.register(fastifyWebsocket);
  
  // Health check
  server.get('/health', async (request, reply) => {
    return { status: 'healthy', timestamp: new Date().toISOString() };
  });
  
  // Node metrics routes
  server.register(async function (server) {
    // GET /api/node/metrics
    server.get('/metrics', async (request, reply) => {
      const upstream = await fetch('http://localhost:8081/metrics');
      const text = await upstream.text();
      reply.header('content-type', 'text/plain; version=0.0.4; charset=utf-8');
      return reply.send(text);
    });
    
    // GET /api/node/health
    server.get('/health', async (request, reply) => {
      const response = await fetch('http://localhost:8081/health');
      return response.json();
    });
    
    // GET /api/node/latency/:type
    server.get<{ Params: { type: string } }>('/latency/:type', async (request, reply) => {
      const { type } = request.params;
      const response = await fetch(`http://localhost:8081/api/latency/${type}`);
      return response.json();
    });
    
    // WebSocket endpoint for real-time metrics
    server.get('/ws', { websocket: true }, (connection, req) => {
      const ws = new WebSocket('ws://localhost:8081');
      
      ws.on('open', () => {
        // Subscribe to all channels
        ws.send(JSON.stringify({
          type: 'Subscribe',
          channels: ['metrics', 'health', 'latency']
        }));
      });
      
      ws.on('message', (data) => {
        connection.socket.send(data.toString());
      });
      
      connection.socket.on('message', (message) => {
        ws.send(message.toString());
      });
      
      connection.socket.on('close', () => {
        ws.close();
      });
    });
  }, { prefix: '/api/node' });
  
  // Data scrapper routes
  server.register(async function (server) {
    // Scrapper control
    server.post('/scrapper/start', async (request, reply) => {
      const response = await fetch('http://localhost:8082/api/scrapper/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request.body)
      });
      return response.json();
    });
    
    server.post('/scrapper/stop', async (request, reply) => {
      const response = await fetch('http://localhost:8082/api/scrapper/stop', {
        method: 'POST'
      });
      return response.json();
    });
    
    server.get('/scrapper/status', async (request, reply) => {
      const response = await fetch('http://localhost:8082/api/scrapper/status');
      return response.json();
    });
    
    server.get('/scrapper/progress', async (request, reply) => {
      const response = await fetch('http://localhost:8082/api/scrapper/progress');
      return response.json();
    });
    
    // SSE endpoint for progress streaming
    server.get('/scrapper/progress/stream', async (request, reply) => {
      reply.raw.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
      });
      
      const interval = setInterval(async () => {
        try {
          const response = await fetch('http://localhost:8082/api/scrapper/progress');
          const data = await response.json();
          reply.raw.write(`data: ${JSON.stringify(data)}\n\n`);
        } catch (error) {
          server.log.error(error);
        }
      }, 1000);
      
      request.raw.on('close', () => {
        clearInterval(interval);
        reply.raw.end();
      });
    });
    
    // Dataset management
    server.get('/datasets', async (request, reply) => {
      const params = new URLSearchParams(request.query as any);
      const response = await fetch(`http://localhost:8082/api/datasets?${params}`);
      return response.json();
    });
    
    server.post('/datasets', async (request, reply) => {
      const response = await fetch('http://localhost:8082/api/datasets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request.body)
      });
      return response.json();
    });
    
    server.get<{ Params: { id: string } }>('/datasets/:id', async (request, reply) => {
      const response = await fetch(`http://localhost:8082/api/datasets/${request.params.id}`);
      return response.json();
    });
    
    server.delete<{ Params: { id: string } }>('/datasets/:id', async (request, reply) => {
      const response = await fetch(`http://localhost:8082/api/datasets/${request.params.id}`, {
        method: 'DELETE'
      });
      return response.json();
    });
    
    server.post<{ Params: { id: string } }>('/datasets/:id/export', async (request, reply) => {
      const response = await fetch(`http://localhost:8082/api/datasets/${request.params.id}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request.body)
      });
      
      // Stream the response
      reply.header('Content-Type', response.headers.get('content-type') || 'application/octet-stream');
      reply.header('Content-Disposition', response.headers.get('content-disposition') || 'attachment');
      
      return reply.send(response.body);
    });
    
    // ML endpoints
    server.post('/ml/train', async (request, reply) => {
      const response = await fetch('http://localhost:8082/api/ml/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request.body)
      });
      return response.json();
    });
    
    server.get('/ml/models', async (request, reply) => {
      const response = await fetch('http://localhost:8082/api/ml/models');
      return response.json();
    });
    
    server.get<{ Params: { id: string } }>('/ml/models/:id', async (request, reply) => {
      const response = await fetch(`http://localhost:8082/api/ml/models/${request.params.id}`);
      return response.json();
    });
    
    server.post<{ Params: { id: string } }>('/ml/models/:id/evaluate', async (request, reply) => {
      const response = await fetch(`http://localhost:8082/api/ml/models/${request.params.id}/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request.body)
      });
      return response.json();
    });
    
    // Historical data
    server.get('/data/blocks', async (request, reply) => {
      const params = new URLSearchParams(request.query as any);
      const response = await fetch(`http://localhost:8082/api/data/blocks?${params}`);
      return response.json();
    });
    
    server.get('/data/transactions', async (request, reply) => {
      const params = new URLSearchParams(request.query as any);
      const response = await fetch(`http://localhost:8082/api/data/transactions?${params}`);
      return response.json();
    });
    
    server.get('/data/accounts', async (request, reply) => {
      const params = new URLSearchParams(request.query as any);
      const response = await fetch(`http://localhost:8082/api/data/accounts?${params}`);
      return response.json();
    });
    
    server.get('/data/programs', async (request, reply) => {
      const params = new URLSearchParams(request.query as any);
      const response = await fetch(`http://localhost:8082/api/data/programs?${params}`);
      return response.json();
    });
  }, { prefix: '/api' });
  
  // MEV endpoints (existing)
  server.register(async function (server) {
    server.get('/opportunities', async (request, reply) => {
      // Proxy to MEV engine
      const response = await fetch('http://localhost:8001/api/mev/opportunities');
      return response.json();
    });
    
    server.post('/submit', async (request, reply) => {
      const response = await fetch('http://localhost:8001/api/mev/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request.body)
      });
      return response.json();
    });
    
    server.get('/stats', async (request, reply) => {
      const response = await fetch('http://localhost:8001/api/mev/stats');
      return response.json();
    });
  }, { prefix: '/api/mev' });
}