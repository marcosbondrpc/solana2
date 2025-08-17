import type { FastifyInstance } from 'fastify';

export async function setupMetrics(server: FastifyInstance) {
  server.get('/api/node/metrics', async (request, reply) => {
    const upstream = await fetch('http://localhost:8081/metrics');
    const text = await upstream.text();
    reply.header('content-type', 'text/plain; version=0.0.4; charset=utf-8');
    return reply.send(text);
  });

  server.get('/api/node/health', async (request, reply) => {
    const upstream = await fetch('http://localhost:8081/health');
    const json = await upstream.json();
    reply.header('content-type', 'application/json');
    return reply.send(json);
  });
}