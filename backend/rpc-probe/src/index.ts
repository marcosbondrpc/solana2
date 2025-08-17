import { Connection, PublicKey, Commitment } from '@solana/web3.js';
import Fastify from 'fastify';
import { register as promRegister, Histogram, Counter, Gauge } from 'prom-client';
import PQueue from 'p-queue';
import Redis from 'ioredis';
import { RPCProbe } from './probe';
import { HealthScorer } from './health-scorer';
import { config } from './config';

const server = Fastify({
  logger: {
    level: 'info',
    transport: {
      target: 'pino-pretty',
      options: {
        colorize: true,
        translateTime: 'HH:MM:ss Z',
        ignore: 'pid,hostname'
      }
    }
  }
});

const redis = new Redis({
  host: config.redis.host,
  port: config.redis.port,
  password: config.redis.password,
  keyPrefix: 'rpc-probe:'
});

// Metrics
const rpcLatency = new Histogram({
  name: 'solana_rpc_latency_seconds',
  help: 'RPC method latency in seconds',
  labelNames: ['method', 'commitment', 'status'],
  buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
});

const rpcErrors = new Counter({
  name: 'solana_rpc_errors_total',
  help: 'Total RPC errors',
  labelNames: ['method', 'error_type']
});

const rpcRequests = new Counter({
  name: 'solana_rpc_requests_total',
  help: 'Total RPC requests',
  labelNames: ['method', 'commitment']
});

const healthScore = new Gauge({
  name: 'solana_rpc_health_score',
  help: 'RPC health score (0-100)',
  labelNames: ['endpoint']
});

const slotLag = new Gauge({
  name: 'solana_rpc_slot_lag',
  help: 'Slot lag behind the network',
  labelNames: ['endpoint']
});

async function bootstrap() {
  try {
    // Initialize RPC connections
    const connections = config.endpoints.map(endpoint => ({
      endpoint,
      connection: new Connection(endpoint, {
        commitment: 'confirmed' as Commitment,
        wsEndpoint: endpoint.replace('http', 'ws'),
        disableRetryOnRateLimit: false,
        confirmTransactionInitialTimeout: 60000
      })
    }));
    
    // Initialize probes
    const probes = connections.map(({ endpoint, connection }) => 
      new RPCProbe(connection, endpoint, {
        redis,
        metrics: {
          latency: rpcLatency,
          errors: rpcErrors,
          requests: rpcRequests,
          healthScore,
          slotLag
        }
      })
    );
    
    // Initialize health scorer
    const scorer = new HealthScorer(redis);
    
    // Start continuous monitoring
    const queue = new PQueue({ 
      concurrency: config.probe.concurrency,
      interval: config.probe.interval,
      intervalCap: config.probe.intervalCap
    });
    
    // Schedule probes
    const scheduleProbes = () => {
      probes.forEach(probe => {
        // Core RPC methods to monitor
        const methods = [
          'getAccountInfo',
          'getProgramAccounts',
          'getBlock',
          'getBlockHeight',
          'getSlot',
          'getTransaction',
          'getSignatureStatuses',
          'getRecentBlockhash',
          'getLatestBlockhash',
          'sendTransaction',
          'simulateTransaction',
          'getHealth'
        ];
        
        methods.forEach(method => {
          queue.add(async () => {
            try {
              const result = await probe.measureMethod(method);
              
              // Calculate health score
              const score = await scorer.calculateScore({
                endpoint: probe.endpoint,
                method,
                latency: result.latency,
                success: result.success,
                errorRate: result.errorRate
              });
              
              healthScore.set({ endpoint: probe.endpoint }, score);
              
              // Store in Redis for historical analysis
              await redis.zadd(
                `latency:${method}:${probe.endpoint}`,
                Date.now(),
                JSON.stringify({
                  latency: result.latency,
                  success: result.success,
                  timestamp: new Date().toISOString()
                })
              );
              
              // Cleanup old data (keep 24 hours)
              await redis.zremrangebyscore(
                `latency:${method}:${probe.endpoint}`,
                0,
                Date.now() - 86400000
              );
              
            } catch (error) {
              server.log.error(`Probe error for ${method}:`, error);
            }
          });
        });
      });
    };
    
    // Schedule probes every interval
    setInterval(scheduleProbes, config.probe.scheduleInterval);
    scheduleProbes(); // Initial run
    
    // API Routes
    server.get('/metrics', async (request, reply) => {
      reply.type('text/plain');
      reply.send(await promRegister.metrics());
    });
    
    server.get('/health', async (request, reply) => {
      const scores = await Promise.all(
        probes.map(async probe => ({
          endpoint: probe.endpoint,
          score: await scorer.getScore(probe.endpoint),
          status: await probe.getStatus()
        }))
      );
      
      const overallHealth = scores.reduce((sum, s) => sum + s.score, 0) / scores.length;
      
      reply.send({
        healthy: overallHealth > 70,
        overallScore: overallHealth,
        endpoints: scores,
        timestamp: new Date().toISOString()
      });
    });
    
    server.get('/latency/:method', async (request: any, reply) => {
      const { method } = request.params;
      const { endpoint, hours = 1 } = request.query;
      
      const since = Date.now() - (hours * 3600000);
      const key = `latency:${method}:${endpoint || config.endpoints[0]}`;
      
      const data = await redis.zrangebyscore(key, since, Date.now());
      const parsed = data.map(d => JSON.parse(d));
      
      // Calculate statistics
      const latencies = parsed.map(d => d.latency);
      const stats = {
        min: Math.min(...latencies),
        max: Math.max(...latencies),
        avg: latencies.reduce((a, b) => a + b, 0) / latencies.length,
        p50: percentile(latencies, 0.5),
        p95: percentile(latencies, 0.95),
        p99: percentile(latencies, 0.99),
        successRate: parsed.filter(d => d.success).length / parsed.length
      };
      
      reply.send({
        method,
        endpoint: endpoint || config.endpoints[0],
        period: `${hours}h`,
        dataPoints: parsed.length,
        stats,
        data: parsed
      });
    });
    
    server.get('/slot-lag', async (request, reply) => {
      const lags = await Promise.all(
        probes.map(async probe => ({
          endpoint: probe.endpoint,
          lag: await probe.getSlotLag()
        }))
      );
      
      reply.send({
        lags,
        timestamp: new Date().toISOString()
      });
    });
    
    // WebSocket for real-time metrics
    server.register(require('@fastify/websocket'));
    
    server.register(async function (fastify) {
      fastify.get('/ws', { websocket: true }, (connection, req) => {
        const interval = setInterval(async () => {
          const metrics = await Promise.all(
            probes.map(async probe => ({
              endpoint: probe.endpoint,
              latency: await probe.getCurrentLatency(),
              health: await scorer.getScore(probe.endpoint),
              slotLag: await probe.getSlotLag()
            }))
          );
          
          connection.socket.send(JSON.stringify({
            type: 'metrics',
            data: metrics,
            timestamp: new Date().toISOString()
          }));
        }, 1000);
        
        connection.socket.on('close', () => {
          clearInterval(interval);
        });
      });
    });
    
    await server.listen({ 
      port: config.port, 
      host: config.host 
    });
    
    server.log.info(`RPC Probe Service started on ${config.host}:${config.port}`);
    
  } catch (err) {
    server.log.error(err);
    process.exit(1);
  }
}

function percentile(values: number[], p: number): number {
  const sorted = values.sort((a, b) => a - b);
  const index = Math.ceil(sorted.length * p) - 1;
  return sorted[index];
}

bootstrap();