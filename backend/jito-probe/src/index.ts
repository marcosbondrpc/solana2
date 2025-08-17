import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import { Connection, PublicKey, LAMPORTS_PER_SOL } from '@solana/web3.js';
import Fastify from 'fastify';
import { Histogram, Counter, Gauge, register as promRegister } from 'prom-client';
import WebSocket from 'ws';
import Redis from 'ioredis';
import Decimal from 'decimal.js';
import axios from 'axios';
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
  keyPrefix: 'jito-probe:'
});

// Metrics
const bundleAcceptanceRate = new Gauge({
  name: 'jito_bundle_acceptance_rate',
  help: 'Bundle acceptance rate percentage',
  labelNames: ['region', 'relay']
});

const bundleLandingRate = new Gauge({
  name: 'jito_bundle_landing_rate',
  help: 'Bundle landing rate percentage',
  labelNames: ['region', 'relay']
});

const tipSpend = new Histogram({
  name: 'jito_tip_spend_sol',
  help: 'Tip spend in SOL',
  labelNames: ['region', 'status'],
  buckets: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
});

const relayLatency = new Histogram({
  name: 'jito_relay_latency_ms',
  help: 'Relay round-trip time in milliseconds',
  labelNames: ['region', 'relay'],
  buckets: [1, 5, 10, 25, 50, 100, 250, 500, 1000]
});

const expectedValue = new Gauge({
  name: 'jito_expected_value_sol',
  help: 'Expected value of bundles in SOL',
  labelNames: ['strategy', 'region']
});

const priorityFees = new Histogram({
  name: 'jito_priority_fees_microlamports',
  help: 'Priority fees in microlamports per CU',
  labelNames: ['percentile'],
  buckets: [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
});

const bundlesSubmitted = new Counter({
  name: 'jito_bundles_submitted_total',
  help: 'Total bundles submitted',
  labelNames: ['region', 'relay']
});

const bundlesLanded = new Counter({
  name: 'jito_bundles_landed_total',
  help: 'Total bundles landed on-chain',
  labelNames: ['region', 'relay']
});

interface Bundle {
  id: string;
  transactions: string[];
  tipAmount: number;
  slot: number;
  timestamp: number;
  relay: string;
  region: string;
  status: 'pending' | 'accepted' | 'landed' | 'dropped';
  landedSlot?: number;
  expectedValue?: number;
}

class JitoMEVProbe {
  private connection: Connection;
  private blockEngineUrl: string;
  private ws: WebSocket | null = null;
  private bundles: Map<string, Bundle> = new Map();
  private grpcClient: any;
  
  constructor(
    connection: Connection,
    blockEngineUrl: string,
    private relayRegion: string
  ) {
    this.connection = connection;
    this.blockEngineUrl = blockEngineUrl;
    this.initializeGRPC();
    this.connectWebSocket();
  }
  
  private initializeGRPC() {
    // Load Jito proto files
    const packageDefinition = protoLoader.loadSync(
      config.jito.protoPath,
      {
        keepCase: true,
        longs: String,
        enums: String,
        defaults: true,
        oneofs: true
      }
    );
    
    const proto = grpc.loadPackageDefinition(packageDefinition) as any;
    
    // Create gRPC client
    this.grpcClient = new proto.block_engine.BlockEngineService(
      this.blockEngineUrl,
      grpc.credentials.createSsl()
    );
  }
  
  private connectWebSocket() {
    const wsUrl = this.blockEngineUrl.replace('grpc', 'wss');
    
    this.ws = new WebSocket(wsUrl);
    
    this.ws.on('open', () => {
      server.log.info(`Connected to Jito WebSocket: ${wsUrl}`);
      
      // Subscribe to bundle updates
      this.ws?.send(JSON.stringify({
        type: 'subscribe',
        channel: 'bundles'
      }));
    });
    
    this.ws.on('message', (data) => {
      try {
        const message = JSON.parse(data.toString());
        this.handleBundleUpdate(message);
      } catch (error) {
        server.log.error('WebSocket message error:', error);
      }
    });
    
    this.ws.on('close', () => {
      server.log.warn('Jito WebSocket disconnected, reconnecting...');
      setTimeout(() => this.connectWebSocket(), 5000);
    });
    
    this.ws.on('error', (error) => {
      server.log.error('WebSocket error:', error);
    });
  }
  
  private handleBundleUpdate(message: any) {
    if (message.type === 'bundle_update') {
      const bundle = this.bundles.get(message.bundleId);
      if (bundle) {
        bundle.status = message.status;
        if (message.status === 'landed') {
          bundle.landedSlot = message.slot;
          bundlesLanded.inc({ 
            region: this.relayRegion, 
            relay: bundle.relay 
          });
        }
        this.bundles.set(message.bundleId, bundle);
      }
    }
  }
  
  async submitBundle(transactions: string[], tipAmount: number): Promise<string> {
    const bundleId = this.generateBundleId();
    const slot = await this.connection.getSlot();
    
    const bundle: Bundle = {
      id: bundleId,
      transactions,
      tipAmount,
      slot,
      timestamp: Date.now(),
      relay: this.blockEngineUrl,
      region: this.relayRegion,
      status: 'pending'
    };
    
    this.bundles.set(bundleId, bundle);
    bundlesSubmitted.inc({ 
      region: this.relayRegion, 
      relay: this.blockEngineUrl 
    });
    
    // Submit via gRPC
    return new Promise((resolve, reject) => {
      const request = {
        bundle: {
          header: {
            bundle_id: bundleId,
            tip_lamports: tipAmount * LAMPORTS_PER_SOL
          },
          transactions: transactions.map(tx => ({ data: tx }))
        }
      };
      
      const startTime = Date.now();
      
      this.grpcClient.SendBundle(request, (error: any, response: any) => {
        const latency = Date.now() - startTime;
        relayLatency.observe(
          { region: this.relayRegion, relay: this.blockEngineUrl },
          latency
        );
        
        if (error) {
          reject(error);
        } else {
          bundle.status = 'accepted';
          this.bundles.set(bundleId, bundle);
          resolve(bundleId);
        }
      });
    });
  }
  
  async measureRelayLatency(): Promise<number> {
    const startTime = Date.now();
    
    return new Promise((resolve) => {
      this.grpcClient.GetTipAccounts({}, (error: any, response: any) => {
        const latency = Date.now() - startTime;
        relayLatency.observe(
          { region: this.relayRegion, relay: this.blockEngineUrl },
          latency
        );
        resolve(latency);
      });
    });
  }
  
  async getTipAccounts(): Promise<string[]> {
    return new Promise((resolve, reject) => {
      this.grpcClient.GetTipAccounts({}, (error: any, response: any) => {
        if (error) {
          reject(error);
        } else {
          resolve(response.accounts || []);
        }
      });
    });
  }
  
  async calculateExpectedValue(
    strategy: string,
    tipAmount: number,
    estimatedProfit: number
  ): Promise<number> {
    const acceptanceProb = await this.getAcceptanceProbability(tipAmount);
    const landingProb = await this.getLandingProbability();
    
    const ev = (estimatedProfit - tipAmount) * acceptanceProb * landingProb;
    
    expectedValue.set(
      { strategy, region: this.relayRegion },
      ev / LAMPORTS_PER_SOL
    );
    
    return ev;
  }
  
  private async getAcceptanceProbability(tipAmount: number): Promise<number> {
    // Fetch historical acceptance rates based on tip amount
    const key = `acceptance:${this.relayRegion}`;
    const data = await redis.zrangebyscore(key, tipAmount - 1000, tipAmount + 1000);
    
    if (data.length === 0) return 0.5; // Default probability
    
    const accepted = data.filter(d => JSON.parse(d).accepted).length;
    return accepted / data.length;
  }
  
  private async getLandingProbability(): Promise<number> {
    // Calculate landing probability from recent bundles
    const recentBundles = Array.from(this.bundles.values())
      .filter(b => Date.now() - b.timestamp < 3600000); // Last hour
    
    if (recentBundles.length === 0) return 0.5;
    
    const landed = recentBundles.filter(b => b.status === 'landed').length;
    return landed / recentBundles.length;
  }
  
  getMetrics() {
    const total = this.bundles.size;
    const accepted = Array.from(this.bundles.values())
      .filter(b => b.status === 'accepted' || b.status === 'landed').length;
    const landed = Array.from(this.bundles.values())
      .filter(b => b.status === 'landed').length;
    
    const acceptanceRate = total > 0 ? (accepted / total) * 100 : 0;
    const landingRate = accepted > 0 ? (landed / accepted) * 100 : 0;
    
    bundleAcceptanceRate.set(
      { region: this.relayRegion, relay: this.blockEngineUrl },
      acceptanceRate
    );
    
    bundleLandingRate.set(
      { region: this.relayRegion, relay: this.blockEngineUrl },
      landingRate
    );
    
    return {
      total,
      accepted,
      landed,
      acceptanceRate,
      landingRate
    };
  }
  
  private generateBundleId(): string {
    return `bundle_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  }
}

async function bootstrap() {
  try {
    // Initialize Solana connection
    const connection = new Connection(config.solana.rpcUrl, {
      commitment: 'confirmed',
      wsEndpoint: config.solana.wsUrl
    });
    
    // Initialize Jito probes for different regions
    const probes = config.jito.relays.map(relay => 
      new JitoMEVProbe(connection, relay.url, relay.region)
    );
    
    // Monitor priority fees
    const monitorPriorityFees = async () => {
      try {
        const response = await axios.get(config.jito.priorityFeeApi);
        const fees = response.data;
        
        priorityFees.observe({ percentile: 'p50' }, fees.p50);
        priorityFees.observe({ percentile: 'p75' }, fees.p75);
        priorityFees.observe({ percentile: 'p90' }, fees.p90);
        priorityFees.observe({ percentile: 'p99' }, fees.p99);
        
        // Store in Redis for historical analysis
        await redis.zadd(
          'priority-fees',
          Date.now(),
          JSON.stringify({
            timestamp: new Date().toISOString(),
            ...fees
          })
        );

        // Also store a current snapshot for routing consumers (bandit route selection)
        await redis.mset({
          'priority-fees:current:p50': String(fees.p50),
          'priority-fees:current:p75': String(fees.p75),
          'priority-fees:current:p90': String(fees.p90),
          'priority-fees:current:p99': String(fees.p99)
        });
      } catch (error) {
        server.log.error('Priority fee monitoring error:', error);
      }
    };
    
    // Schedule monitoring tasks
    setInterval(monitorPriorityFees, 10000); // Every 10 seconds
    setInterval(() => {
      probes.forEach(probe => probe.getMetrics());
    }, 5000); // Every 5 seconds
    
    // API Routes
    server.get('/metrics', async (request, reply) => {
      reply.type('text/plain');
      reply.send(await promRegister.metrics());
    });
    
    server.get('/health', async (request, reply) => {
      const health = await Promise.all(
        probes.map(async probe => ({
          region: probe.relayRegion,
          metrics: probe.getMetrics(),
          latency: await probe.measureRelayLatency()
        }))
      );
      
      reply.send({
        healthy: true,
        relays: health,
        timestamp: new Date().toISOString()
      });
    });
    
    server.post('/bundle/submit', async (request: any, reply) => {
      const { transactions, tipAmount, relay } = request.body;
      
      const probe = probes.find(p => p.relayRegion === relay);
      if (!probe) {
        reply.code(400).send({ error: 'Invalid relay region' });
        return;
      }
      
      try {
        const bundleId = await probe.submitBundle(transactions, tipAmount);
        
        tipSpend.observe(
          { region: relay, status: 'submitted' },
          tipAmount
        );
        
        reply.send({
          success: true,
          bundleId,
          relay,
          tipAmount
        });
      } catch (error) {
        reply.code(500).send({ 
          error: 'Bundle submission failed',
          details: error.message 
        });
      }
    });
    
    server.get('/bundle/:bundleId', async (request: any, reply) => {
      const { bundleId } = request.params;
      
      for (const probe of probes) {
        const bundle = probe.bundles.get(bundleId);
        if (bundle) {
          reply.send(bundle);
          return;
        }
      }
      
      reply.code(404).send({ error: 'Bundle not found' });
    });
    
    server.get('/priority-fees', async (request, reply) => {
      const { hours = 1 } = request.query as any;
      const since = Date.now() - (hours * 3600000);
      
      const data = await redis.zrangebyscore('priority-fees', since, Date.now());
      const parsed = data.map(d => JSON.parse(d));
      
      reply.send({
        period: `${hours}h`,
        dataPoints: parsed.length,
        current: parsed[parsed.length - 1],
        history: parsed
      });
    });
    
    server.get('/expected-value', async (request: any, reply) => {
      const { strategy, tipAmount, estimatedProfit } = request.query;
      
      const probe = probes[0]; // Use first probe for calculation
      const ev = await probe.calculateExpectedValue(
        strategy as string,
        parseFloat(tipAmount as string),
        parseFloat(estimatedProfit as string)
      );
      
      reply.send({
        strategy,
        tipAmount: parseFloat(tipAmount as string),
        estimatedProfit: parseFloat(estimatedProfit as string),
        expectedValue: ev / LAMPORTS_PER_SOL,
        currency: 'SOL'
      });
    });
    
    // WebSocket for real-time bundle updates
    server.register(require('@fastify/websocket'));
    
    server.register(async function (fastify) {
      fastify.get('/ws', { websocket: true }, (connection, req) => {
        const interval = setInterval(() => {
          const metrics = probes.map(probe => ({
            region: probe.relayRegion,
            ...probe.getMetrics()
          }));
          
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
    
    server.log.info(`Jito MEV Probe started on ${config.host}:${config.port}`);
    
  } catch (err) {
    server.log.error(err);
    process.exit(1);
  }
}

bootstrap();