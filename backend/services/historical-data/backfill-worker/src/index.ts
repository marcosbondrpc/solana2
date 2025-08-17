import { Connection, PublicKey } from '@solana/web3.js';
import { Kafka, Producer, CompressionTypes } from 'kafkajs';
import { ClickHouse } from 'clickhouse';
import Redis from 'ioredis';
import Queue from 'bull';
import pino from 'pino';
import { register, Counter, Gauge, Histogram } from 'prom-client';
import express from 'express';
import PQueue from 'p-queue';
import pRetry from 'p-retry';
import axios from 'axios';
import zlib from 'zlib';
import { promisify } from 'util';
import crypto from 'crypto';
import bs58 from 'bs58';
import { config } from 'dotenv';
import { Mutex } from 'async-mutex';
import * as _ from 'lodash';

config();

const gzip = promisify(zlib.gzip);
const gunzip = promisify(zlib.gunzip);
const brotliCompress = promisify(zlib.brotliCompress);
const brotliDecompress = promisify(zlib.brotliDecompress);

// Configuration
const CONFIG = {
  RPC_ENDPOINTS: (process.env.RPC_ENDPOINTS || 'https://api.mainnet-beta.solana.com').split(','),
  KAFKA_BROKERS: (process.env.KAFKA_BROKERS || 'localhost:19092').split(','),
  CLICKHOUSE_HOST: process.env.CLICKHOUSE_HOST || 'localhost',
  CLICKHOUSE_PORT: parseInt(process.env.CLICKHOUSE_PORT || '8123'),
  CLICKHOUSE_USER: process.env.CLICKHOUSE_USER || 'solana',
  CLICKHOUSE_PASSWORD: process.env.CLICKHOUSE_PASSWORD || 'mev_billions_2025',
  CLICKHOUSE_DATABASE: process.env.CLICKHOUSE_DATABASE || 'solana_history',
  REDIS_URL: process.env.REDIS_URL || 'redis://localhost:6379',
  CONCURRENCY: parseInt(process.env.CONCURRENCY || '100'),
  BATCH_SIZE: parseInt(process.env.BATCH_SIZE || '100'),
  START_SLOT: parseInt(process.env.START_SLOT || '0'),
  END_SLOT: parseInt(process.env.END_SLOT || '999999999'),
  COMPRESSION: process.env.COMPRESSION || 'gzip',
  METRICS_PORT: parseInt(process.env.METRICS_PORT || '9091'),
};

// Logger
const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  transport: {
    target: 'pino-pretty',
    options: {
      colorize: true,
      translateTime: 'SYS:standard',
    },
  },
});

// Metrics
const metrics = {
  slotsProcessed: new Counter({
    name: 'backfill_slots_processed_total',
    help: 'Total number of slots processed',
  }),
  transactionsProcessed: new Counter({
    name: 'backfill_transactions_processed_total',
    help: 'Total number of transactions processed',
  }),
  errors: new Counter({
    name: 'backfill_errors_total',
    help: 'Total number of errors',
    labelNames: ['type'],
  }),
  currentSlot: new Gauge({
    name: 'backfill_current_slot',
    help: 'Current slot being processed',
  }),
  processingDuration: new Histogram({
    name: 'backfill_processing_duration_seconds',
    help: 'Duration of slot processing',
    buckets: [0.1, 0.5, 1, 2, 5, 10, 30],
  }),
  rpcLatency: new Histogram({
    name: 'backfill_rpc_latency_seconds',
    help: 'RPC call latency',
    buckets: [0.1, 0.25, 0.5, 1, 2, 5],
  }),
  kafkaLatency: new Histogram({
    name: 'backfill_kafka_latency_seconds',
    help: 'Kafka send latency',
    buckets: [0.01, 0.05, 0.1, 0.25, 0.5, 1],
  }),
};

// Interfaces
interface BackfillState {
  currentSlot: number;
  lastProcessedSlot: number;
  totalProcessed: number;
  totalTransactions: number;
  errors: number;
  startTime: Date;
  lastCheckpoint: Date;
}

interface SlotData {
  slot: number;
  parent_slot: number;
  block_height: number;
  block_time: number;
  leader: string;
  rewards_json: string;
  block_hash: string;
  parent_hash: string;
  transaction_count: number;
  entry_count: number;
  tick_count: number;
}

interface TransactionData {
  signature: string;
  slot: number;
  block_time: number;
  block_index: number;
  transaction_index: number;
  is_vote: boolean;
  success: boolean;
  fee: number;
  compute_units_consumed: number;
  err: string;
  memo: string;
  signer: string;
  signers: string[];
  account_keys: string[];
  pre_balances: number[];
  post_balances: number[];
  pre_token_balances_json: string;
  post_token_balances_json: string;
  instructions_json: string;
  inner_instructions_json: string;
  log_messages: string[];
  rewards_json: string;
  loaded_addresses_json: string;
  return_data_json: string;
}

class BackfillWorker {
  private kafka: Kafka;
  private producer: Producer;
  private clickhouse: ClickHouse;
  private redis: Redis;
  private queue: PQueue;
  private connections: Connection[];
  private currentConnectionIndex: number = 0;
  private state: BackfillState;
  private stateMutex: Mutex;
  private checkpointInterval: NodeJS.Timeout | null = null;

  constructor() {
    // Initialize Kafka
    this.kafka = new Kafka({
      clientId: 'solana-backfill-worker',
      brokers: CONFIG.KAFKA_BROKERS,
      compression: CompressionTypes.Snappy,
      retry: {
        initialRetryTime: 100,
        retries: 10,
      },
    });
    this.producer = this.kafka.producer({
      allowAutoTopicCreation: true,
      transactionTimeout: 30000,
    });

    // Initialize ClickHouse
    this.clickhouse = new ClickHouse({
      url: CONFIG.CLICKHOUSE_HOST,
      port: CONFIG.CLICKHOUSE_PORT,
      debug: false,
      basicAuth: {
        username: CONFIG.CLICKHOUSE_USER,
        password: CONFIG.CLICKHOUSE_PASSWORD,
      },
      isUseGzip: true,
      trimQuery: false,
      usePost: true,
      format: 'json',
      raw: false,
      config: {
        session_timeout: 60,
        output_format_json_quote_64bit_integers: 0,
        enable_http_compression: 1,
      },
    });

    // Initialize Redis
    this.redis = new Redis(CONFIG.REDIS_URL, {
      maxRetriesPerRequest: 5,
      enableReadyCheck: true,
      retryStrategy: (times) => Math.min(times * 50, 2000),
    });

    // Initialize processing queue
    this.queue = new PQueue({ 
      concurrency: CONFIG.CONCURRENCY,
      interval: 1000,
      intervalCap: CONFIG.CONCURRENCY * 2,
    });

    // Initialize RPC connections with load balancing
    this.connections = CONFIG.RPC_ENDPOINTS.map(endpoint => 
      new Connection(endpoint, {
        commitment: 'confirmed',
        disableRetryOnRateLimit: false,
        confirmTransactionInitialTimeout: 60000,
        httpHeaders: {
          'Accept-Encoding': 'gzip, br',
          'User-Agent': 'Solana-Backfill-Worker/1.0',
        },
      })
    );

    // Initialize state
    this.state = {
      currentSlot: CONFIG.START_SLOT,
      lastProcessedSlot: 0,
      totalProcessed: 0,
      totalTransactions: 0,
      errors: 0,
      startTime: new Date(),
      lastCheckpoint: new Date(),
    };

    this.stateMutex = new Mutex();
  }

  async initialize(): Promise<void> {
    logger.info('Initializing backfill worker...');

    // Connect to Kafka
    await this.producer.connect();
    logger.info('Connected to Kafka');

    // Test ClickHouse connection
    const result = await this.clickhouse.query('SELECT 1').toPromise();
    logger.info('Connected to ClickHouse');

    // Load state from Redis
    await this.loadState();

    // Start checkpoint timer
    this.checkpointInterval = setInterval(() => {
      this.saveState().catch(err => logger.error('Failed to save state:', err));
    }, 10000);

    logger.info('Backfill worker initialized');
  }

  private getConnection(): Connection {
    // Round-robin load balancing
    const connection = this.connections[this.currentConnectionIndex];
    this.currentConnectionIndex = (this.currentConnectionIndex + 1) % this.connections.length;
    return connection;
  }

  private async loadState(): Promise<void> {
    try {
      const savedState = await this.redis.get('backfill:state');
      if (savedState) {
        const parsed = JSON.parse(savedState);
        this.state = {
          ...parsed,
          startTime: new Date(parsed.startTime),
          lastCheckpoint: new Date(parsed.lastCheckpoint),
        };
        logger.info(`Resuming from slot ${this.state.currentSlot}`);
      }
    } catch (err) {
      logger.warn('No saved state found, starting fresh');
    }
  }

  private async saveState(): Promise<void> {
    const release = await this.stateMutex.acquire();
    try {
      this.state.lastCheckpoint = new Date();
      await this.redis.set('backfill:state', JSON.stringify(this.state), 'EX', 3600);
      
      // Update metrics
      metrics.currentSlot.set(this.state.currentSlot);
      
      // Save watermark to ClickHouse
      await this.saveWatermark();
    } finally {
      release();
    }
  }

  private async saveWatermark(): Promise<void> {
    const query = `
      INSERT INTO solana_history.consumer_progress (
        consumer_group, topic, partition, current_offset, 
        committed_offset, lag, update_time
      ) VALUES (
        'backfill-worker', 'slots', 0, 
        ${this.state.currentSlot}, ${this.state.lastProcessedSlot},
        ${this.state.currentSlot - this.state.lastProcessedSlot},
        now()
      )
    `;
    
    try {
      await this.clickhouse.query(query).toPromise();
    } catch (err) {
      logger.error('Failed to save watermark:', err);
    }
  }

  private async fetchBlockWithCompression(slot: number): Promise<any> {
    const connection = this.getConnection();
    const timer = metrics.rpcLatency.startTimer();

    try {
      const block = await pRetry(
        async () => {
          const response = await connection.getBlock(slot, {
            maxSupportedTransactionVersion: 0,
            transactionDetails: 'full',
            rewards: true,
          });

          if (!response) {
            throw new Error(`Block ${slot} not found`);
          }

          return response;
        },
        {
          retries: 5,
          minTimeout: 1000,
          maxTimeout: 10000,
          onFailedAttempt: (error) => {
            logger.warn(`Retry ${error.attemptNumber} for slot ${slot}: ${error.message}`);
          },
        }
      );

      timer();
      return block;
    } catch (err) {
      timer();
      throw err;
    }
  }

  private async processSlot(slot: number): Promise<void> {
    const timer = metrics.processingDuration.startTimer();

    try {
      logger.debug(`Processing slot ${slot}`);
      
      // Fetch block data
      const block = await this.fetchBlockWithCompression(slot);
      
      if (!block) {
        logger.debug(`Slot ${slot} is empty or not available`);
        return;
      }

      // Prepare slot data
      const slotData: SlotData = {
        slot,
        parent_slot: block.parentSlot,
        block_height: block.blockHeight || slot,
        block_time: block.blockTime || 0,
        leader: '', // Would need leader schedule
        rewards_json: JSON.stringify(block.rewards || []),
        block_hash: block.blockhash,
        parent_hash: block.previousBlockhash,
        transaction_count: block.transactions?.length || 0,
        entry_count: 0,
        tick_count: 0,
      };

      // Send slot data to Kafka
      await this.sendToKafka('solana.slots', slot.toString(), slotData);

      // Process transactions
      if (block.transactions && block.transactions.length > 0) {
        const transactionBatch = [];
        
        for (let i = 0; i < block.transactions.length; i++) {
          const tx = block.transactions[i];
          const meta = tx.meta;
          
          if (!meta) continue;

          const signature = tx.transaction.signatures[0];
          const message = tx.transaction.message;
          
          // Check if vote transaction
          const isVote = message.instructions.some((ix: any) => {
            const programId = message.accountKeys[ix.programIdIndex];
            return programId && programId.toBase58().startsWith('Vote');
          });

          const transactionData: TransactionData = {
            signature,
            slot,
            block_time: block.blockTime || 0,
            block_index: i,
            transaction_index: i,
            is_vote: isVote,
            success: meta.err === null,
            fee: meta.fee || 0,
            compute_units_consumed: meta.computeUnitsConsumed || 0,
            err: meta.err ? JSON.stringify(meta.err) : '',
            memo: '',
            signer: message.accountKeys[0]?.toBase58() || '',
            signers: message.accountKeys
              .slice(0, message.header.numRequiredSignatures)
              .map((k: PublicKey) => k.toBase58()),
            account_keys: message.accountKeys.map((k: PublicKey) => k.toBase58()),
            pre_balances: meta.preBalances || [],
            post_balances: meta.postBalances || [],
            pre_token_balances_json: JSON.stringify(meta.preTokenBalances || []),
            post_token_balances_json: JSON.stringify(meta.postTokenBalances || []),
            instructions_json: JSON.stringify(message.instructions || []),
            inner_instructions_json: JSON.stringify(meta.innerInstructions || []),
            log_messages: meta.logMessages || [],
            rewards_json: JSON.stringify(meta.rewards || []),
            loaded_addresses_json: JSON.stringify(meta.loadedAddresses || {}),
            return_data_json: JSON.stringify(meta.returnData || null),
          };

          transactionBatch.push(transactionData);
        }

        // Send transactions in batch
        if (transactionBatch.length > 0) {
          await this.sendBatchToKafka('solana.transactions', transactionBatch);
          
          const release = await this.stateMutex.acquire();
          try {
            this.state.totalTransactions += transactionBatch.length;
          } finally {
            release();
          }
        }
      }

      // Update state
      const release = await this.stateMutex.acquire();
      try {
        this.state.lastProcessedSlot = slot;
        this.state.totalProcessed++;
      } finally {
        release();
      }

      metrics.slotsProcessed.inc();
      metrics.transactionsProcessed.inc(block.transactions?.length || 0);
      
      timer();
    } catch (err: any) {
      timer();
      metrics.errors.inc({ type: 'processing' });
      
      const release = await this.stateMutex.acquire();
      try {
        this.state.errors++;
      } finally {
        release();
      }
      
      logger.error(`Failed to process slot ${slot}:`, err);
      throw err;
    }
  }

  private async sendToKafka(topic: string, key: string, data: any): Promise<void> {
    const timer = metrics.kafkaLatency.startTimer();
    
    try {
      let payload = JSON.stringify(data);
      
      // Apply compression
      if (CONFIG.COMPRESSION === 'gzip') {
        payload = (await gzip(payload)).toString('base64');
      } else if (CONFIG.COMPRESSION === 'brotli') {
        payload = (await brotliCompress(payload)).toString('base64');
      }

      await this.producer.send({
        topic,
        messages: [{
          key,
          value: payload,
          headers: {
            compression: CONFIG.COMPRESSION,
            timestamp: Date.now().toString(),
          },
        }],
        compression: CompressionTypes.Snappy,
      });
      
      timer();
    } catch (err) {
      timer();
      metrics.errors.inc({ type: 'kafka' });
      throw err;
    }
  }

  private async sendBatchToKafka(topic: string, batch: any[]): Promise<void> {
    const timer = metrics.kafkaLatency.startTimer();
    
    try {
      const messages = await Promise.all(
        batch.map(async (data) => {
          let payload = JSON.stringify(data);
          
          if (CONFIG.COMPRESSION === 'gzip') {
            payload = (await gzip(payload)).toString('base64');
          } else if (CONFIG.COMPRESSION === 'brotli') {
            payload = (await brotliCompress(payload)).toString('base64');
          }

          return {
            key: data.signature || data.slot?.toString() || crypto.randomUUID(),
            value: payload,
            headers: {
              compression: CONFIG.COMPRESSION,
              timestamp: Date.now().toString(),
            },
          };
        })
      );

      await this.producer.send({
        topic,
        messages,
        compression: CompressionTypes.Snappy,
      });
      
      timer();
    } catch (err) {
      timer();
      metrics.errors.inc({ type: 'kafka_batch' });
      throw err;
    }
  }

  async run(): Promise<void> {
    logger.info(`Starting backfill from slot ${this.state.currentSlot} to ${CONFIG.END_SLOT}`);
    
    // Process slots in batches
    while (this.state.currentSlot <= CONFIG.END_SLOT) {
      const batchPromises = [];
      const batchSize = Math.min(CONFIG.BATCH_SIZE, CONFIG.END_SLOT - this.state.currentSlot + 1);
      
      for (let i = 0; i < batchSize; i++) {
        const slot = this.state.currentSlot + i;
        batchPromises.push(
          this.queue.add(() => this.processSlot(slot))
        );
      }

      // Wait for batch to complete
      await Promise.allSettled(batchPromises);
      
      // Update current slot
      const release = await this.stateMutex.acquire();
      try {
        this.state.currentSlot += batchSize;
      } finally {
        release();
      }

      // Log progress
      const slotsPerSecond = this.state.totalProcessed / 
        ((Date.now() - this.state.startTime.getTime()) / 1000);
      
      logger.info({
        currentSlot: this.state.currentSlot,
        processed: this.state.totalProcessed,
        transactions: this.state.totalTransactions,
        errors: this.state.errors,
        slotsPerSecond: slotsPerSecond.toFixed(2),
        eta: ((CONFIG.END_SLOT - this.state.currentSlot) / slotsPerSecond / 60).toFixed(2) + ' minutes',
      }, 'Backfill progress');
    }

    logger.info('Backfill completed!');
  }

  async cleanup(): Promise<void> {
    logger.info('Cleaning up...');
    
    if (this.checkpointInterval) {
      clearInterval(this.checkpointInterval);
    }

    await this.saveState();
    await this.producer.disconnect();
    await this.redis.quit();
    
    logger.info('Cleanup completed');
  }
}

// Metrics server
function startMetricsServer(): void {
  const app = express();
  
  app.get('/metrics', async (req, res) => {
    res.set('Content-Type', register.contentType);
    res.end(await register.metrics());
  });

  app.get('/health', (req, res) => {
    res.json({ status: 'healthy' });
  });

  app.listen(CONFIG.METRICS_PORT, () => {
    logger.info(`Metrics server listening on port ${CONFIG.METRICS_PORT}`);
  });
}

// Main
async function main(): Promise<void> {
  const worker = new BackfillWorker();

  // Handle shutdown gracefully
  const shutdown = async () => {
    logger.info('Received shutdown signal');
    await worker.cleanup();
    process.exit(0);
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);

  try {
    // Start metrics server
    startMetricsServer();

    // Initialize and run worker
    await worker.initialize();
    await worker.run();
    await worker.cleanup();
    
    process.exit(0);
  } catch (err) {
    logger.error('Fatal error:', err);
    process.exit(1);
  }
}

// Run if main module
if (require.main === module) {
  main().catch(err => {
    logger.error('Unhandled error:', err);
    process.exit(1);
  });
}

export { BackfillWorker, CONFIG };