/**
 * High-performance protobuf decoder worker
 * Handles 50k msg/s with minimal main thread blocking
 */

import * as protobuf from 'protobufjs';
import { decompress } from 'lz4js';

// Message schema (will be loaded from actual .proto files)
const messageSchema = `
  syntax = "proto3";
  
  message MEVEvent {
    string topic = 1;
    int64 ts_ns = 2;
    uint32 sequence = 3;
    
    oneof data {
      ArbitrageEvent arbitrage = 4;
      SandwichEvent sandwich = 5;
      LiquidationEvent liquidation = 6;
      MarketData market = 7;
    }
  }
  
  message ArbitrageEvent {
    string pool_a = 1;
    string pool_b = 2;
    string token_in = 3;
    string token_out = 4;
    uint64 amount_in = 5;
    uint64 amount_out = 6;
    uint64 profit = 7;
    double price_impact = 8;
    uint32 gas_used = 9;
  }
  
  message SandwichEvent {
    string target_tx = 1;
    string pool = 2;
    uint64 front_amount = 3;
    uint64 back_amount = 4;
    uint64 profit = 5;
    uint32 victim_count = 6;
  }
  
  message LiquidationEvent {
    string protocol = 1;
    string account = 2;
    uint64 debt_amount = 3;
    uint64 collateral_amount = 4;
    double health_factor = 5;
    uint64 liquidation_bonus = 6;
  }
  
  message MarketData {
    string pool = 1;
    uint64 reserve_0 = 2;
    uint64 reserve_1 = 3;
    uint64 volume_24h = 4;
    double price = 5;
    uint32 tx_count = 6;
  }
`;

// Load and compile the schema
let root: protobuf.Root;
let MEVEvent: protobuf.Type;

async function initSchema() {
  root = protobuf.parse(messageSchema).root;
  MEVEvent = root.lookupType('MEVEvent');
}

// Performance metrics
const metrics = {
  messagesProcessed: 0,
  bytesProcessed: 0,
  errorsCount: 0,
  avgProcessTime: 0,
  processTimeBuffer: new Float32Array(1000),
  bufferIndex: 0
};

// Message processing queue
const messageQueue: ArrayBuffer[] = [];
let processing = false;

async function processMessage(data: ArrayBuffer, receiveTime: number): Promise<any> {
  const startTime = performance.now();
  
  try {
    metrics.bytesProcessed += data.byteLength;
    
    // Check for compression
    const view = new Uint8Array(data);
    let messageData = view;
    
    if (view[0] === 0x04) { // LZ4 magic byte
      messageData = new Uint8Array(decompress(view.slice(1)));
    }
    
    // Decode protobuf
    const message = MEVEvent.decode(messageData);
    const obj = MEVEvent.toObject(message, {
      longs: String,
      enums: String,
      bytes: String,
      defaults: true,
      arrays: true,
      objects: true,
      oneofs: true
    });
    
    // Calculate latency
    const now = BigInt(Math.floor(receiveTime * 1e6));
    const latencyNs = Number(now - BigInt(obj.ts_ns || 0));
    const latencyMs = latencyNs / 1e6;
    
    // Update metrics
    const processTime = performance.now() - startTime;
    metrics.processTimeBuffer[metrics.bufferIndex] = processTime;
    metrics.bufferIndex = (metrics.bufferIndex + 1) % 1000;
    metrics.messagesProcessed++;
    
    // Calculate running average
    const validSamples = metrics.processTimeBuffer.filter(t => t > 0);
    metrics.avgProcessTime = validSamples.reduce((a, b) => a + b, 0) / validSamples.length;
    
    return {
      ...obj,
      latencyMs,
      processTimeMs: processTime
    };
  } catch (error) {
    metrics.errorsCount++;
    throw error;
  }
}

async function processQueue() {
  if (processing || messageQueue.length === 0) return;
  
  processing = true;
  
  while (messageQueue.length > 0) {
    const batch = messageQueue.splice(0, 100); // Process up to 100 messages at once
    
    await Promise.all(batch.map(async (data) => {
      try {
        const result = await processMessage(data, performance.now());
        // Send result back to main thread
        self.postMessage({
          type: 'decoded',
          result,
          metrics: getMetrics()
        });
      } catch (error) {
        self.postMessage({
          type: 'error',
          error: error.message,
          metrics: getMetrics()
        });
      }
    }));
  }
  
  processing = false;
}

function getMetrics() {
  return {
    messagesProcessed: metrics.messagesProcessed,
    bytesProcessed: metrics.bytesProcessed,
    errorsCount: metrics.errorsCount,
    avgProcessTimeMs: metrics.avgProcessTime,
    queueLength: messageQueue.length
  };
}

// Handle messages from main thread
self.addEventListener('message', async (event) => {
  const { id, data, receiveTime } = event.data;
  
  switch (event.data.type) {
    case 'init':
      await initSchema();
      self.postMessage({ type: 'ready' });
      break;
      
    case 'decode':
      messageQueue.push(data);
      processQueue();
      break;
      
    case 'getMetrics':
      self.postMessage({
        type: 'metrics',
        metrics: getMetrics()
      });
      break;
      
    default:
      // Legacy support for direct decode
      if (data instanceof ArrayBuffer) {
        try {
          const result = await processMessage(data, receiveTime || performance.now());
          self.postMessage({
            id,
            result,
            metrics: getMetrics()
          });
        } catch (error) {
          self.postMessage({
            id,
            error: error.message,
            metrics: getMetrics()
          });
        }
      }
  }
});

// Initialize on load
initSchema().then(() => {
  self.postMessage({ type: 'ready' });
});

// Export for TypeScript
export {};