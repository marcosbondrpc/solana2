/**
 * High-performance Web Worker for decoding WebSocket messages
 * Handles protobuf decoding, decompression, and parsing off the main thread
 */

// Message types matching main thread
enum MessageType {
  HEARTBEAT = 0,
  MEV_OPPORTUNITY = 1,
  BUNDLE_UPDATE = 2,
  LATENCY_METRIC = 3,
  BANDIT_STATE = 4,
  DECISION_DNA = 5,
  SLO_VIOLATION = 6,
  CONTROL_COMMAND = 7,
  SYSTEM_METRIC = 8,
  ROUTE_SELECTION = 9,
  PROFIT_UPDATE = 10,
  KILL_SWITCH = 11,
  MODEL_INFERENCE = 12,
  CANARY_RESULT = 13,
  LEADER_SCHEDULE = 14,
  NETWORK_TOPOLOGY = 15
}

// Decoding state
const decoder = new TextDecoder();
const ringBuffer = new Uint8Array(16 * 1024 * 1024); // 16MB ring buffer
let ringBufferOffset = 0;

// Performance tracking
let messagesProcessed = 0;
let bytesProcessed = 0;
let lastStatsTime = Date.now();

// Cache for frequently used objects (object pooling)
const objectPool = {
  opportunities: [] as any[],
  bundles: [] as any[],
  metrics: [] as any[]
};

function getFromPool(type: string): any {
  const pool = (objectPool as any)[type];
  return pool?.pop() || {};
}

function returnToPool(type: string, obj: any): void {
  const pool = (objectPool as any)[type];
  if (pool && pool.length < 100) {
    // Clear object for reuse
    for (const key in obj) {
      delete obj[key];
    }
    pool.push(obj);
  }
}

// Handle incoming messages from main thread
self.onmessage = async (event: MessageEvent) => {
  const { type, buffer } = event.data;

  switch (type) {
    case 'decode':
      await decodeMessage(buffer);
      break;
    case 'stats':
      sendStats();
      break;
    case 'clear':
      clearBuffers();
      break;
    default:
      self.postMessage({
        error: `Unknown message type: ${type}`
      });
  }
};

async function decodeMessage(buffer: ArrayBuffer): Promise<void> {
  const startTime = performance.now();

  try {
    const view = new DataView(buffer);
    let offset = 0;

    // Read header (16 bytes)
    const header = {
      version: view.getUint8(offset),
      type: view.getUint8(offset + 1) as MessageType,
      flags: view.getUint16(offset + 2, true),
      timestamp: view.getBigUint64(offset + 4, true),
      length: view.getUint32(offset + 12, true)
    };
    offset += 16;

    // Check for compression
    const isCompressed = (header.flags & 0x01) !== 0;
    const compressionType = (header.flags >> 1) & 0x03;
    const isPriority = (header.flags & 0x08) !== 0;

    let payload: ArrayBuffer;
    
    if (isCompressed) {
      payload = await decompress(buffer.slice(offset), compressionType);
    } else {
      payload = buffer.slice(offset);
    }

    // Decode based on message type
    const data = await decodeByType(header.type, payload);

    // Track stats
    messagesProcessed++;
    bytesProcessed += buffer.byteLength;

    // Calculate decode time
    const decodeTime = performance.now() - startTime;

    // Send decoded message back to main thread
    self.postMessage({
      type: header.type,
      data,
      timestamp: Number(header.timestamp),
      priority: isPriority,
      decodeTime
    });

    // Send stats periodically
    if (Date.now() - lastStatsTime > 1000) {
      sendStats();
    }

  } catch (error) {
    self.postMessage({
      error: `Decode error: ${(error as Error).message}`
    });
  }
}

async function decompress(buffer: ArrayBuffer, type: number): Promise<ArrayBuffer> {
  // In production, would use actual compression libraries
  // 0: no compression, 1: zstd, 2: brotli, 3: lz4
  
  switch (type) {
    case 1: // zstd
      // Would use zstd-wasm here
      return buffer;
    case 2: // brotli
      // Would use brotli decompression
      return buffer;
    case 3: // lz4
      // Would use lz4 decompression
      return buffer;
    default:
      return buffer;
  }
}

async function decodeByType(type: MessageType, buffer: ArrayBuffer): Promise<any> {
  switch (type) {
    case MessageType.MEV_OPPORTUNITY:
      return decodeMEVOpportunity(buffer);
    case MessageType.BUNDLE_UPDATE:
      return decodeBundleUpdate(buffer);
    case MessageType.LATENCY_METRIC:
      return decodeLatencyMetric(buffer);
    case MessageType.BANDIT_STATE:
      return decodeBanditState(buffer);
    case MessageType.DECISION_DNA:
      return decodeDecisionDNA(buffer);
    case MessageType.SLO_VIOLATION:
      return decodeSLOViolation(buffer);
    case MessageType.SYSTEM_METRIC:
      return decodeSystemMetric(buffer);
    case MessageType.ROUTE_SELECTION:
      return decodeRouteSelection(buffer);
    case MessageType.PROFIT_UPDATE:
      return decodeProfitUpdate(buffer);
    case MessageType.MODEL_INFERENCE:
      return decodeModelInference(buffer);
    case MessageType.CANARY_RESULT:
      return decodeCanaryResult(buffer);
    case MessageType.LEADER_SCHEDULE:
      return decodeLeaderSchedule(buffer);
    default:
      // Fallback to JSON
      try {
        const text = decoder.decode(buffer);
        return JSON.parse(text);
      } catch {
        return buffer;
      }
  }
}

function decodeMEVOpportunity(buffer: ArrayBuffer): any {
  // In production, would use protobuf decoding
  // For now, parse as JSON
  try {
    const text = decoder.decode(buffer);
    const data = JSON.parse(text);
    
    // Use object pooling for performance
    const opportunity = getFromPool('opportunities');
    Object.assign(opportunity, {
      id: data.id,
      timestamp: data.timestamp,
      type: data.type,
      dexA: data.dexA,
      dexB: data.dexB,
      tokenIn: data.tokenIn,
      tokenOut: data.tokenOut,
      amountIn: data.amountIn,
      expectedProfit: data.expectedProfit,
      confidence: data.confidence,
      route: data.route,
      dna: data.dna,
      slippage: data.slippage,
      gasEstimate: data.gasEstimate
    });
    
    return opportunity;
  } catch {
    return null;
  }
}

function decodeBundleUpdate(buffer: ArrayBuffer): any {
  try {
    const text = decoder.decode(buffer);
    const data = JSON.parse(text);
    
    const bundle = getFromPool('bundles');
    Object.assign(bundle, {
      id: data.id,
      bundleId: data.bundleId,
      timestamp: data.timestamp,
      status: data.status,
      transactions: data.transactions,
      tip: data.tip,
      landingSlot: data.landingSlot,
      relayUrl: data.relayUrl,
      submitLatency: data.submitLatency
    });
    
    return bundle;
  } catch {
    return null;
  }
}

function decodeLatencyMetric(buffer: ArrayBuffer): any {
  try {
    const text = decoder.decode(buffer);
    const data = JSON.parse(text);
    
    const metric = getFromPool('metrics');
    Object.assign(metric, {
      timestamp: data.timestamp,
      rpcLatency: data.rpcLatency,
      quicLatency: data.quicLatency,
      tcpLatency: data.tcpLatency,
      websocketLatency: data.websocketLatency,
      p50: data.p50,
      p95: data.p95,
      p99: data.p99
    });
    
    return metric;
  } catch {
    return null;
  }
}

function decodeBanditState(buffer: ArrayBuffer): any {
  try {
    const text = decoder.decode(buffer);
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function decodeDecisionDNA(buffer: ArrayBuffer): any {
  // DecisionDNA is a 256-bit fingerprint
  // Can be represented as hex string or Uint8Array
  if (buffer.byteLength === 32) {
    // Binary format
    const bytes = new Uint8Array(buffer);
    return {
      fingerprint: Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join(''),
      raw: bytes
    };
  } else {
    // JSON format
    try {
      const text = decoder.decode(buffer);
      return JSON.parse(text);
    } catch {
      return null;
    }
  }
}

function decodeSLOViolation(buffer: ArrayBuffer): any {
  try {
    const text = decoder.decode(buffer);
    const data = JSON.parse(text);
    
    return {
      timestamp: data.timestamp,
      metric: data.metric,
      threshold: data.threshold,
      actual: data.actual,
      severity: data.severity,
      duration: data.duration,
      affected: data.affected
    };
  } catch {
    return null;
  }
}

function decodeSystemMetric(buffer: ArrayBuffer): any {
  try {
    const text = decoder.decode(buffer);
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function decodeRouteSelection(buffer: ArrayBuffer): any {
  try {
    const text = decoder.decode(buffer);
    const data = JSON.parse(text);
    
    return {
      opportunityId: data.opportunityId,
      selectedRoute: data.selectedRoute, // 'direct', 'jito', 'hedged'
      confidence: data.confidence,
      expectedLatency: data.expectedLatency,
      expectedLandingRate: data.expectedLandingRate,
      backupRoutes: data.backupRoutes
    };
  } catch {
    return null;
  }
}

function decodeProfitUpdate(buffer: ArrayBuffer): any {
  try {
    const text = decoder.decode(buffer);
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function decodeModelInference(buffer: ArrayBuffer): any {
  try {
    const text = decoder.decode(buffer);
    const data = JSON.parse(text);
    
    return {
      modelId: data.modelId,
      inferenceTime: data.inferenceTime, // microseconds
      inputShape: data.inputShape,
      outputShape: data.outputShape,
      prediction: data.prediction,
      confidence: data.confidence
    };
  } catch {
    return null;
  }
}

function decodeCanaryResult(buffer: ArrayBuffer): any {
  try {
    const text = decoder.decode(buffer);
    const data = JSON.parse(text);
    
    return {
      canaryId: data.canaryId,
      timestamp: data.timestamp,
      route: data.route,
      success: data.success,
      latency: data.latency,
      landingSlot: data.landingSlot,
      cost: data.cost
    };
  } catch {
    return null;
  }
}

function decodeLeaderSchedule(buffer: ArrayBuffer): any {
  try {
    const text = decoder.decode(buffer);
    const data = JSON.parse(text);
    
    return {
      epoch: data.epoch,
      slot: data.slot,
      leaders: data.leaders, // Array of upcoming leaders
      currentLeader: data.currentLeader,
      nextRotation: data.nextRotation
    };
  } catch {
    return null;
  }
}

function sendStats(): void {
  const now = Date.now();
  const deltaTime = (now - lastStatsTime) / 1000;
  
  self.postMessage({
    type: 'stats',
    data: {
      messagesPerSecond: messagesProcessed / deltaTime,
      bytesPerSecond: bytesProcessed / deltaTime,
      totalMessages: messagesProcessed,
      totalBytes: bytesProcessed
    }
  });
  
  messagesProcessed = 0;
  bytesProcessed = 0;
  lastStatsTime = now;
}

function clearBuffers(): void {
  ringBufferOffset = 0;
  objectPool.opportunities = [];
  objectPool.bundles = [];
  objectPool.metrics = [];
}

// Export for TypeScript
export {};