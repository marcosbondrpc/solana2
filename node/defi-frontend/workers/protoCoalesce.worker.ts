/**
 * High-Performance Protobuf Coalescing Worker
 * Reduces postMessage overhead by batching and compressing updates
 */

import { performance } from 'perf_hooks';

// Message types
enum MessageType {
  POOL_UPDATE = 1,
  TRANSACTION = 2,
  MEV_SIGNAL = 3,
  METRICS = 4,
  CONTROL = 5,
}

interface WorkerMessage {
  type: MessageType;
  payload: any;
  timestamp: number;
  sequence: number;
}

interface CoalescedBatch {
  startTime: number;
  endTime: number;
  messageCount: number;
  compressed: boolean;
  data: Uint8Array | any[];
}

class ProtoCoalesceWorker {
  private messageBuffer: WorkerMessage[] = [];
  private sequenceNumber = 0;
  private batchSize = 100;
  private batchIntervalMs = 16; // ~60fps
  private compressionThreshold = 1024; // bytes
  private lastBatchTime = performance.now();
  private metrics = {
    messagesProcessed: 0,
    batchesSent: 0,
    bytesCompressed: 0,
    compressionRatio: 0,
  };
  
  // Ring buffer for ultra-fast queuing
  private ringBuffer: RingBuffer<WorkerMessage>;
  
  // Preallocated buffers to avoid GC
  private scratchBuffer: ArrayBuffer;
  private encoderBuffer: Uint8Array;
  
  constructor() {
    this.ringBuffer = new RingBuffer(10000);
    this.scratchBuffer = new ArrayBuffer(65536); // 64KB
    this.encoderBuffer = new Uint8Array(65536);
    
    // Start processing loop
    this.startProcessingLoop();
    
    // Setup message handler
    self.onmessage = this.handleMessage.bind(this);
  }
  
  private handleMessage(event: MessageEvent) {
    const { type, payload } = event.data;
    
    switch (type) {
      case 'DATA':
        this.enqueueMessage(payload);
        break;
      case 'CONFIG':
        this.updateConfig(payload);
        break;
      case 'FLUSH':
        this.flush();
        break;
      case 'METRICS':
        this.sendMetrics();
        break;
    }
  }
  
  private enqueueMessage(data: any) {
    const message: WorkerMessage = {
      type: this.detectMessageType(data),
      payload: data,
      timestamp: performance.now(),
      sequence: this.sequenceNumber++,
    };
    
    // Try to add to ring buffer
    if (!this.ringBuffer.push(message)) {
      // Buffer full, force flush
      this.processBatch();
    }
    
    this.metrics.messagesProcessed++;
  }
  
  private detectMessageType(data: any): MessageType {
    // Simple heuristic for message type detection
    if (data.pool_address) return MessageType.POOL_UPDATE;
    if (data.signature) return MessageType.TRANSACTION;
    if (data.signal_type) return MessageType.MEV_SIGNAL;
    if (data.metric_name) return MessageType.METRICS;
    return MessageType.CONTROL;
  }
  
  private startProcessingLoop() {
    const processLoop = () => {
      const now = performance.now();
      
      if (now - this.lastBatchTime >= this.batchIntervalMs) {
        this.processBatch();
        this.lastBatchTime = now;
      }
      
      // Use setImmediate for better performance than setTimeout
      if (typeof setImmediate !== 'undefined') {
        setImmediate(processLoop);
      } else {
        setTimeout(processLoop, 0);
      }
    };
    
    processLoop();
  }
  
  private processBatch() {
    const messages = this.ringBuffer.drain();
    
    if (messages.length === 0) return;
    
    // Group messages by type for better compression
    const grouped = this.groupMessagesByType(messages);
    
    // Coalesce each group
    const coalesced = this.coalesceGroups(grouped);
    
    // Send coalesced batch
    this.sendBatch(coalesced);
    
    this.metrics.batchesSent++;
  }
  
  private groupMessagesByType(messages: WorkerMessage[]): Map<MessageType, WorkerMessage[]> {
    const groups = new Map<MessageType, WorkerMessage[]>();
    
    for (const msg of messages) {
      if (!groups.has(msg.type)) {
        groups.set(msg.type, []);
      }
      groups.get(msg.type)!.push(msg);
    }
    
    return groups;
  }
  
  private coalesceGroups(groups: Map<MessageType, WorkerMessage[]>): CoalescedBatch {
    const coalescedData: any = {};
    let totalSize = 0;
    
    // Special handling for each message type
    groups.forEach((messages, type) => {
      switch (type) {
        case MessageType.POOL_UPDATE:
          coalescedData.poolUpdates = this.coalescePoolUpdates(messages);
          break;
        case MessageType.TRANSACTION:
          coalescedData.transactions = this.coalesceTransactions(messages);
          break;
        case MessageType.MEV_SIGNAL:
          coalescedData.mevSignals = this.coalesceMevSignals(messages);
          break;
        case MessageType.METRICS:
          coalescedData.metrics = this.coalesceMetrics(messages);
          break;
        default:
          coalescedData.control = messages.map(m => m.payload);
      }
    });
    
    // Calculate size for compression decision
    const jsonStr = JSON.stringify(coalescedData);
    totalSize = jsonStr.length;
    
    // Compress if over threshold
    let compressed = false;
    let data: Uint8Array | any[] = [coalescedData];
    
    if (totalSize > this.compressionThreshold) {
      data = this.compress(jsonStr);
      compressed = true;
      
      this.metrics.bytesCompressed += totalSize;
      this.metrics.compressionRatio = data.length / totalSize;
    }
    
    return {
      startTime: messages[0]?.timestamp || performance.now(),
      endTime: performance.now(),
      messageCount: messages.length,
      compressed,
      data,
    };
  }
  
  private coalescePoolUpdates(messages: WorkerMessage[]): any {
    // Delta compression for pool updates
    const pools = new Map<string, any>();
    
    for (const msg of messages) {
      const update = msg.payload;
      const poolId = update.pool_address;
      
      if (!pools.has(poolId)) {
        pools.set(poolId, {
          address: poolId,
          updates: [],
          latestState: null,
        });
      }
      
      const pool = pools.get(poolId)!;
      
      // Store only deltas from previous state
      if (pool.latestState) {
        const delta = this.calculateDelta(pool.latestState, update);
        pool.updates.push({
          t: msg.timestamp,
          d: delta,
        });
      } else {
        pool.updates.push({
          t: msg.timestamp,
          d: update,
        });
      }
      
      pool.latestState = update;
    }
    
    return Array.from(pools.values());
  }
  
  private coalesceTransactions(messages: WorkerMessage[]): any {
    // Group by status for better compression
    const byStatus = {
      pending: [] as any[],
      confirmed: [] as any[],
      failed: [] as any[],
    };
    
    for (const msg of messages) {
      const tx = msg.payload;
      const status = tx.status || 'pending';
      
      // Extract only essential fields
      byStatus[status].push({
        s: tx.signature.substring(0, 8), // Short signature
        t: msg.timestamp,
        p: tx.profit,
        g: tx.gas_used,
      });
    }
    
    return byStatus;
  }
  
  private coalesceMevSignals(messages: WorkerMessage[]): any {
    // Aggregate similar signals
    const signals = new Map<string, any>();
    
    for (const msg of messages) {
      const signal = msg.payload;
      const key = `${signal.signal_type}_${signal.pool_address}`;
      
      if (!signals.has(key)) {
        signals.set(key, {
          type: signal.signal_type,
          pool: signal.pool_address,
          opportunities: [],
        });
      }
      
      signals.get(key)!.opportunities.push({
        t: msg.timestamp,
        p: signal.estimated_profit,
        g: signal.gas_estimate,
      });
    }
    
    return Array.from(signals.values());
  }
  
  private coalesceMetrics(messages: WorkerMessage[]): any {
    // Aggregate metrics by name
    const metrics = new Map<string, any>();
    
    for (const msg of messages) {
      const metric = msg.payload;
      const name = metric.metric_name;
      
      if (!metrics.has(name)) {
        metrics.set(name, {
          name,
          values: [],
          min: Infinity,
          max: -Infinity,
          sum: 0,
          count: 0,
        });
      }
      
      const agg = metrics.get(name)!;
      const value = metric.value;
      
      agg.values.push([msg.timestamp, value]);
      agg.min = Math.min(agg.min, value);
      agg.max = Math.max(agg.max, value);
      agg.sum += value;
      agg.count++;
    }
    
    // Calculate averages
    metrics.forEach((agg) => {
      agg.avg = agg.sum / agg.count;
      // Keep only every Nth value for large datasets
      if (agg.values.length > 100) {
        const step = Math.ceil(agg.values.length / 100);
        agg.values = agg.values.filter((_: any, i: number) => i % step === 0);
      }
    });
    
    return Array.from(metrics.values());
  }
  
  private calculateDelta(prev: any, curr: any): any {
    const delta: any = {};
    
    for (const key in curr) {
      if (curr[key] !== prev[key]) {
        if (typeof curr[key] === 'number' && typeof prev[key] === 'number') {
          // Store as delta for numbers
          delta[key] = curr[key] - prev[key];
        } else {
          // Store full value for non-numbers
          delta[key] = curr[key];
        }
      }
    }
    
    return delta;
  }
  
  private compress(data: string): Uint8Array {
    // Simple LZ-string style compression
    // In production, use pako or similar
    const encoder = new TextEncoder();
    const bytes = encoder.encode(data);
    
    // Simple RLE compression for demonstration
    const compressed: number[] = [];
    let i = 0;
    
    while (i < bytes.length) {
      let runLength = 1;
      const currentByte = bytes[i];
      
      while (i + runLength < bytes.length && bytes[i + runLength] === currentByte && runLength < 255) {
        runLength++;
      }
      
      if (runLength > 3) {
        compressed.push(0xFF); // RLE marker
        compressed.push(runLength);
        compressed.push(currentByte);
        i += runLength;
      } else {
        compressed.push(currentByte);
        i++;
      }
    }
    
    return new Uint8Array(compressed);
  }
  
  private sendBatch(batch: CoalescedBatch) {
    // Transfer data to main thread
    const message = {
      type: 'BATCH',
      batch,
      metrics: { ...this.metrics },
    };
    
    if (batch.compressed && batch.data instanceof Uint8Array) {
      // Transfer buffer for zero-copy
      self.postMessage(message, [batch.data.buffer]);
    } else {
      self.postMessage(message);
    }
  }
  
  private flush() {
    this.processBatch();
  }
  
  private sendMetrics() {
    self.postMessage({
      type: 'METRICS_RESPONSE',
      metrics: { ...this.metrics },
    });
  }
  
  private updateConfig(config: any) {
    if (config.batchSize) this.batchSize = config.batchSize;
    if (config.batchIntervalMs) this.batchIntervalMs = config.batchIntervalMs;
    if (config.compressionThreshold) this.compressionThreshold = config.compressionThreshold;
  }
}

// Fast ring buffer implementation
class RingBuffer<T> {
  private buffer: (T | undefined)[];
  private head = 0;
  private tail = 0;
  private size = 0;
  
  constructor(private capacity: number) {
    this.buffer = new Array(capacity);
  }
  
  push(item: T): boolean {
    if (this.size === this.capacity) {
      return false; // Buffer full
    }
    
    this.buffer[this.tail] = item;
    this.tail = (this.tail + 1) % this.capacity;
    this.size++;
    return true;
  }
  
  pop(): T | undefined {
    if (this.size === 0) {
      return undefined;
    }
    
    const item = this.buffer[this.head];
    this.buffer[this.head] = undefined; // Help GC
    this.head = (this.head + 1) % this.capacity;
    this.size--;
    return item;
  }
  
  drain(): T[] {
    const items: T[] = [];
    
    while (this.size > 0) {
      const item = this.pop();
      if (item !== undefined) {
        items.push(item);
      }
    }
    
    return items;
  }
  
  isEmpty(): boolean {
    return this.size === 0;
  }
}

// Initialize worker
const worker = new ProtoCoalesceWorker();

// Export for TypeScript
export default ProtoCoalesceWorker;