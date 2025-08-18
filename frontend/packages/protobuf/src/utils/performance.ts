/**
 * @fileoverview Ultra-high performance utilities for protobuf operations
 * 
 * Microsecond-level optimizations for MEV trading systems requiring
 * sub-millisecond decision latency.
 */

/**
 * High-resolution timer for performance measurement
 */
export class HRTimer {
  private start: number = 0;

  /**
   * Start timing measurement
   */
  begin(): void {
    this.start = performance.now();
  }

  /**
   * End timing and return microseconds elapsed
   */
  endMicroseconds(): number {
    return (performance.now() - this.start) * 1000;
  }

  /**
   * End timing and return nanoseconds elapsed (estimated)
   */
  endNanoseconds(): number {
    return this.endMicroseconds() * 1000;
  }
}

/**
 * Memory pool for zero-allocation message processing
 */
export class MessagePool<T> {
  private readonly pool: T[] = [];
  private readonly factory: () => T;
  private readonly reset: (obj: T) => void;

  constructor(factory: () => T, reset: (obj: T) => void, initialSize = 100) {
    this.factory = factory;
    this.reset = reset;
    
    // Pre-allocate objects
    for (let i = 0; i < initialSize; i++) {
      this.pool.push(factory());
    }
  }

  /**
   * Get object from pool (zero allocation if available)
   */
  acquire(): T {
    const obj = this.pool.pop();
    if (obj) {
      this.reset(obj);
      return obj;
    }
    return this.factory();
  }

  /**
   * Return object to pool for reuse
   */
  release(obj: T): void {
    if (this.pool.length < 1000) { // Prevent unbounded growth
      this.pool.push(obj);
    }
  }
}

/**
 * Circular buffer for high-throughput message streaming
 */
export class CircularBuffer<T> {
  private readonly buffer: (T | undefined)[];
  private readonly size: number;
  private head = 0;
  private tail = 0;
  private count = 0;

  constructor(size: number) {
    this.size = size;
    this.buffer = new Array(size);
  }

  /**
   * Add item to buffer (overwrites oldest if full)
   */
  push(item: T): boolean {
    this.buffer[this.tail] = item;
    this.tail = (this.tail + 1) % this.size;
    
    if (this.count < this.size) {
      this.count++;
      return true;
    } else {
      // Buffer full, advance head
      this.head = (this.head + 1) % this.size;
      return false; // Overwrote data
    }
  }

  /**
   * Get next item from buffer
   */
  pop(): T | undefined {
    if (this.count === 0) return undefined;
    
    const item = this.buffer[this.head];
    this.buffer[this.head] = undefined;
    this.head = (this.head + 1) % this.size;
    this.count--;
    
    return item;
  }

  /**
   * Batch pop multiple items
   */
  popBatch(maxItems: number): T[] {
    const items: T[] = [];
    const count = Math.min(maxItems, this.count);
    
    for (let i = 0; i < count; i++) {
      const item = this.pop();
      if (item !== undefined) {
        items.push(item);
      }
    }
    
    return items;
  }

  /**
   * Get current buffer utilization
   */
  utilization(): number {
    return this.count / this.size;
  }

  /**
   * Check if buffer is nearly full (>80%)
   */
  isNearlyFull(): boolean {
    return this.utilization() > 0.8;
  }
}

/**
 * Performance statistics collector
 */
export class PerfStats {
  private readonly samples: number[] = [];
  private readonly maxSamples: number;

  constructor(maxSamples = 10000) {
    this.maxSamples = maxSamples;
  }

  /**
   * Record a latency sample in microseconds
   */
  record(latencyUs: number): void {
    this.samples.push(latencyUs);
    if (this.samples.length > this.maxSamples) {
      this.samples.shift();
    }
  }

  /**
   * Get comprehensive statistics
   */
  getStats() {
    if (this.samples.length === 0) {
      return {
        count: 0,
        mean: 0,
        p50: 0,
        p95: 0,
        p99: 0,
        p999: 0,
        min: 0,
        max: 0,
        stddev: 0
      };
    }

    const sorted = [...this.samples].sort((a, b) => a - b);
    const count = sorted.length;
    const mean = sorted.reduce((sum, x) => sum + x, 0) / count;
    const variance = sorted.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / count;

    return {
      count,
      mean,
      p50: sorted[Math.floor(count * 0.5)],
      p95: sorted[Math.floor(count * 0.95)],
      p99: sorted[Math.floor(count * 0.99)],
      p999: sorted[Math.floor(count * 0.999)],
      min: sorted[0],
      max: sorted[count - 1],
      stddev: Math.sqrt(variance)
    };
  }

  /**
   * Check if performance is within SLA
   */
  checkSLA(p99ThresholdUs: number): boolean {
    const stats = this.getStats();
    return stats.p99 <= p99ThresholdUs;
  }

  /**
   * Reset all statistics
   */
  reset(): void {
    this.samples.length = 0;
  }
}

/**
 * Adaptive batch size optimizer for maximum throughput
 */
export class BatchOptimizer {
  private currentBatchSize = 32;
  private readonly minBatchSize = 1;
  private readonly maxBatchSize = 1024;
  private readonly perfStats = new PerfStats(1000);
  private lastOptimization = 0;
  private readonly optimizationIntervalMs = 5000;

  /**
   * Record processing latency for batch size optimization
   */
  recordBatchLatency(batchSize: number, latencyUs: number): void {
    // Record throughput as messages per microsecond
    const throughput = batchSize / latencyUs;
    this.perfStats.record(throughput * 1_000_000); // Convert to msg/sec
  }

  /**
   * Get optimal batch size based on recent performance
   */
  getOptimalBatchSize(): number {
    const now = Date.now();
    if (now - this.lastOptimization > this.optimizationIntervalMs) {
      this.optimizeBatchSize();
      this.lastOptimization = now;
    }
    return this.currentBatchSize;
  }

  private optimizeBatchSize(): void {
    const stats = this.perfStats.getStats();
    if (stats.count < 10) return; // Need more samples

    // Simple hill climbing algorithm
    if (stats.p95 > stats.mean * 1.5) {
      // High variance, reduce batch size
      this.currentBatchSize = Math.max(
        this.minBatchSize,
        Math.floor(this.currentBatchSize * 0.9)
      );
    } else if (stats.p95 < stats.mean * 1.1) {
      // Low variance, can increase batch size
      this.currentBatchSize = Math.min(
        this.maxBatchSize,
        Math.floor(this.currentBatchSize * 1.1)
      );
    }
  }
}