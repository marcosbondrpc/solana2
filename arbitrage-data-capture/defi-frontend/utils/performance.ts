/**
 * Performance monitoring and optimization utilities
 * Tracks frame rates, memory usage, and provides optimization hints
 */

export interface PerformanceMetrics {
  fps: number;
  frameTime: number;
  memoryUsed: number;
  memoryLimit: number;
  cpuUsage: number;
  renderTime: number;
  idleTime: number;
  droppedFrames: number;
}

/**
 * High-precision performance monitor
 */
export class PerformanceMonitor {
  private frameCount = 0;
  private lastFrameTime = 0;
  private frameTimings: number[] = [];
  private renderTimings: number[] = [];
  private droppedFrames = 0;
  private rafHandle: number | null = null;
  private metricsCallbacks: Set<(metrics: PerformanceMetrics) => void> = new Set();
  private updateInterval = 1000; // Update metrics every second
  private lastUpdate = 0;
  
  // Frame budget (16.67ms for 60 FPS)
  private readonly FRAME_BUDGET = 1000 / 60;
  
  constructor() {
    this.startMonitoring();
  }
  
  /**
   * Start performance monitoring
   */
  private startMonitoring(): void {
    const measureFrame = (timestamp: number) => {
      if (this.lastFrameTime > 0) {
        const frameTime = timestamp - this.lastFrameTime;
        this.frameTimings.push(frameTime);
        
        // Keep last 60 frames
        if (this.frameTimings.length > 60) {
          this.frameTimings.shift();
        }
        
        // Check for dropped frames
        if (frameTime > this.FRAME_BUDGET * 1.5) {
          this.droppedFrames++;
        }
      }
      
      this.lastFrameTime = timestamp;
      this.frameCount++;
      
      // Update metrics periodically
      if (timestamp - this.lastUpdate > this.updateInterval) {
        this.updateMetrics();
        this.lastUpdate = timestamp;
      }
      
      this.rafHandle = requestAnimationFrame(measureFrame);
    };
    
    this.rafHandle = requestAnimationFrame(measureFrame);
  }
  
  /**
   * Update and broadcast metrics
   */
  private updateMetrics(): void {
    const metrics = this.getMetrics();
    this.metricsCallbacks.forEach(callback => {
      try {
        callback(metrics);
      } catch (error) {
        console.error('Performance callback error:', error);
      }
    });
  }
  
  /**
   * Get current performance metrics
   */
  getMetrics(): PerformanceMetrics {
    const avgFrameTime = this.frameTimings.length > 0
      ? this.frameTimings.reduce((a, b) => a + b, 0) / this.frameTimings.length
      : 16.67;
    
    const fps = 1000 / avgFrameTime;
    
    // Get memory info if available
    const memory = (performance as any).memory;
    const memoryUsed = memory ? memory.usedJSHeapSize : 0;
    const memoryLimit = memory ? memory.jsHeapSizeLimit : 0;
    
    // Calculate render time
    const avgRenderTime = this.renderTimings.length > 0
      ? this.renderTimings.reduce((a, b) => a + b, 0) / this.renderTimings.length
      : 0;
    
    return {
      fps: Math.round(fps),
      frameTime: avgFrameTime,
      memoryUsed,
      memoryLimit,
      cpuUsage: (avgFrameTime / this.FRAME_BUDGET) * 100,
      renderTime: avgRenderTime,
      idleTime: Math.max(0, this.FRAME_BUDGET - avgFrameTime),
      droppedFrames: this.droppedFrames
    };
  }
  
  /**
   * Record render timing
   */
  recordRenderTime(ms: number): void {
    this.renderTimings.push(ms);
    if (this.renderTimings.length > 60) {
      this.renderTimings.shift();
    }
  }
  
  /**
   * Subscribe to metrics updates
   */
  onMetrics(callback: (metrics: PerformanceMetrics) => void): () => void {
    this.metricsCallbacks.add(callback);
    return () => {
      this.metricsCallbacks.delete(callback);
    };
  }
  
  /**
   * Stop monitoring
   */
  stop(): void {
    if (this.rafHandle !== null) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = null;
    }
  }
  
  /**
   * Reset metrics
   */
  reset(): void {
    this.frameCount = 0;
    this.frameTimings = [];
    this.renderTimings = [];
    this.droppedFrames = 0;
  }
}

/**
 * Request idle callback with fallback
 */
export function requestIdleCallback(
  callback: (deadline: IdleDeadline) => void,
  options?: { timeout?: number }
): number {
  if ('requestIdleCallback' in window) {
    return (window as any).requestIdleCallback(callback, options);
  }
  
  // Fallback using setTimeout
  const start = Date.now();
  return window.setTimeout(() => {
    callback({
      didTimeout: false,
      timeRemaining: () => Math.max(0, 50 - (Date.now() - start))
    } as IdleDeadline);
  }, 1);
}

/**
 * Cancel idle callback with fallback
 */
export function cancelIdleCallback(handle: number): void {
  if ('cancelIdleCallback' in window) {
    (window as any).cancelIdleCallback(handle);
  } else {
    clearTimeout(handle);
  }
}

/**
 * Debounce function with requestAnimationFrame
 */
export function rafDebounce<T extends (...args: any[]) => void>(
  fn: T,
  delay: number = 0
): T {
  let rafId: number | null = null;
  let timeoutId: number | null = null;
  
  return ((...args: Parameters<T>) => {
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
    }
    if (timeoutId !== null) {
      clearTimeout(timeoutId);
    }
    
    if (delay > 0) {
      timeoutId = window.setTimeout(() => {
        rafId = requestAnimationFrame(() => {
          fn(...args);
          rafId = null;
        });
      }, delay);
    } else {
      rafId = requestAnimationFrame(() => {
        fn(...args);
        rafId = null;
      });
    }
  }) as T;
}

/**
 * Throttle function with trailing call
 */
export function throttle<T extends (...args: any[]) => void>(
  fn: T,
  limit: number
): T {
  let inThrottle = false;
  let lastArgs: Parameters<T> | null = null;
  
  return ((...args: Parameters<T>) => {
    lastArgs = args;
    
    if (!inThrottle) {
      fn(...args);
      inThrottle = true;
      
      setTimeout(() => {
        inThrottle = false;
        if (lastArgs !== null) {
          fn(...lastArgs);
          lastArgs = null;
        }
      }, limit);
    }
  }) as T;
}

/**
 * Memory pool for object reuse
 */
export class ObjectPool<T> {
  private pool: T[] = [];
  private factory: () => T;
  private reset: (obj: T) => void;
  private maxSize: number;
  
  constructor(
    factory: () => T,
    reset: (obj: T) => void,
    maxSize: number = 100
  ) {
    this.factory = factory;
    this.reset = reset;
    this.maxSize = maxSize;
  }
  
  /**
   * Get object from pool or create new one
   */
  acquire(): T {
    if (this.pool.length > 0) {
      return this.pool.pop()!;
    }
    return this.factory();
  }
  
  /**
   * Return object to pool
   */
  release(obj: T): void {
    if (this.pool.length < this.maxSize) {
      this.reset(obj);
      this.pool.push(obj);
    }
  }
  
  /**
   * Clear the pool
   */
  clear(): void {
    this.pool = [];
  }
  
  /**
   * Get pool size
   */
  size(): number {
    return this.pool.length;
  }
}

/**
 * Batch updates using microtasks
 */
export class UpdateBatcher {
  private pending = new Set<() => void>();
  private scheduled = false;
  
  /**
   * Add update to batch
   */
  add(update: () => void): void {
    this.pending.add(update);
    
    if (!this.scheduled) {
      this.scheduled = true;
      queueMicrotask(() => this.flush());
    }
  }
  
  /**
   * Flush all pending updates
   */
  private flush(): void {
    const updates = Array.from(this.pending);
    this.pending.clear();
    this.scheduled = false;
    
    updates.forEach(update => {
      try {
        update();
      } catch (error) {
        console.error('Batch update error:', error);
      }
    });
  }
  
  /**
   * Clear pending updates
   */
  clear(): void {
    this.pending.clear();
    this.scheduled = false;
  }
}

/**
 * Virtual list for rendering large datasets
 */
export class VirtualList<T> {
  private items: T[] = [];
  private itemHeight: number;
  private containerHeight: number;
  private scrollTop = 0;
  private overscan = 3;
  
  constructor(itemHeight: number, containerHeight: number) {
    this.itemHeight = itemHeight;
    this.containerHeight = containerHeight;
  }
  
  /**
   * Update items
   */
  setItems(items: T[]): void {
    this.items = items;
  }
  
  /**
   * Update scroll position
   */
  setScrollTop(scrollTop: number): void {
    this.scrollTop = scrollTop;
  }
  
  /**
   * Get visible range
   */
  getVisibleRange(): { start: number; end: number; items: T[] } {
    const visibleCount = Math.ceil(this.containerHeight / this.itemHeight);
    const start = Math.max(0, Math.floor(this.scrollTop / this.itemHeight) - this.overscan);
    const end = Math.min(
      this.items.length,
      start + visibleCount + this.overscan * 2
    );
    
    return {
      start,
      end,
      items: this.items.slice(start, end)
    };
  }
  
  /**
   * Get total height
   */
  getTotalHeight(): number {
    return this.items.length * this.itemHeight;
  }
  
  /**
   * Get offset for visible items
   */
  getOffset(): number {
    const { start } = this.getVisibleRange();
    return start * this.itemHeight;
  }
}

// Export singleton monitor
export const performanceMonitor = new PerformanceMonitor();