/**
 * Advanced Memory Management Hook for High-Frequency Data Streams
 * Prevents memory leaks and maintains optimal performance with 10,000+ updates/sec
 */

import { useRef, useEffect, useCallback, useMemo } from 'react';

interface MemoryConfig {
  maxHistoricalPoints?: number;
  maxWebSocketMessages?: number;
  cleanupInterval?: number;
  enableWeakCache?: boolean;
  enableAutoQualityAdjustment?: boolean;
}

interface MemoryStats {
  usedJSHeapSize: number;
  totalJSHeapSize: number;
  jsHeapSizeLimit: number;
  pressure: number; // 0-1 scale
}

// Circular buffer for efficient memory usage
class CircularBuffer<T> {
  private buffer: (T | undefined)[];
  private head = 0;
  private tail = 0;
  private size = 0;
  private capacity: number;

  constructor(capacity: number) {
    this.capacity = capacity;
    this.buffer = new Array(capacity);
  }

  push(item: T): void {
    this.buffer[this.tail] = item;
    this.tail = (this.tail + 1) % this.capacity;
    
    if (this.size < this.capacity) {
      this.size++;
    } else {
      this.head = (this.head + 1) % this.capacity;
    }
  }

  getAll(): T[] {
    const result: T[] = [];
    let current = this.head;
    
    for (let i = 0; i < this.size; i++) {
      const val = this.buffer[current];
      if (val !== undefined) {
        result.push(val as T);
      }
      current = (current + 1) % this.capacity;
    }
    
    return result;
  }

  clear(): void {
    this.head = 0;
    this.tail = 0;
    this.size = 0;
    this.buffer = new Array(this.capacity);
  }

  getSize(): number {
    return this.size;
  }

  isFull(): boolean {
    return this.size === this.capacity;
  }

  resize(newCapacity: number): void {
    const items = this.getAll();
    this.capacity = newCapacity;
    this.buffer = new Array(newCapacity);
    this.head = 0;
    this.tail = 0;
    this.size = 0;
    
    // Re-add items up to new capacity
    const itemsToAdd = Math.min(items.length, newCapacity);
    for (let i = items.length - itemsToAdd; i < items.length; i++) {
      this.push(items[i]);
    }
  }
}

// LRU Cache with WeakMap for component caches
class LRUCache<K, V> {
  private cache: Map<K, { value: V; timestamp: number }>;
  private accessOrder: K[];
  private maxSize: number;
  private ttl: number;

  constructor(maxSize: number, ttl: number = Infinity) {
    this.cache = new Map();
    this.accessOrder = [];
    this.maxSize = maxSize;
    this.ttl = ttl;
  }

  get(key: K): V | undefined {
    const entry = this.cache.get(key);
    
    if (!entry) return undefined;
    
    // Check TTL
    if (Date.now() - entry.timestamp > this.ttl) {
      this.delete(key);
      return undefined;
    }
    
    // Update access order
    const index = this.accessOrder.indexOf(key);
    if (index > -1) {
      this.accessOrder.splice(index, 1);
    }
    this.accessOrder.push(key);
    
    return entry.value;
  }

  set(key: K, value: V): void {
    // Remove oldest if at capacity
    if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
      const oldest = this.accessOrder.shift();
      if (oldest !== undefined) {
        this.cache.delete(oldest);
      }
    }
    
    this.cache.set(key, { value, timestamp: Date.now() });
    
    // Update access order
    const index = this.accessOrder.indexOf(key);
    if (index > -1) {
      this.accessOrder.splice(index, 1);
    }
    this.accessOrder.push(key);
  }

  delete(key: K): boolean {
    const index = this.accessOrder.indexOf(key);
    if (index > -1) {
      this.accessOrder.splice(index, 1);
    }
    return this.cache.delete(key);
  }

  clear(): void {
    this.cache.clear();
    this.accessOrder = [];
  }

  size(): number {
    return this.cache.size;
  }

  // Clean up expired entries
  cleanup(): void {
    const now = Date.now();
    const keysToDelete: K[] = [];
    
    this.cache.forEach((entry, key) => {
      if (now - entry.timestamp > this.ttl) {
        keysToDelete.push(key);
      }
    });
    
    keysToDelete.forEach(key => this.delete(key));
  }
}

export function useMemoryManager(config: MemoryConfig = {}) {
  const {
    maxHistoricalPoints = 10000,
    maxWebSocketMessages = 5000,
    cleanupInterval = 30000, // 30 seconds
    enableWeakCache = true,
    enableAutoQualityAdjustment = true
  } = config;

  // Memory pressure tracking
  const memoryStatsRef = useRef<MemoryStats>({
    usedJSHeapSize: 0,
    totalJSHeapSize: 0,
    jsHeapSizeLimit: 0,
    pressure: 0
  });

  // Data structures
  const historicalDataRef = useRef(new CircularBuffer<any>(maxHistoricalPoints));
  const wsMessagesRef = useRef(new CircularBuffer<any>(maxWebSocketMessages));
  const componentCacheRef = useRef(new LRUCache<string, any>(100, 60000)); // 1 minute TTL
  const weakCacheRef = useRef(enableWeakCache ? new WeakMap() : null);
  
  // Quality settings for auto-adjustment
  const qualitySettingsRef = useRef({
    updateFrequency: 16, // ms
    maxDataPoints: maxHistoricalPoints,
    enableAnimations: true,
    enableHighQualityCharts: true,
    enableRealtimeUpdates: true
  });

  // Cleanup timer
  const cleanupTimerRef = useRef<number>();

  // Monitor memory usage
  const checkMemoryPressure = useCallback((): MemoryStats => {
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      const used = memory.usedJSHeapSize;
      const total = memory.totalJSHeapSize;
      const limit = memory.jsHeapSizeLimit;
      
      const pressure = used / limit;
      
      memoryStatsRef.current = {
        usedJSHeapSize: used,
        totalJSHeapSize: total,
        jsHeapSizeLimit: limit,
        pressure
      };
      
      // Auto-adjust quality if enabled
      if (enableAutoQualityAdjustment) {
        adjustQualitySettings(pressure);
      }
      
      return memoryStatsRef.current;
    }
    
    return memoryStatsRef.current;
  }, [enableAutoQualityAdjustment]);

  // Adjust quality settings based on memory pressure
  const adjustQualitySettings = useCallback((pressure: number) => {
    const settings = qualitySettingsRef.current;
    
    if (pressure > 0.9) {
      // Critical pressure - minimum quality
      settings.updateFrequency = 100;
      settings.maxDataPoints = 1000;
      settings.enableAnimations = false;
      settings.enableHighQualityCharts = false;
      settings.enableRealtimeUpdates = false;
      
      // Force cleanup
      performCleanup();
      
      console.warn('Critical memory pressure detected, reducing quality settings');
    } else if (pressure > 0.7) {
      // High pressure - reduced quality
      settings.updateFrequency = 50;
      settings.maxDataPoints = 5000;
      settings.enableAnimations = false;
      settings.enableHighQualityCharts = false;
      settings.enableRealtimeUpdates = true;
    } else if (pressure > 0.5) {
      // Moderate pressure - balanced quality
      settings.updateFrequency = 33;
      settings.maxDataPoints = 7500;
      settings.enableAnimations = true;
      settings.enableHighQualityCharts = false;
      settings.enableRealtimeUpdates = true;
    } else {
      // Low pressure - maximum quality
      settings.updateFrequency = 16;
      settings.maxDataPoints = maxHistoricalPoints;
      settings.enableAnimations = true;
      settings.enableHighQualityCharts = true;
      settings.enableRealtimeUpdates = true;
    }
    
    // Resize buffers if needed
    if (settings.maxDataPoints !== historicalDataRef.current['capacity']) {
      historicalDataRef.current.resize(settings.maxDataPoints);
    }
  }, [maxHistoricalPoints]);

  // Perform cleanup
  const performCleanup = useCallback(() => {
    // Clean component cache
    componentCacheRef.current.cleanup();
    
    // Trim buffers if over 80% full
    if (historicalDataRef.current.getSize() > maxHistoricalPoints * 0.8) {
      const data = historicalDataRef.current.getAll();
      historicalDataRef.current.clear();
      
      // Keep only recent half
      const keepCount = Math.floor(data.length / 2);
      data.slice(-keepCount).forEach(item => {
        historicalDataRef.current.push(item);
      });
    }
    
    // Force garbage collection if available (Chrome with --expose-gc flag)
    if (typeof (globalThis as any).gc === 'function') {
      (globalThis as any).gc();
    }
    
    // Check memory after cleanup
    const stats = checkMemoryPressure();
    
    return stats;
  }, [maxHistoricalPoints, checkMemoryPressure]);

  // Add data to historical buffer
  const addHistoricalData = useCallback((data: any) => {
    historicalDataRef.current.push(data);
  }, []);

  // Add WebSocket message
  const addWebSocketMessage = useCallback((message: any) => {
    wsMessagesRef.current.push(message);
  }, []);

  // Get cached component data
  const getCached = useCallback(<T,>(key: string, factory: () => T): T => {
    let value = componentCacheRef.current.get(key);
    
    if (value === undefined) {
      value = factory();
      componentCacheRef.current.set(key, value);
    }
    
    return value as T;
  }, []);

  // Store in weak cache
  const setWeakCached = useCallback((key: object, value: any) => {
    if (weakCacheRef.current) {
      weakCacheRef.current.set(key, value);
    }
  }, []);

  // Get from weak cache
  const getWeakCached = useCallback((key: object) => {
    return weakCacheRef.current?.get(key);
  }, []);

  // Get current memory stats
  const getMemoryStats = useCallback((): MemoryStats => {
    return { ...memoryStatsRef.current };
  }, []);

  // Get quality settings
  const getQualitySettings = useCallback(() => {
    return { ...qualitySettingsRef.current };
  }, []);

  // Get buffer sizes
  const getBufferStats = useCallback(() => {
    return {
      historicalData: {
        size: historicalDataRef.current.getSize(),
        capacity: historicalDataRef.current['capacity'],
        isFull: historicalDataRef.current.isFull()
      },
      wsMessages: {
        size: wsMessagesRef.current.getSize(),
        capacity: wsMessagesRef.current['capacity'],
        isFull: wsMessagesRef.current.isFull()
      },
      componentCache: {
        size: componentCacheRef.current.size()
      }
    };
  }, []);

  // Clear all caches
  const clearAll = useCallback(() => {
    historicalDataRef.current.clear();
    wsMessagesRef.current.clear();
    componentCacheRef.current.clear();
    
    if (weakCacheRef.current) {
      weakCacheRef.current = new WeakMap();
    }
    
    performCleanup();
  }, [performCleanup]);

  // Setup cleanup interval
  useEffect(() => {
    cleanupTimerRef.current = window.setInterval(() => {
      performCleanup();
    }, cleanupInterval);
    
    // Initial memory check
    checkMemoryPressure();
    
    // Monitor memory pressure more frequently
    const pressureTimer = window.setInterval(() => {
      checkMemoryPressure();
    }, 5000);
    
    return () => {
      if (cleanupTimerRef.current) {
        clearInterval(cleanupTimerRef.current);
      }
      clearInterval(pressureTimer);
    };
  }, [cleanupInterval, performCleanup, checkMemoryPressure]);

  // Public API
  return useMemo(() => ({
    // Data management
    addHistoricalData,
    addWebSocketMessage,
    getCached,
    setWeakCached,
    getWeakCached,
    
    // Memory monitoring
    getMemoryStats,
    getQualitySettings,
    getBufferStats,
    checkMemoryPressure,
    
    // Cleanup
    performCleanup,
    clearAll,
    
    // Direct buffer access (use carefully)
    buffers: {
      historical: historicalDataRef.current,
      wsMessages: wsMessagesRef.current
    }
  }), [
    addHistoricalData,
    addWebSocketMessage,
    getCached,
    setWeakCached,
    getWeakCached,
    getMemoryStats,
    getQualitySettings,
    getBufferStats,
    checkMemoryPressure,
    performCleanup,
    clearAll
  ]);
}

// Export types
export type { MemoryConfig, MemoryStats };
export { CircularBuffer, LRUCache };