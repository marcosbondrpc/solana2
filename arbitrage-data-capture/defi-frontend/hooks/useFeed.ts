/**
 * High-performance React hooks for feed store
 * Optimized for minimal re-renders and maximum throughput
 */

import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { useSnapshot, subscribe } from 'valtio';
import { subscribeKey } from 'valtio/utils';
import {
  feedStore,
  getMevOpportunities,
  getArbOpportunities,
  getBundleOutcomes,
  subscribeToMev,
  subscribeToMetric,
  updateFilters,
  type MevOpportunity,
  type ArbitrageOpportunity,
  type BundleOutcome,
  type AggregatedMetrics
} from '../stores/feed';
import { rafDebounce, throttle } from '../utils/performance';

/**
 * Hook for connection status
 */
export function useConnectionStatus() {
  const snap = useSnapshot(feedStore.connection);
  return snap;
}

/**
 * Hook for MEV opportunities with virtual scrolling support
 */
export function useMevOpportunities(
  limit?: number,
  offset?: number
) {
  const [opportunities, setOpportunities] = useState<MevOpportunity[]>([]);
  const [loading, setLoading] = useState(true);
  const unsubRef = useRef<(() => void) | null>(null);
  
  useEffect(() => {
    // Initial load
    const initial = getMevOpportunities();
    setOpportunities(initial.slice(offset || 0, (offset || 0) + (limit || initial.length)));
    setLoading(false);
    
    // Subscribe to updates with debouncing
    const debouncedUpdate = rafDebounce(() => {
      const all = getMevOpportunities();
      setOpportunities(all.slice(offset || 0, (offset || 0) + (limit || all.length)));
    }, 50);
    
    unsubRef.current = subscribeToMev(debouncedUpdate);
    
    return () => {
      unsubRef.current?.();
    };
  }, [limit, offset]);
  
  return { opportunities, loading };
}

/**
 * Hook for arbitrage opportunities
 */
export function useArbOpportunities(
  limit?: number,
  offset?: number
) {
  const [opportunities, setOpportunities] = useState<ArbitrageOpportunity[]>([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    setLoading(true);
    
    const update = throttle(() => {
      const all = getArbOpportunities();
      setOpportunities(all.slice(offset || 0, (offset || 0) + (limit || all.length)));
      setLoading(false);
    }, 100);
    
    // Initial update
    update();
    
    // Subscribe to changes
    const unsub = subscribe(feedStore, update);
    
    return () => {
      unsub();
    };
  }, [limit, offset]);
  
  return { opportunities, loading };
}

/**
 * Hook for bundle outcomes
 */
export function useBundleOutcomes(
  onlySuccessful: boolean = false
) {
  const [outcomes, setOutcomes] = useState<BundleOutcome[]>([]);
  
  useEffect(() => {
    const update = () => {
      const all = getBundleOutcomes();
      const filtered = onlySuccessful 
        ? all.filter(o => o?.landed)
        : all;
      setOutcomes(filtered);
    };
    
    // Initial update
    update();
    
    // Subscribe with throttling
    const throttledUpdate = throttle(update, 250);
    const unsub = subscribe(feedStore, throttledUpdate);
    
    return unsub;
  }, [onlySuccessful]);
  
  return outcomes;
}

/**
 * Hook for current metrics
 */
export function useCurrentMetrics() {
  const snap = useSnapshot(feedStore.currentMetrics);
  return snap;
}

/**
 * Hook for specific metric with subscription
 */
export function useMetric<K extends keyof typeof feedStore.currentMetrics>(
  key: K
): number {
  const [value, setValue] = useState(feedStore.currentMetrics[key]);
  
  useEffect(() => {
    setValue(feedStore.currentMetrics[key]);
    
    const unsub = subscribeToMetric(key, setValue);
    return unsub;
  }, [key]);
  
  return value;
}

/**
 * Hook for aggregated metrics history
 */
export function useAggregatedMetrics(
  windowSize?: number
): AggregatedMetrics[] {
  const snap = useSnapshot(feedStore.aggregatedMetrics);
  
  return useMemo(() => {
    if (!windowSize) return snap as AggregatedMetrics[];
    return (snap as AggregatedMetrics[]).slice(-windowSize);
  }, [snap, windowSize]);
}

/**
 * Hook for market ticks
 */
export function useMarketTicks(
  marketIds?: string[]
): Map<string, any> {
  const [ticks, setTicks] = useState(new Map());
  
  useEffect(() => {
    const update = () => {
      if (marketIds && marketIds.length > 0) {
        const filtered = new Map();
        marketIds.forEach(id => {
          const tick = feedStore.marketTicks.get(id);
          if (tick) filtered.set(id, tick);
        });
        setTicks(filtered);
      } else {
        setTicks(new Map(feedStore.marketTicks));
      }
    };
    
    // Initial update
    update();
    
    // Subscribe with debouncing for market ticks
    const debouncedUpdate = rafDebounce(update, 100);
    const unsub = subscribe(feedStore.marketTicks, debouncedUpdate);
    
    return unsub;
  }, [marketIds?.join(',')]);
  
  return ticks;
}

/**
 * Hook for filters
 */
export function useFilters() {
  const snap = useSnapshot(feedStore.filters);
  
  const setFilters = useCallback((filters: Partial<typeof feedStore.filters>) => {
    updateFilters(filters);
  }, []);
  
  return [snap, setFilters] as const;
}

/**
 * Hook for performance stats
 */
export function usePerformanceStats() {
  const snap = useSnapshot(feedStore.performance);
  const connectionStats = useSnapshot(feedStore.connection.stats);
  
  return {
    ...snap,
    connection: connectionStats
  };
}

/**
 * Hook for virtual list of opportunities
 */
export function useVirtualList<T>(
  items: T[],
  itemHeight: number,
  containerHeight: number,
  overscan: number = 3
) {
  const [scrollTop, setScrollTop] = useState(0);
  
  const visibleRange = useMemo(() => {
    const visibleCount = Math.ceil(containerHeight / itemHeight);
    const start = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
    const end = Math.min(items.length, start + visibleCount + overscan * 2);
    
    return {
      start,
      end,
      items: items.slice(start, end),
      totalHeight: items.length * itemHeight,
      offsetY: start * itemHeight
    };
  }, [items, itemHeight, containerHeight, scrollTop, overscan]);
  
  const handleScroll = useCallback((e: Event) => {
    const target = e.target as HTMLElement;
    setScrollTop(target.scrollTop);
  }, []);
  
  return {
    visibleRange,
    handleScroll,
    setScrollTop
  };
}

/**
 * Hook for auto-refresh with configurable interval
 */
export function useAutoRefresh(
  callback: () => void,
  interval: number = 1000,
  enabled: boolean = true
) {
  const callbackRef = useRef(callback);
  
  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);
  
  useEffect(() => {
    if (!enabled) return;
    
    const id = setInterval(() => {
      callbackRef.current();
    }, interval);
    
    return () => clearInterval(id);
  }, [interval, enabled]);
}

/**
 * Hook for WebWorker communication
 */
export function useWebWorker<T, R>(
  workerFactory: () => Worker,
  deps: any[] = []
) {
  const workerRef = useRef<Worker | null>(null);
  const [result, setResult] = useState<R | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(false);
  
  useEffect(() => {
    const worker = workerFactory();
    workerRef.current = worker;
    
    worker.onmessage = (e) => {
      setResult(e.data);
      setLoading(false);
    };
    
    worker.onerror = (e) => {
      setError(new Error(e.message));
      setLoading(false);
    };
    
    return () => {
      worker.terminate();
    };
  }, deps);
  
  const postMessage = useCallback((data: T) => {
    if (workerRef.current) {
      setLoading(true);
      setError(null);
      workerRef.current.postMessage(data);
    }
  }, []);
  
  return { postMessage, result, error, loading };
}