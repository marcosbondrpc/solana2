import { useEffect, useRef, useCallback } from 'react';

interface PerformanceMetrics {
  renderTime: number;
  updateTime: number;
  fps: number;
  memoryUsage?: number;
  lastUpdate: number;
}

export function usePerformanceMonitor(componentName: string) {
  const metricsRef = useRef<PerformanceMetrics>({
    renderTime: 0,
    updateTime: 0,
    fps: 60,
    lastUpdate: Date.now(),
  });
  
  const frameCountRef = useRef(0);
  const lastFrameTimeRef = useRef(performance.now());
  const rafIdRef = useRef<number>();
  
  // Measure FPS
  const measureFPS = useCallback(() => {
    const currentTime = performance.now();
    const delta = currentTime - lastFrameTimeRef.current;
    
    if (delta >= 1000) {
      metricsRef.current.fps = Math.round((frameCountRef.current * 1000) / delta);
      frameCountRef.current = 0;
      lastFrameTimeRef.current = currentTime;
      
      // Log performance if in development
      if (process.env.NODE_ENV === 'development') {
        if (metricsRef.current.fps < 30) {
          console.warn(`[PERF] ${componentName} - Low FPS: ${metricsRef.current.fps}`);
        }
      }
    }
    
    frameCountRef.current++;
    rafIdRef.current = requestAnimationFrame(measureFPS);
  }, [componentName]);
  
  // Measure render time
  const measureRender = useCallback(() => {
    const startTime = performance.now();
    
    return () => {
      const endTime = performance.now();
      const renderTime = endTime - startTime;
      metricsRef.current.renderTime = renderTime;
      
      // Alert if render takes too long
      if (renderTime > 16.67) { // More than one frame at 60fps
        if (process.env.NODE_ENV === 'development') {
          console.warn(`[PERF] ${componentName} - Slow render: ${renderTime.toFixed(2)}ms`);
        }
      }
    };
  }, [componentName]);
  
  // Measure update time
  const measureUpdate = useCallback((updateFn: () => void | Promise<void>) => {
    const startTime = performance.now();
    
    const execute = async () => {
      await updateFn();
      const endTime = performance.now();
      const updateTime = endTime - startTime;
      metricsRef.current.updateTime = updateTime;
      metricsRef.current.lastUpdate = Date.now();
      
      // Alert if update takes too long
      if (updateTime > 100) { // Target sub-100ms updates
        if (process.env.NODE_ENV === 'development') {
          console.warn(`[PERF] ${componentName} - Slow update: ${updateTime.toFixed(2)}ms`);
        }
      }
    };
    
    execute();
  }, [componentName]);
  
  // Measure memory usage
  const measureMemory = useCallback(() => {
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      const usedMemory = memory.usedJSHeapSize / 1048576; // Convert to MB
      metricsRef.current.memoryUsage = usedMemory;
      
      // Alert if memory usage is high
      if (usedMemory > 100) { // More than 100MB
        if (process.env.NODE_ENV === 'development') {
          console.warn(`[PERF] ${componentName} - High memory usage: ${usedMemory.toFixed(2)}MB`);
        }
      }
    }
  }, [componentName]);
  
  // Start monitoring
  useEffect(() => {
    rafIdRef.current = requestAnimationFrame(measureFPS);
    
    // Measure memory every 5 seconds
    const memoryInterval = setInterval(measureMemory, 5000);
    
    return () => {
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
      }
      clearInterval(memoryInterval);
    };
  }, [measureFPS, measureMemory]);
  
  // Performance mark for specific operations
  const mark = useCallback((markName: string) => {
    performance.mark(`${componentName}-${markName}`);
  }, [componentName]);
  
  // Performance measure between marks
  const measure = useCallback((measureName: string, startMark: string, endMark?: string) => {
    const start = `${componentName}-${startMark}`;
    const end = endMark ? `${componentName}-${endMark}` : undefined;
    
    try {
      performance.measure(`${componentName}-${measureName}`, start, end);
      const measures = performance.getEntriesByName(`${componentName}-${measureName}`);
      const lastMeasure = measures[measures.length - 1];
      
      if (lastMeasure && process.env.NODE_ENV === 'development') {
        console.log(`[PERF] ${componentName} - ${measureName}: ${lastMeasure.duration.toFixed(2)}ms`);
      }
      
      return lastMeasure?.duration || 0;
    } catch (err) {
      console.error(`Failed to measure performance: ${err}`);
      return 0;
    }
  }, [componentName]);
  
  return {
    measureRender,
    measureUpdate,
    mark,
    measure,
    metrics: metricsRef.current,
  };
}

// Hook for tracking React component updates
export function useWhyDidYouUpdate(name: string, props: Record<string, any>) {
  const previousProps = useRef<Record<string, any>>();
  
  useEffect(() => {
    if (previousProps.current && process.env.NODE_ENV === 'development') {
      const allKeys = Object.keys({ ...previousProps.current, ...props });
      const changedProps: Record<string, any> = {};
      
      allKeys.forEach(key => {
        if (previousProps.current![key] !== props[key]) {
          changedProps[key] = {
            from: previousProps.current![key],
            to: props[key],
          };
        }
      });
      
      if (Object.keys(changedProps).length > 0) {
        console.log('[WHY-UPDATE]', name, changedProps);
      }
    }
    
    previousProps.current = props;
  });
}

// Hook for debouncing expensive operations
export function useDebouncedCallback<T extends (...args: any[]) => any>(
  callback: T,
  delay: number,
  maxWait?: number
): T {
  const timeoutRef = useRef<NodeJS.Timeout>();
  const maxTimeoutRef = useRef<NodeJS.Timeout>();
  const lastCallTimeRef = useRef<number>(0);
  
  const debouncedCallback = useCallback((...args: Parameters<T>) => {
    const now = Date.now();
    
    // Clear existing timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    
    // Set up max wait timeout if specified and not already running
    if (maxWait && !maxTimeoutRef.current) {
      lastCallTimeRef.current = now;
      maxTimeoutRef.current = setTimeout(() => {
        callback(...args);
        maxTimeoutRef.current = undefined;
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
          timeoutRef.current = undefined;
        }
      }, maxWait);
    }
    
    // Set up regular debounce timeout
    timeoutRef.current = setTimeout(() => {
      callback(...args);
      timeoutRef.current = undefined;
      if (maxTimeoutRef.current) {
        clearTimeout(maxTimeoutRef.current);
        maxTimeoutRef.current = undefined;
      }
    }, delay);
  }, [callback, delay, maxWait]) as T;
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      if (maxTimeoutRef.current) {
        clearTimeout(maxTimeoutRef.current);
      }
    };
  }, []);
  
  return debouncedCallback;
}