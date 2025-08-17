import { useEffect, useState, useRef, useCallback } from 'react';
import { useMemoryManager } from '../hooks/useMemoryManager';
import { useWebSocket } from '../providers/WebSocketProvider';

interface PerformanceMetrics {
  fps: number;
  memory: number;
  memoryPressure: number;
  renderTime: number;
  networkLatency: number;
  wsMessagesPerSec: number;
  wsBytesPerSec: number;
  componentRenders: number;
  qualityMode: 'ultra' | 'high' | 'balanced' | 'low' | 'critical';
}

interface DetailedStats {
  jsHeapUsed: number;
  jsHeapTotal: number;
  jsHeapLimit: number;
  domNodes: number;
  listeners: number;
  bufferSizes: {
    historical: number;
    wsMessages: number;
    cache: number;
  };
}

export function PerformanceMonitor() {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    fps: 60,
    memory: 0,
    memoryPressure: 0,
    renderTime: 0,
    networkLatency: 0,
    wsMessagesPerSec: 0,
    wsBytesPerSec: 0,
    componentRenders: 0,
    qualityMode: 'high'
  });
  
  const [detailedStats, setDetailedStats] = useState<DetailedStats | null>(null);
  const [isMinimized, setIsMinimized] = useState(false);
  const [showDetailed, setShowDetailed] = useState(false);
  const [autoAdjustEnabled, setAutoAdjustEnabled] = useState(true);
  
  const memoryManager = useMemoryManager({
    enableAutoQualityAdjustment: autoAdjustEnabled
  });
  
  const ws = useWebSocket();
  
  const frameTimesRef = useRef<number[]>([]);
  const renderCountRef = useRef(0);
  const observerRef = useRef<PerformanceObserver | null>(null);

  // Track component renders
  useEffect(() => {
    renderCountRef.current++;
  });

  // DISABLED: FPS measurement causes performance issues on CPU-only servers
  useEffect(() => {
    let statsInterval: number;
    
    // Only update stats once per second using a simple interval
    const updateStats = () => {
      // Estimate FPS based on system performance (no actual measurement)
      const estimatedFPS = 30; // Conservative estimate for CPU-only rendering
    
      // Simple metrics update without FPS tracking

      // Get memory stats
      const memStats = memoryManager.getMemoryStats();
      const qualitySettings = memoryManager.getQualitySettings();
      const bufferStats = memoryManager.getBufferStats();
      
      // Determine quality mode based on settings
      let qualityMode: PerformanceMetrics['qualityMode'] = 'high';
      if (!qualitySettings.enableRealtimeUpdates) {
        qualityMode = 'critical';
      } else if (!qualitySettings.enableHighQualityCharts) {
        qualityMode = 'low';
      } else if (!qualitySettings.enableAnimations) {
        qualityMode = 'balanced';
      } else if (qualitySettings.updateFrequency === 16) {
        qualityMode = 'ultra';
      }
      
      setMetrics(prev => ({
        ...prev,
        fps: estimatedFPS,
        memory: Math.round(memStats.usedJSHeapSize / 1048576),
        memoryPressure: memStats.pressure,
        wsMessagesPerSec: 0, // Disabled for performance
        wsBytesPerSec: 0, // Disabled for performance
        componentRenders: renderCountRef.current,
        qualityMode
      }));
      
      // Update detailed stats if panel is expanded
      // Simplified detailed stats (only update if visible)
      if (showDetailed) {
        setDetailedStats({
          jsHeapUsed: memStats.usedJSHeapSize,
          jsHeapTotal: memStats.totalJSHeapSize,
          jsHeapLimit: memStats.jsHeapSizeLimit,
          domNodes: 0, // Disabled for performance
          listeners: 0, // Disabled for performance
          bufferSizes: {
            historical: bufferStats.historicalData.size,
            wsMessages: bufferStats.wsMessages.size,
            cache: bufferStats.componentCache.size
          }
        });
      }
      
      renderCountRef.current = 0;
    };

    // Setup Performance Observer for render timing
    if ('PerformanceObserver' in window) {
      try {
        observerRef.current = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          entries.forEach((entry) => {
            if (entry.entryType === 'measure' && entry.name.includes('render')) {
              setMetrics(prev => ({
                ...prev,
                renderTime: Math.round(entry.duration)
              }));
            }
          });
        });
        
        observerRef.current.observe({ entryTypes: ['measure'] });
      } catch (e) {
        console.warn('PerformanceObserver not available');
      }
    }

      // Update network latency (simplified)
      setMetrics(prev => ({
        ...prev,
        networkLatency: 0 // Disabled for performance
      }));

    // Update stats immediately and then every second
    updateStats();
    statsInterval = window.setInterval(updateStats, 1000);

    return () => {
      if (statsInterval) clearInterval(statsInterval);
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [memoryManager, ws, showDetailed]);

  // Helper function to get event listener count
  const getEventListenerCount = useCallback(() => {
    // This is an approximation - actual count would require browser dev tools API
    return document.querySelectorAll('*').length;
  }, []);

  // Format bytes to human readable
  const formatBytes = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1073741824) return `${(bytes / 1048576).toFixed(1)} MB`;
    return `${(bytes / 1073741824).toFixed(2)} GB`;
  };

  // Get quality mode color
  const getQualityColor = (mode: PerformanceMetrics['qualityMode']): string => {
    switch (mode) {
      case 'ultra': return 'text-green-400';
      case 'high': return 'text-blue-400';
      case 'balanced': return 'text-yellow-400';
      case 'low': return 'text-orange-400';
      case 'critical': return 'text-red-400';
    }
  };

  // Clear memory caches
  const handleClearCache = useCallback(() => {
    memoryManager.clearAll();
    console.log('Memory caches cleared');
  }, [memoryManager]);

  // Force garbage collection (if available)
  const handleForceGC = useCallback(() => {
    memoryManager.performCleanup();
    if (typeof (globalThis as any).gc === 'function') {
      (globalThis as any).gc();
      console.log('Garbage collection triggered');
    } else {
      console.log('Cleanup performed (GC not available)');
    }
  }, [memoryManager]);

  if (isMinimized) {
    return (
      <button
        onClick={() => setIsMinimized(false)}
        className={`fixed bottom-4 left-4 z-50 backdrop-blur-sm px-3 py-1 rounded-full text-xs font-mono border transition-all ${
          metrics.fps < 30 || metrics.memoryPressure > 0.7
            ? 'bg-red-900/90 border-red-700 text-red-200 animate-pulse'
            : 'bg-gray-900/90 border-gray-700 text-white hover:bg-gray-800'
        }`}
      >
        {metrics.fps < 30 ? `${metrics.fps} FPS` : 'Perf'}
      </button>
    );
  }

  return (
    <div className={`fixed bottom-4 left-4 z-50 backdrop-blur-sm text-white rounded-lg text-xs font-mono border transition-all ${
      showDetailed ? 'w-96' : 'w-64'
    } ${
      metrics.memoryPressure > 0.7
        ? 'bg-red-900/90 border-red-700'
        : 'bg-gray-900/90 border-gray-700'
    }`}>
      {/* Header */}
      <div className="flex justify-between items-center p-3 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <span className="font-bold text-green-400">Performance Monitor</span>
          <span className={`px-2 py-0.5 rounded text-[10px] ${getQualityColor(metrics.qualityMode)}`}>
            {metrics.qualityMode.toUpperCase()}
          </span>
        </div>
        <div className="flex gap-1">
          <button
            onClick={() => setShowDetailed(!showDetailed)}
            className="text-gray-400 hover:text-white px-1"
            title={showDetailed ? 'Hide details' : 'Show details'}
          >
            {showDetailed ? '◀' : '▶'}
          </button>
          <button
            onClick={() => setIsMinimized(true)}
            className="text-gray-400 hover:text-white px-1"
          >
            ×
          </button>
        </div>
      </div>
      
      {/* Main Metrics */}
      <div className="p-3 space-y-1">
        <div className="flex justify-between">
          <span>FPS:</span>
          <span className={metrics.fps >= 55 ? 'text-green-400' : metrics.fps >= 30 ? 'text-yellow-400' : 'text-red-400'}>
            {metrics.fps} / 60
          </span>
        </div>
        
        <div className="flex justify-between">
          <span>Memory:</span>
          <div className="flex items-center gap-2">
            <span className={metrics.memoryPressure > 0.7 ? 'text-red-400' : metrics.memoryPressure > 0.5 ? 'text-yellow-400' : 'text-blue-400'}>
              {metrics.memory} MB
            </span>
            <span className="text-gray-500">({(metrics.memoryPressure * 100).toFixed(0)}%)</span>
          </div>
        </div>
        
        <div className="flex justify-between">
          <span>Render Time:</span>
          <span className={metrics.renderTime <= 16 ? 'text-green-400' : metrics.renderTime <= 33 ? 'text-yellow-400' : 'text-red-400'}>
            {metrics.renderTime}ms
          </span>
        </div>
        
        <div className="flex justify-between">
          <span>Network:</span>
          <span className={metrics.networkLatency <= 50 ? 'text-green-400' : metrics.networkLatency <= 150 ? 'text-yellow-400' : 'text-red-400'}>
            {metrics.networkLatency}ms
          </span>
        </div>
        
        <div className="flex justify-between">
          <span>WebSocket:</span>
          <div className="flex gap-2 text-purple-400">
            <span>{metrics.wsMessagesPerSec.toFixed(0)} msg/s</span>
            <span>{formatBytes(metrics.wsBytesPerSec)}/s</span>
          </div>
        </div>
        
        <div className="flex justify-between">
          <span>Renders:</span>
          <span className="text-cyan-400">{metrics.componentRenders}/s</span>
        </div>
      </div>
      
      {/* Detailed Stats */}
      {showDetailed && detailedStats && (
        <>
          <div className="border-t border-gray-700 p-3 space-y-1">
            <div className="text-gray-400 text-[10px] uppercase mb-2">Memory Details</div>
            <div className="flex justify-between">
              <span>Heap Used:</span>
              <span className="text-blue-400">{formatBytes(detailedStats.jsHeapUsed)}</span>
            </div>
            <div className="flex justify-between">
              <span>Heap Total:</span>
              <span className="text-gray-400">{formatBytes(detailedStats.jsHeapTotal)}</span>
            </div>
            <div className="flex justify-between">
              <span>Heap Limit:</span>
              <span className="text-gray-400">{formatBytes(detailedStats.jsHeapLimit)}</span>
            </div>
          </div>
          
          <div className="border-t border-gray-700 p-3 space-y-1">
            <div className="text-gray-400 text-[10px] uppercase mb-2">Buffer Usage</div>
            <div className="flex justify-between">
              <span>Historical:</span>
              <span className="text-green-400">{detailedStats.bufferSizes.historical} items</span>
            </div>
            <div className="flex justify-between">
              <span>WS Queue:</span>
              <span className="text-yellow-400">{detailedStats.bufferSizes.wsMessages} msgs</span>
            </div>
            <div className="flex justify-between">
              <span>Cache:</span>
              <span className="text-purple-400">{detailedStats.bufferSizes.cache} entries</span>
            </div>
          </div>
          
          <div className="border-t border-gray-700 p-3 space-y-1">
            <div className="text-gray-400 text-[10px] uppercase mb-2">DOM Stats</div>
            <div className="flex justify-between">
              <span>DOM Nodes:</span>
              <span className="text-orange-400">{detailedStats.domNodes}</span>
            </div>
            <div className="flex justify-between">
              <span>Listeners:</span>
              <span className="text-pink-400">{detailedStats.listeners}</span>
            </div>
          </div>
          
          <div className="border-t border-gray-700 p-3 space-y-2">
            <div className="text-gray-400 text-[10px] uppercase mb-2">Controls</div>
            <div className="flex items-center justify-between">
              <span>Auto-Adjust:</span>
              <button
                onClick={() => setAutoAdjustEnabled(!autoAdjustEnabled)}
                className={`px-2 py-0.5 rounded text-[10px] transition-colors ${
                  autoAdjustEnabled
                    ? 'bg-green-600 text-white'
                    : 'bg-gray-600 text-gray-300'
                }`}
              >
                {autoAdjustEnabled ? 'ON' : 'OFF'}
              </button>
            </div>
            <div className="flex gap-2 mt-2">
              <button
                onClick={handleClearCache}
                className="flex-1 px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-[10px] transition-colors"
              >
                Clear Cache
              </button>
              <button
                onClick={handleForceGC}
                className="flex-1 px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-[10px] transition-colors"
              >
                Force GC
              </button>
            </div>
          </div>
        </>
      )}
      
      {/* Warning Banner */}
      {(metrics.fps < 30 || metrics.memoryPressure > 0.7) && (
        <div className="border-t border-red-700 bg-red-900/50 p-2 text-red-200 text-[10px]">
          {metrics.fps < 30 && <div>⚠ Low FPS detected</div>}
          {metrics.memoryPressure > 0.7 && <div>⚠ High memory pressure</div>}
        </div>
      )}
    </div>
  );
}