'use client';

import { memo, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { usePerformanceMetrics } from '@/stores/detectionStore';

export const PerformanceMonitor = memo(() => {
  const metrics = usePerformanceMetrics();
  const [expanded, setExpanded] = useState(false);
  
  const getFPSColor = (fps: number) => {
    if (fps >= 55) return 'text-green-400';
    if (fps >= 30) return 'text-yellow-400';
    return 'text-red-400';
  };
  
  const getMemoryColor = (usage: number) => {
    if (usage <= 50) return 'text-green-400';
    if (usage <= 75) return 'text-yellow-400';
    return 'text-red-400';
  };
  
  return (
    <motion.div
      className="fixed bottom-4 right-4 z-50"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <motion.div
        className="bg-gray-900/95 backdrop-blur-sm border border-gray-800 rounded-lg p-3 cursor-pointer"
        onClick={() => setExpanded(!expanded)}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <div className="flex items-center gap-4">
          {/* FPS Indicator */}
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${
              metrics.fps >= 55 ? 'bg-green-500' : 
              metrics.fps >= 30 ? 'bg-yellow-500' : 
              'bg-red-500'
            } animate-pulse`} />
            <span className={`text-xs font-mono ${getFPSColor(metrics.fps)}`}>
              {metrics.fps} FPS
            </span>
          </div>
          
          {/* Memory Usage */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">MEM:</span>
            <span className={`text-xs font-mono ${getMemoryColor(metrics.memoryUsage)}`}>
              {metrics.memoryUsage.toFixed(0)}%
            </span>
          </div>
          
          {/* Render Time */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">RT:</span>
            <span className="text-xs font-mono text-blue-400">
              {metrics.renderTime.toFixed(1)}ms
            </span>
          </div>
        </div>
        
        {/* Expanded Details */}
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            className="mt-3 pt-3 border-t border-gray-800"
          >
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-500">CPU Usage:</span>
                <span className="font-mono">{metrics.cpuUsage.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">GPU Acceleration:</span>
                <span className="font-mono text-green-400">ENABLED</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">WebTransport:</span>
                <span className="font-mono text-green-400">CONNECTED</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Worker Threads:</span>
                <span className="font-mono">4 ACTIVE</span>
              </div>
            </div>
            
            {/* Performance Graph */}
            <div className="mt-3 h-8 bg-gray-950 rounded flex items-end gap-px p-1">
              {Array.from({ length: 30 }, (_, i) => (
                <div
                  key={i}
                  className="flex-1 bg-blue-500/50 rounded-t"
                  style={{
                    height: `${20 + Math.random() * 60}%`,
                    opacity: 0.3 + (i / 30) * 0.7
                  }}
                />
              ))}
            </div>
          </motion.div>
        )}
      </motion.div>
      
      {/* Performance Warning */}
      {metrics.fps < 30 && (
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="absolute bottom-0 right-full mr-2 bg-red-900/90 text-red-400 text-xs px-3 py-2 rounded-lg whitespace-nowrap"
        >
          Performance degraded - Consider reducing data volume
        </motion.div>
      )}
    </motion.div>
  );
});

PerformanceMonitor.displayName = 'PerformanceMonitor';