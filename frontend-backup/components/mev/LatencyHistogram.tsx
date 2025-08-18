'use client';

import { memo, useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';

interface LatencyData {
  timestamp: number;
  p50: number;
  p95: number;
  p99: number;
  max: number;
  count: number;
}

export const LatencyHistogram = memo(() => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const [data, setData] = useState<LatencyData[]>([]);
  const [stats, setStats] = useState({
    currentP50: 0,
    currentP95: 0,
    currentP99: 0,
    avgLatency: 0,
    totalSamples: 0
  });
  
  // Generate streaming data
  useEffect(() => {
    const generateData = () => {
      const baseLatency = 5 + Math.random() * 3;
      return {
        timestamp: Date.now(),
        p50: baseLatency + Math.random() * 2,
        p95: baseLatency + 5 + Math.random() * 5,
        p99: baseLatency + 10 + Math.random() * 10,
        max: baseLatency + 15 + Math.random() * 20,
        count: Math.floor(100 + Math.random() * 200)
      };
    };
    
    // Initial data
    const initialData = Array.from({ length: 100 }, () => generateData());
    setData(initialData);
    
    // Stream new data
    const interval = setInterval(() => {
      setData(prev => {
        const newData = [...prev, generateData()];
        return newData.slice(-100); // Keep last 100 points
      });
    }, 100);
    
    return () => clearInterval(interval);
  }, []);
  
  // Update stats
  useEffect(() => {
    if (data.length === 0) return;
    
    const latest = data[data.length - 1];
    const avgLatency = data.reduce((sum, d) => sum + d.p50, 0) / data.length;
    const totalSamples = data.reduce((sum, d) => sum + d.count, 0);
    
    setStats({
      currentP50: latest.p50,
      currentP95: latest.p95,
      currentP99: latest.p99,
      avgLatency,
      totalSamples
    });
  }, [data]);
  
  // Canvas rendering with 60fps
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d', { 
      alpha: false,
      desynchronized: true 
    });
    if (!ctx) return;
    
    // Set canvas size
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    
    const width = rect.width;
    const height = rect.height;
    
    const render = () => {
      // Clear canvas
      ctx.fillStyle = '#030712';
      ctx.fillRect(0, 0, width, height);
      
      if (data.length === 0) {
        animationRef.current = requestAnimationFrame(render);
        return;
      }
      
      const padding = 40;
      const chartWidth = width - padding * 2;
      const chartHeight = height - padding * 2;
      
      // Draw grid
      ctx.strokeStyle = '#1f2937';
      ctx.lineWidth = 0.5;
      
      // Horizontal grid lines
      for (let i = 0; i <= 5; i++) {
        const y = padding + (chartHeight / 5) * i;
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
      }
      
      // Draw axes
      ctx.strokeStyle = '#4b5563';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding, padding);
      ctx.lineTo(padding, height - padding);
      ctx.lineTo(width - padding, height - padding);
      ctx.stroke();
      
      // Draw histogram bars
      const barWidth = chartWidth / data.length;
      const maxLatency = 40;
      
      data.forEach((d, i) => {
        const x = padding + i * barWidth;
        
        // P99 bar (red)
        const p99Height = (d.p99 / maxLatency) * chartHeight;
        ctx.fillStyle = 'rgba(239, 68, 68, 0.3)';
        ctx.fillRect(
          x,
          height - padding - p99Height,
          barWidth - 1,
          p99Height
        );
        
        // P95 bar (yellow)
        const p95Height = (d.p95 / maxLatency) * chartHeight;
        ctx.fillStyle = 'rgba(245, 158, 11, 0.5)';
        ctx.fillRect(
          x,
          height - padding - p95Height,
          barWidth - 1,
          p95Height
        );
        
        // P50 bar (green)
        const p50Height = (d.p50 / maxLatency) * chartHeight;
        ctx.fillStyle = 'rgba(16, 185, 129, 0.7)';
        ctx.fillRect(
          x,
          height - padding - p50Height,
          barWidth - 1,
          p50Height
        );
      });
      
      // Draw target lines
      const targets = [
        { value: 8, label: 'P50 Target', color: '#10b981' },
        { value: 20, label: 'P99 Target', color: '#ef4444' }
      ];
      
      targets.forEach(target => {
        const y = height - padding - (target.value / maxLatency) * chartHeight;
        
        ctx.strokeStyle = target.color;
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Label
        ctx.fillStyle = target.color;
        ctx.font = '10px monospace';
        ctx.fillText(target.label, width - padding - 60, y - 5);
      });
      
      // Draw labels
      ctx.fillStyle = '#9ca3af';
      ctx.font = '11px monospace';
      
      // Y-axis labels
      for (let i = 0; i <= 5; i++) {
        const value = (maxLatency / 5) * (5 - i);
        const y = padding + (chartHeight / 5) * i;
        ctx.fillText(`${value}ms`, 5, y + 3);
      }
      
      // X-axis label
      ctx.fillText('Time (last 100 samples)', width / 2 - 50, height - 10);
      
      animationRef.current = requestAnimationFrame(render);
    };
    
    render();
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [data]);
  
  const getLatencyColor = (value: number, target: number) => {
    if (value <= target) return 'text-green-400';
    if (value <= target * 1.5) return 'text-yellow-400';
    return 'text-red-400';
  };
  
  return (
    <div className="space-y-4">
      {/* Stats Cards */}
      <div className="grid grid-cols-4 gap-3">
        <motion.div
          className="bg-gray-900/50 rounded-lg p-3 border border-gray-800"
          whileHover={{ scale: 1.02 }}
        >
          <div className="text-xs text-gray-500 mb-1">P50 Latency</div>
          <div className={`text-2xl font-bold ${getLatencyColor(stats.currentP50, 8)}`}>
            {stats.currentP50.toFixed(1)}ms
          </div>
          <div className="text-xs text-gray-600">Target: ≤8ms</div>
        </motion.div>
        
        <motion.div
          className="bg-gray-900/50 rounded-lg p-3 border border-gray-800"
          whileHover={{ scale: 1.02 }}
        >
          <div className="text-xs text-gray-500 mb-1">P95 Latency</div>
          <div className={`text-2xl font-bold ${getLatencyColor(stats.currentP95, 15)}`}>
            {stats.currentP95.toFixed(1)}ms
          </div>
          <div className="text-xs text-gray-600">Target: ≤15ms</div>
        </motion.div>
        
        <motion.div
          className="bg-gray-900/50 rounded-lg p-3 border border-gray-800"
          whileHover={{ scale: 1.02 }}
        >
          <div className="text-xs text-gray-500 mb-1">P99 Latency</div>
          <div className={`text-2xl font-bold ${getLatencyColor(stats.currentP99, 20)}`}>
            {stats.currentP99.toFixed(1)}ms
          </div>
          <div className="text-xs text-gray-600">Target: ≤20ms</div>
        </motion.div>
        
        <motion.div
          className="bg-gray-900/50 rounded-lg p-3 border border-gray-800"
          whileHover={{ scale: 1.02 }}
        >
          <div className="text-xs text-gray-500 mb-1">Samples/sec</div>
          <div className="text-2xl font-bold text-blue-400">
            {Math.floor(stats.totalSamples / 100)}
          </div>
          <div className="text-xs text-gray-600">Total: {stats.totalSamples}</div>
        </motion.div>
      </div>
      
      {/* Canvas Histogram */}
      <div className="bg-gray-950 rounded-lg p-4 border border-gray-800">
        <canvas
          ref={canvasRef}
          className="w-full h-64"
          style={{
            imageRendering: 'pixelated',
            transform: 'translateZ(0)',
            willChange: 'transform'
          }}
        />
      </div>
      
      {/* Legend */}
      <div className="flex justify-center gap-6 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-green-500/70 rounded" />
          <span className="text-gray-400">P50</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-yellow-500/50 rounded" />
          <span className="text-gray-400">P95</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-red-500/30 rounded" />
          <span className="text-gray-400">P99</span>
        </div>
      </div>
      
      {/* Performance Indicator */}
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${
            stats.currentP50 <= 8 && stats.currentP99 <= 20
              ? 'bg-green-500 animate-pulse'
              : 'bg-red-500'
          }`} />
          <span className="text-gray-500">
            System Status: {
              stats.currentP50 <= 8 && stats.currentP99 <= 20
                ? 'OPTIMAL'
                : 'DEGRADED'
            }
          </span>
        </div>
        
        <span className="text-gray-600">
          Rendering at 60 FPS • Canvas Accelerated
        </span>
      </div>
    </div>
  );
});

LatencyHistogram.displayName = 'LatencyHistogram';