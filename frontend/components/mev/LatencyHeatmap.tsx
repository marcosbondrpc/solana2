'use client';

import { useEffect, useState, useRef } from 'react';
import { useSnapshot } from 'valtio';
import { mevStore } from '../../stores/mevStore';

interface LatencyComponent {
  name: string;
  p50: number;
  p95: number;
  p99: number;
  current: number;
  trend: 'up' | 'down' | 'stable';
}

export default function LatencyHeatmap() {
  const store = useSnapshot(mevStore);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  
  const [components, setComponents] = useState<LatencyComponent[]>([
    { name: 'Detection', p50: 2.3, p95: 4.8, p99: 7.2, current: 2.5, trend: 'stable' },
    { name: 'Decision', p50: 3.1, p95: 5.2, p99: 8.4, current: 3.3, trend: 'up' },
    { name: 'Execution', p50: 1.8, p95: 3.1, p99: 4.5, current: 1.9, trend: 'down' },
    { name: 'Confirmation', p50: 0.7, p95: 1.2, p99: 2.1, current: 0.8, trend: 'stable' },
  ]);

  const [heatmapData, setHeatmapData] = useState<number[][]>([]);
  const [timeWindow, setTimeWindow] = useState<number[]>([]);

  // Initialize heatmap data
  useEffect(() => {
    const cols = 60; // 60 seconds
    const rows = 4;  // 4 components
    const data = Array(rows).fill(null).map(() => 
      Array(cols).fill(null).map(() => Math.random() * 10 + 2)
    );
    setHeatmapData(data);
    setTimeWindow(Array(cols).fill(null).map((_, i) => Date.now() - (cols - i) * 1000));
  }, []);

  // Update heatmap with new data
  useEffect(() => {
    const interval = setInterval(() => {
      setHeatmapData(prev => {
        const newData = prev.map((row, idx) => {
          const newRow = [...row.slice(1)];
          const component = components[idx];
          // Simulate new latency value
          const jitter = (Math.random() - 0.5) * 2;
          const baseValue = component.current + jitter;
          newRow.push(Math.max(0, baseValue));
          return newRow;
        });
        return newData;
      });

      setTimeWindow(prev => {
        const newWindow = [...prev.slice(1)];
        newWindow.push(Date.now());
        return newWindow;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [components]);

  // Draw heatmap
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || heatmapData.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      const width = canvas.width;
      const height = canvas.height;
      const cellWidth = width / heatmapData[0].length;
      const cellHeight = height / heatmapData.length;

      // Clear canvas
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, width, height);

      // Draw heatmap cells
      heatmapData.forEach((row, rowIdx) => {
        row.forEach((value, colIdx) => {
          const x = colIdx * cellWidth;
          const y = rowIdx * cellHeight;
          
          // Color based on latency
          let color: string;
          if (value < 8) {
            // Green gradient
            const intensity = 1 - (value / 8);
            color = `rgba(34, 197, 94, ${0.3 + intensity * 0.7})`;
          } else if (value < 15) {
            // Yellow gradient
            const intensity = 1 - ((value - 8) / 7);
            color = `rgba(250, 204, 21, ${0.3 + intensity * 0.7})`;
          } else {
            // Red gradient
            const intensity = Math.min(1, (value - 15) / 10);
            color = `rgba(239, 68, 68, ${0.3 + intensity * 0.7})`;
          }
          
          ctx.fillStyle = color;
          ctx.fillRect(x, y, cellWidth - 1, cellHeight - 1);
        });
      });

      // Draw grid lines
      ctx.strokeStyle = '#18181b';
      ctx.lineWidth = 1;
      for (let i = 0; i <= heatmapData.length; i++) {
        const y = i * cellHeight;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
    };

    const animate = () => {
      draw();
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [heatmapData]);

  // Update from store
  useEffect(() => {
    if (store.health.decisionLatencyP50 > 0) {
      setComponents(prev => prev.map(comp => {
        if (comp.name === 'Decision') {
          return {
            ...comp,
            p50: store.health.decisionLatencyP50,
            p99: store.health.decisionLatencyP99,
            current: store.currentLatency,
            trend: store.currentLatency > comp.current ? 'up' : 
                   store.currentLatency < comp.current ? 'down' : 'stable',
          };
        }
        return comp;
      }));
    }
  }, [store.health.decisionLatencyP50, store.health.decisionLatencyP99, store.currentLatency]);

  const getLatencyColor = (value: number) => {
    if (value < 8) return 'text-green-400';
    if (value < 15) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getTrendIcon = (trend: string) => {
    if (trend === 'up') return '↑';
    if (trend === 'down') return '↓';
    return '→';
  };

  return (
    <div className="h-full flex flex-col">
      <h3 className="text-sm font-semibold text-zinc-400 mb-2">LATENCY HEATMAP</h3>
      
      {/* Heatmap Canvas */}
      <div className="flex-1 min-h-0 flex gap-2 mb-2">
        <div className="flex flex-col justify-around text-[10px] text-zinc-600">
          {components.map(comp => (
            <div key={comp.name} className="font-mono">
              {comp.name.slice(0, 3).toUpperCase()}
            </div>
          ))}
        </div>
        <div className="flex-1 relative">
          <canvas 
            ref={canvasRef}
            width={480}
            height={80}
            className="w-full h-full"
          />
          <div className="absolute bottom-0 left-0 right-0 flex justify-between text-[9px] text-zinc-600">
            <span>-60s</span>
            <span>-30s</span>
            <span>now</span>
          </div>
        </div>
      </div>

      {/* Latency Stats Table */}
      <div className="border-t border-zinc-800 pt-2">
        <div className="grid grid-cols-5 gap-2 text-[10px] text-zinc-600 mb-1">
          <div>Component</div>
          <div className="text-right">Current</div>
          <div className="text-right">P50</div>
          <div className="text-right">P95</div>
          <div className="text-right">P99</div>
        </div>
        {components.map(comp => (
          <div key={comp.name} className="grid grid-cols-5 gap-2 text-xs font-mono">
            <div className="text-zinc-400">{comp.name}</div>
            <div className={`text-right ${getLatencyColor(comp.current)}`}>
              {comp.current.toFixed(1)}ms
              <span className="ml-1 text-[10px]">{getTrendIcon(comp.trend)}</span>
            </div>
            <div className={`text-right ${getLatencyColor(comp.p50)}`}>
              {comp.p50.toFixed(1)}
            </div>
            <div className={`text-right ${getLatencyColor(comp.p95)}`}>
              {comp.p95.toFixed(1)}
            </div>
            <div className={`text-right ${getLatencyColor(comp.p99)}`}>
              {comp.p99.toFixed(1)}
            </div>
          </div>
        ))}
      </div>

      {/* Total Latency Indicator */}
      <div className="mt-2 pt-2 border-t border-zinc-800 flex items-center justify-between">
        <span className="text-xs text-zinc-600">Total Latency</span>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1">
            <span className="text-[10px] text-zinc-600">P50:</span>
            <span className={`text-xs font-mono ${getLatencyColor(store.health.decisionLatencyP50)}`}>
              {store.health.decisionLatencyP50.toFixed(1)}ms
            </span>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-[10px] text-zinc-600">P99:</span>
            <span className={`text-xs font-mono ${getLatencyColor(store.health.decisionLatencyP99)}`}>
              {store.health.decisionLatencyP99.toFixed(1)}ms
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}