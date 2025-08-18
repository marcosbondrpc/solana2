'use client';

import { useEffect, useState, useRef } from 'react';
import { useSnapshot } from 'valtio';
import { mevStore } from '../../stores/mevStore';

interface RouteStats {
  name: string;
  alpha: number; // Beta distribution alpha (successes)
  beta: number;  // Beta distribution beta (failures)
  successRate: number;
  totalTries: number;
  avgProfit: number;
  lastSample: number;
}

export default function ExecutionMonitor() {
  const store = useSnapshot(mevStore);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  
  const [routeStats, setRouteStats] = useState<RouteStats[]>([
    { name: 'Direct TPU', alpha: 10, beta: 3, successRate: 0.77, totalTries: 13, avgProfit: 125.4, lastSample: 0.82 },
    { name: 'Jito Main', alpha: 25, beta: 5, successRate: 0.83, totalTries: 30, avgProfit: 210.5, lastSample: 0.85 },
    { name: 'Jito Alt', alpha: 8, beta: 4, successRate: 0.67, totalTries: 12, avgProfit: 95.2, lastSample: 0.71 },
  ]);

  const [tipLadder, setTipLadder] = useState([
    { range: '0-5', count: 45, successRate: 0.42 },
    { range: '5-10', count: 82, successRate: 0.61 },
    { range: '10-25', count: 124, successRate: 0.78 },
    { range: '25-50', count: 67, successRate: 0.85 },
    { range: '50+', count: 23, successRate: 0.91 },
  ]);

  // Beta distribution PDF for visualization
  const betaPDF = (x: number, alpha: number, beta: number): number => {
    if (x <= 0 || x >= 1) return 0;
    const B = (a: number, b: number) => {
      // Approximation of Beta function
      return Math.exp(
        Math.log(Math.pow(x, a - 1)) + 
        Math.log(Math.pow(1 - x, b - 1))
      );
    };
    return B(alpha, beta);
  };

  // Draw Beta distributions on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      const width = canvas.width;
      const height = canvas.height;
      
      // Clear canvas
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, width, height);

      // Draw grid
      ctx.strokeStyle = '#27272a';
      ctx.lineWidth = 0.5;
      for (let i = 0; i <= 10; i++) {
        const x = (width / 10) * i;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let i = 0; i <= 4; i++) {
        const y = (height / 4) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Draw Beta distributions
      const colors = ['#a78bfa', '#06b6d4', '#f472b6'];
      routeStats.forEach((route, idx) => {
        ctx.strokeStyle = colors[idx];
        ctx.lineWidth = 2;
        ctx.beginPath();

        for (let px = 0; px < width; px++) {
          const x = px / width;
          const y = betaPDF(x, route.alpha, route.beta);
          const py = height - (y * height / 5); // Scale to fit
          
          if (px === 0) {
            ctx.moveTo(px, py);
          } else {
            ctx.lineTo(px, py);
          }
        }
        ctx.stroke();

        // Draw sample point
        const sampleX = route.lastSample * width;
        ctx.fillStyle = colors[idx];
        ctx.beginPath();
        ctx.arc(sampleX, height - 10, 3, 0, Math.PI * 2);
        ctx.fill();
      });

      // Draw legend
      ctx.font = '10px monospace';
      routeStats.forEach((route, idx) => {
        ctx.fillStyle = colors[idx];
        ctx.fillText(route.name, 10, 15 + idx * 12);
      });
    };

    draw();
    
    // Animate sampling
    const animate = () => {
      // Update last samples with smooth transitions
      setRouteStats(prev => prev.map(route => ({
        ...route,
        lastSample: route.lastSample + (Math.random() - 0.5) * 0.05,
        lastSample: Math.max(0.1, Math.min(0.9, route.lastSample)),
      })));
      
      draw();
      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [routeStats]);

  // Update stats from store
  useEffect(() => {
    if (store.topRoutes && store.topRoutes.length > 0) {
      const newStats = store.topRoutes.slice(0, 3).map(route => {
        const totalTries = route.count;
        const successes = Math.round(route.successRate * totalTries);
        const failures = totalTries - successes;
        
        return {
          name: route.route,
          alpha: successes + 1, // Add 1 for prior
          beta: failures + 1,
          successRate: route.successRate,
          totalTries,
          avgProfit: route.profit / route.count,
          lastSample: route.successRate + (Math.random() - 0.5) * 0.1,
        };
      });
      
      if (newStats.length > 0) {
        setRouteStats(newStats);
      }
    }
  }, [store.topRoutes]);

  return (
    <div className="h-full flex flex-col">
      <h3 className="text-sm font-semibold text-zinc-400 mb-2">THOMPSON SAMPLING</h3>
      
      {/* Beta Distribution Canvas */}
      <div className="flex-1 min-h-0 relative mb-2">
        <canvas 
          ref={canvasRef} 
          width={400} 
          height={120}
          className="w-full h-full"
        />
      </div>

      {/* Route Performance Stats */}
      <div className="space-y-1 mb-2">
        {routeStats.map((route, idx) => (
          <div key={route.name} className="flex items-center justify-between text-xs">
            <span className="font-mono text-zinc-500">{route.name}</span>
            <div className="flex items-center gap-3">
              <span className="text-green-400">{(route.successRate * 100).toFixed(1)}%</span>
              <span className="text-zinc-600">{route.totalTries} tries</span>
              <span className="text-cyan-400">${route.avgProfit.toFixed(1)}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Adaptive Tip Ladder */}
      <div className="border-t border-zinc-800 pt-2">
        <div className="text-[10px] text-zinc-600 mb-1">TIP LADDER</div>
        <div className="grid grid-cols-5 gap-1">
          {tipLadder.map((tier) => (
            <div 
              key={tier.range}
              className="text-center"
            >
              <div 
                className="h-8 bg-gradient-to-t from-purple-900/20 to-purple-600/20 border border-purple-800/30 rounded-sm relative overflow-hidden"
              >
                <div 
                  className="absolute bottom-0 left-0 right-0 bg-purple-600/40"
                  style={{ height: `${tier.successRate * 100}%` }}
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-[10px] font-mono text-white">
                    {(tier.successRate * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <div className="text-[9px] text-zinc-600 mt-1">{tier.range}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Land Rate Indicator */}
      <div className="mt-2 flex items-center justify-between text-xs">
        <span className="text-zinc-600">Land Rate</span>
        <div className="flex items-center gap-2">
          <div className="w-32 h-2 bg-zinc-900 rounded-full overflow-hidden">
            <div 
              className={`h-full transition-all duration-300 ${
                store.currentLandRate >= 65 ? 'bg-green-500' :
                store.currentLandRate >= 55 ? 'bg-yellow-500' :
                'bg-red-500'
              }`}
              style={{ width: `${Math.min(100, store.currentLandRate)}%` }}
            />
          </div>
          <span className={`font-mono ${
            store.currentLandRate >= 65 ? 'text-green-400' :
            store.currentLandRate >= 55 ? 'text-yellow-400' :
            'text-red-400'
          }`}>
            {store.currentLandRate.toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
}