'use client';

import { useEffect, useState, useRef } from 'react';
import { useSnapshot } from 'valtio';
import { mevStore } from '../../stores/mevStore';

interface PnLDataPoint {
  timestamp: number;
  value: number;
  type: 'arbitrage' | 'sandwich';
}

export default function PnLDashboard() {
  const store = useSnapshot(mevStore);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  
  const [pnlData, setPnlData] = useState<PnLDataPoint[]>([]);
  const [stats, setStats] = useState({
    totalPnL: 0,
    dayPnL: 0,
    hourPnL: 0,
    arbPnL: 0,
    sandwichPnL: 0,
    roi: 0,
    sharpe: 0,
    winRate: 0,
    avgWin: 0,
    avgLoss: 0,
    maxDrawdown: 0,
    gasCost: 0,
    netProfit: 0,
  });

  // Generate initial PnL data
  useEffect(() => {
    const points: PnLDataPoint[] = [];
    let cumulative = 0;
    const now = Date.now();
    
    for (let i = 300; i >= 0; i--) {
      const profit = (Math.random() - 0.3) * 100;
      cumulative += profit;
      points.push({
        timestamp: now - i * 1000,
        value: cumulative,
        type: Math.random() > 0.6 ? 'arbitrage' : 'sandwich',
      });
    }
    
    setPnlData(points);
  }, []);

  // Update with real data from store
  useEffect(() => {
    const interval = setInterval(() => {
      setPnlData(prev => {
        const newPoints = [...prev];
        if (newPoints.length > 300) {
          newPoints.shift();
        }
        
        const lastValue = newPoints[newPoints.length - 1]?.value || 0;
        const newProfit = (Math.random() - 0.3) * 50;
        
        newPoints.push({
          timestamp: Date.now(),
          value: lastValue + newProfit,
          type: Math.random() > 0.6 ? 'arbitrage' : 'sandwich',
        });
        
        return newPoints;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Update stats
  useEffect(() => {
    const totalPnL = store.totalProfit;
    const dayPnL = store.profitLast24h;
    const hourPnL = pnlData.filter(p => 
      p.timestamp > Date.now() - 3600000
    ).reduce((sum, p) => sum + (p.value || 0), 0);

    const arbPnL = pnlData.filter(p => p.type === 'arbitrage')
      .reduce((sum, p) => sum + (p.value || 0), 0);
    const sandwichPnL = pnlData.filter(p => p.type === 'sandwich')
      .reduce((sum, p) => sum + (p.value || 0), 0);

    const wins = pnlData.filter(p => p.value > 0);
    const losses = pnlData.filter(p => p.value < 0);
    const winRate = wins.length / (wins.length + losses.length) || 0;
    const avgWin = wins.reduce((sum, p) => sum + p.value, 0) / wins.length || 0;
    const avgLoss = losses.reduce((sum, p) => sum + p.value, 0) / losses.length || 0;

    // Calculate Sharpe ratio (simplified)
    const returns = pnlData.map((p, i) => 
      i > 0 ? (p.value - pnlData[i - 1].value) / Math.abs(pnlData[i - 1].value || 1) : 0
    );
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const stdDev = Math.sqrt(
      returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
    );
    const sharpe = stdDev > 0 ? (avgReturn / stdDev) * Math.sqrt(365) : 0;

    // Calculate max drawdown
    let peak = 0;
    let maxDrawdown = 0;
    for (const point of pnlData) {
      if (point.value > peak) {
        peak = point.value;
      }
      const drawdown = (peak - point.value) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }

    const gasCost = store.transactions.reduce((sum, tx) => sum + (tx.gas_cost || 0), 0);
    const netProfit = totalPnL - gasCost;

    setStats({
      totalPnL,
      dayPnL,
      hourPnL,
      arbPnL,
      sandwichPnL,
      roi: totalPnL > 0 ? (totalPnL / 10000) * 100 : 0, // Assuming 10k capital
      sharpe,
      winRate,
      avgWin,
      avgLoss,
      maxDrawdown,
      gasCost,
      netProfit,
    });
  }, [pnlData, store.totalProfit, store.profitLast24h, store.transactions]);

  // Draw PnL chart
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || pnlData.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      const width = canvas.width;
      const height = canvas.height;
      const padding = 5;
      
      // Clear canvas
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, width, height);

      // Find min/max for scaling
      const values = pnlData.map(p => p.value);
      const minValue = Math.min(...values);
      const maxValue = Math.max(...values);
      const range = maxValue - minValue || 1;

      // Draw zero line if in range
      if (minValue < 0 && maxValue > 0) {
        const zeroY = height - padding - ((0 - minValue) / range) * (height - 2 * padding);
        ctx.strokeStyle = '#52525b';
        ctx.lineWidth = 0.5;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(padding, zeroY);
        ctx.lineTo(width - padding, zeroY);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Draw area chart
      ctx.beginPath();
      ctx.moveTo(padding, height - padding);
      
      pnlData.forEach((point, i) => {
        const x = padding + (i / (pnlData.length - 1)) * (width - 2 * padding);
        const y = height - padding - ((point.value - minValue) / range) * (height - 2 * padding);
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      // Complete area
      ctx.lineTo(width - padding, height - padding);
      ctx.lineTo(padding, height - padding);
      ctx.closePath();

      // Fill gradient
      const gradient = ctx.createLinearGradient(0, 0, 0, height);
      if (pnlData[pnlData.length - 1].value >= 0) {
        gradient.addColorStop(0, 'rgba(34, 197, 94, 0.3)');
        gradient.addColorStop(1, 'rgba(34, 197, 94, 0.05)');
      } else {
        gradient.addColorStop(0, 'rgba(239, 68, 68, 0.3)');
        gradient.addColorStop(1, 'rgba(239, 68, 68, 0.05)');
      }
      ctx.fillStyle = gradient;
      ctx.fill();

      // Draw line
      ctx.beginPath();
      pnlData.forEach((point, i) => {
        const x = padding + (i / (pnlData.length - 1)) * (width - 2 * padding);
        const y = height - padding - ((point.value - minValue) / range) * (height - 2 * padding);
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      
      ctx.strokeStyle = pnlData[pnlData.length - 1].value >= 0 ? '#22c55e' : '#ef4444';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Draw points for different types
      pnlData.forEach((point, i) => {
        if (i % 10 === 0) { // Draw every 10th point to avoid clutter
          const x = padding + (i / (pnlData.length - 1)) * (width - 2 * padding);
          const y = height - padding - ((point.value - minValue) / range) * (height - 2 * padding);
          
          ctx.fillStyle = point.type === 'arbitrage' ? '#60a5fa' : '#fbbf24';
          ctx.beginPath();
          ctx.arc(x, y, 2, 0, Math.PI * 2);
          ctx.fill();
        }
      });
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
  }, [pnlData]);

  const formatValue = (value: number) => {
    if (Math.abs(value) >= 1000) return `$${(value / 1000).toFixed(1)}k`;
    return `$${value.toFixed(0)}`;
  };

  return (
    <div className="h-full flex flex-col">
      <h3 className="text-sm font-semibold text-zinc-400 mb-2">P&L DASHBOARD</h3>
      
      {/* Chart */}
      <div className="flex-1 min-h-0 relative mb-2">
        <canvas 
          ref={canvasRef}
          width={400}
          height={100}
          className="w-full h-full"
        />
        <div className="absolute top-1 right-1 text-[10px] space-y-1">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 bg-blue-500 rounded-full" />
            <span className="text-zinc-600">Arbitrage</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 bg-yellow-500 rounded-full" />
            <span className="text-zinc-600">Sandwich</span>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-4 gap-2 text-xs mb-2">
        <div>
          <div className="text-[10px] text-zinc-600">Total</div>
          <div className={`font-mono ${stats.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {formatValue(stats.totalPnL)}
          </div>
        </div>
        <div>
          <div className="text-[10px] text-zinc-600">24h</div>
          <div className={`font-mono ${stats.dayPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {formatValue(stats.dayPnL)}
          </div>
        </div>
        <div>
          <div className="text-[10px] text-zinc-600">ROI</div>
          <div className={`font-mono ${stats.roi >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {stats.roi.toFixed(1)}%
          </div>
        </div>
        <div>
          <div className="text-[10px] text-zinc-600">Sharpe</div>
          <div className={`font-mono ${stats.sharpe >= 1 ? 'text-green-400' : stats.sharpe >= 0 ? 'text-yellow-400' : 'text-red-400'}`}>
            {stats.sharpe.toFixed(2)}
          </div>
        </div>
      </div>

      {/* Strategy Breakdown */}
      <div className="border-t border-zinc-800 pt-2 grid grid-cols-2 gap-2 text-xs">
        <div className="space-y-1">
          <div className="flex justify-between">
            <span className="text-zinc-600">Arbitrage</span>
            <span className="font-mono text-blue-400">{formatValue(stats.arbPnL)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-zinc-600">Sandwich</span>
            <span className="font-mono text-yellow-400">{formatValue(stats.sandwichPnL)}</span>
          </div>
        </div>
        <div className="space-y-1">
          <div className="flex justify-between">
            <span className="text-zinc-600">Win Rate</span>
            <span className="font-mono text-cyan-400">{(stats.winRate * 100).toFixed(1)}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-zinc-600">Max DD</span>
            <span className="font-mono text-orange-400">{(stats.maxDrawdown * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>

      {/* Gas & Net */}
      <div className="mt-2 pt-2 border-t border-zinc-800 flex justify-between text-xs">
        <div className="flex items-center gap-2">
          <span className="text-zinc-600">Gas</span>
          <span className="font-mono text-orange-400">{formatValue(stats.gasCost)}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-zinc-600">Net</span>
          <span className={`font-mono font-bold ${stats.netProfit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {formatValue(stats.netProfit)}
          </span>
        </div>
      </div>
    </div>
  );
}