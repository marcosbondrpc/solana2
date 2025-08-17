'use client';

import { useEffect, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi, LineData } from 'lightweight-charts';
import { useSnapshot } from 'valtio';
import { mevStore } from '../../stores/mevStore';

export default function ProfitChart() {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const snap = useSnapshot(mevStore);
  
  useEffect(() => {
    if (!containerRef.current) return;
    
    // Create chart
    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 256,
      layout: {
        background: { color: 'transparent' },
        textColor: '#71717a',
      },
      grid: {
        vertLines: { color: '#27272a' },
        horzLines: { color: '#27272a' },
      },
      crosshair: {
        mode: 1,
        vertLine: {
          color: '#71717a',
          width: 1,
          style: 3,
        },
        horzLine: {
          color: '#71717a',
          width: 1,
          style: 3,
        },
      },
      timeScale: {
        borderColor: '#27272a',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: '#27272a',
      },
    });
    
    // Create series
    const series = chart.addLineSeries({
      color: '#22c55e',
      lineWidth: 2,
      crosshairMarkerVisible: true,
      priceFormat: {
        type: 'price',
        precision: 2,
        minMove: 0.01,
      },
    });
    
    chartRef.current = chart;
    seriesRef.current = series;
    
    // Handle resize
    const handleResize = () => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: containerRef.current.clientWidth,
        });
      }
    };
    
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);
  
  // Update data when bundle stats change
  useEffect(() => {
    if (!seriesRef.current || snap.bundleStats.length === 0) return;
    
    const data: LineData[] = snap.bundleStats.map(stat => ({
      time: Math.floor(new Date(stat.timestamp).getTime() / 1000) as any,
      value: stat.total_profit,
    })).reverse();
    
    seriesRef.current.setData(data);
  }, [snap.bundleStats]);
  
  return (
    <div className="chart-container">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-zinc-400">Cumulative Profit</h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500">1H</span>
          <div className="text-sm font-medium text-green-400">
            +${snap.profitLast24h.toFixed(2)}
          </div>
        </div>
      </div>
      <div ref={containerRef} />
    </div>
  );
}