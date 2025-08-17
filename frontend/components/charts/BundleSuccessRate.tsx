'use client';

import { useEffect, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi, HistogramData } from 'lightweight-charts';
import { useSnapshot } from 'valtio';
import { mevStore } from '../../stores/mevStore';
import { clsx } from 'clsx';

export default function BundleSuccessRate() {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
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
      timeScale: {
        borderColor: '#27272a',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: '#27272a',
        scaleMargins: {
          top: 0.1,
          bottom: 0.1,
        },
      },
    });
    
    // Create histogram series
    const series = chart.addHistogramSeries({
      color: '#3b82f6',
      priceFormat: {
        type: 'percent',
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
    
    const data: HistogramData[] = snap.bundleStats.map(stat => ({
      time: Math.floor(new Date(stat.timestamp).getTime() / 1000) as any,
      value: stat.land_rate,
      color: stat.land_rate >= 65 ? '#22c55e' : stat.land_rate >= 55 ? '#eab308' : '#ef4444',
    })).reverse();
    
    seriesRef.current.setData(data);
  }, [snap.bundleStats]);
  
  const currentRate = snap.currentLandRate;
  const rateStatus = currentRate >= 65 ? 'excellent' : currentRate >= 55 ? 'good' : 'poor';
  
  return (
    <div className="chart-container">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-zinc-400">Bundle Land Rate</h3>
        <div className="flex items-center gap-2">
          <div className={clsx(
            "text-sm font-medium",
            rateStatus === 'excellent' && "text-green-400",
            rateStatus === 'good' && "text-yellow-400",
            rateStatus === 'poor' && "text-red-400"
          )}>
            {currentRate.toFixed(1)}%
          </div>
          <div className={clsx(
            "w-2 h-2 rounded-full",
            rateStatus === 'excellent' && "bg-green-500",
            rateStatus === 'good' && "bg-yellow-500",
            rateStatus === 'poor' && "bg-red-500"
          )} />
        </div>
      </div>
      <div ref={containerRef} />
      
      {/* Threshold indicators */}
      <div className="flex items-center justify-between mt-2 text-xs">
        <span className="text-zinc-600">Target: ≥65%</span>
        <span className="text-zinc-600">Min: ≥55%</span>
      </div>
    </div>
  );
}