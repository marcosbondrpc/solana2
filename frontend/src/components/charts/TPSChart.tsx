/**
 * Real-time TPS chart using lightweight-charts
 * Optimized for 60fps updates with millions of data points
 */

import React, { useEffect, useRef, useMemo } from 'react';
import { createChart, IChartApi, ISeriesApi, LineData, DeepPartial, ChartOptions, LineSeriesOptions } from 'lightweight-charts';

interface TPSChartProps {
  data: Array<{ time: number; value: number }>;
  width?: number;
  height?: number;
  showVolume?: boolean;
  theme?: 'dark' | 'light';
}

export function TPSChart({
  data,
  width = 800,
  height = 400,
  showVolume = false,
  theme = 'dark'
}: TPSChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

  // Chart options based on theme
  const chartOptions: DeepPartial<ChartOptions> = useMemo(() => ({
    width,
    height,
    layout: {
      background: {
        type: 'solid' as const,
        color: theme === 'dark' ? '#111827' : '#ffffff'
      },
      textColor: theme === 'dark' ? '#9ca3af' : '#374151'
    },
    grid: {
      vertLines: {
        color: theme === 'dark' ? '#1f2937' : '#e5e7eb',
        style: 1
      },
      horzLines: {
        color: theme === 'dark' ? '#1f2937' : '#e5e7eb',
        style: 1
      }
    },
    crosshair: {
      mode: 0,
      vertLine: {
        color: theme === 'dark' ? '#4b5563' : '#9ca3af',
        width: 1,
        style: 3,
        labelBackgroundColor: theme === 'dark' ? '#1f2937' : '#f3f4f6'
      },
      horzLine: {
        color: theme === 'dark' ? '#4b5563' : '#9ca3af',
        width: 1,
        style: 3,
        labelBackgroundColor: theme === 'dark' ? '#1f2937' : '#f3f4f6'
      }
    },
    timeScale: {
      borderColor: theme === 'dark' ? '#374151' : '#d1d5db',
      timeVisible: true,
      secondsVisible: true,
      tickMarkFormatter: (time: any) => {
        const date = new Date(time * 1000);
        return date.toLocaleTimeString();
      }
    },
    rightPriceScale: {
      borderColor: theme === 'dark' ? '#374151' : '#d1d5db',
      scaleMargins: {
        top: 0.1,
        bottom: showVolume ? 0.3 : 0.1
      }
    },
    handleScroll: {
      mouseWheel: true,
      pressedMouseMove: true,
      horzTouchDrag: true,
      vertTouchDrag: false
    },
    handleScale: {
      axisPressedMouseMove: {
        time: true,
        price: false
      },
      axisDoubleClickReset: true,
      mouseWheel: true,
      pinch: true
    }
  }), [width, height, theme, showVolume]);

  // Line series options
  const lineOptions: DeepPartial<LineSeriesOptions> = useMemo(() => ({
    color: '#10b981',
    lineWidth: 2,
    lineStyle: 0,
    crosshairMarkerVisible: true,
    crosshairMarkerRadius: 4,
    crosshairMarkerBorderColor: '#10b981',
    crosshairMarkerBackgroundColor: '#10b981',
    priceScaleId: 'right',
    title: 'TPS',
    priceFormat: {
      type: 'custom' as const,
      formatter: (price: number) => {
        if (price >= 1e6) return `${(price / 1e6).toFixed(2)}M`;
        if (price >= 1e3) return `${(price / 1e3).toFixed(1)}K`;
        return price.toFixed(0);
      }
    },
    lastValueVisible: true,
    priceLineVisible: true
  }), []);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    chartRef.current = createChart(chartContainerRef.current, chartOptions);
    
    // Create line series
    seriesRef.current = chartRef.current.addLineSeries(lineOptions);
    
    // Create volume series if enabled
    if (showVolume) {
      volumeSeriesRef.current = chartRef.current.addHistogramSeries({
        color: '#374151',
        priceFormat: {
          type: 'volume' as const
        },
        priceScaleId: 'volume',
        scaleMargins: {
          top: 0.8,
          bottom: 0
        }
      });
    }

    // Set initial data
    if (data.length > 0) {
      const formattedData: LineData[] = data.map(d => ({
        time: Math.floor(d.time / 1000) as any,
        value: d.value
      }));
      seriesRef.current.setData(formattedData);
      
      if (showVolume && volumeSeriesRef.current) {
        const volumeData = data.map(d => ({
          time: Math.floor(d.time / 1000) as any,
          value: d.value * 0.1, // Sample volume data
          color: d.value > 180000 ? '#10b981' : '#ef4444'
        }));
        volumeSeriesRef.current.setData(volumeData);
      }
    }

    // Auto-scale to fit data
    chartRef.current.timeScale().fitContent();

    return () => {
      chartRef.current?.remove();
    };
  }, [theme]); // Recreate chart on theme change

  // Update data
  useEffect(() => {
    if (!seriesRef.current || data.length === 0) return;

    const lastDataPoint = data[data.length - 1];
    const formattedPoint: LineData = {
      time: Math.floor(lastDataPoint.time / 1000) as any,
      value: lastDataPoint.value
    };

    // Update line series
    seriesRef.current.update(formattedPoint);

    // Update volume series
    if (showVolume && volumeSeriesRef.current) {
      volumeSeriesRef.current.update({
        time: Math.floor(lastDataPoint.time / 1000) as any,
        value: lastDataPoint.value * 0.1,
        color: lastDataPoint.value > 180000 ? '#10b981' : '#ef4444'
      });
    }

    // Auto-scroll to latest data
    const scrollToNow = () => {
      if (chartRef.current) {
        const timeScale = chartRef.current.timeScale();
        const currentRange = timeScale.getVisibleRange();
        if (currentRange) {
          const barSpacing = timeScale.width() / (currentRange.to - currentRange.from);
          const barsToShow = 100; // Show last 100 data points
          const scrollPosition = data.length - barsToShow;
          timeScale.scrollToPosition(scrollPosition * barSpacing, false);
        }
      }
    };

    // Debounce scroll to avoid excessive updates
    const scrollTimer = setTimeout(scrollToNow, 100);
    return () => clearTimeout(scrollTimer);
  }, [data, showVolume]);

  // Handle resize
  useEffect(() => {
    if (!chartRef.current) return;

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight
        });
      }
    };

    const resizeObserver = new ResizeObserver(handleResize);
    if (chartContainerRef.current) {
      resizeObserver.observe(chartContainerRef.current);
    }

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  return (
    <div className="relative">
      <div ref={chartContainerRef} className="w-full h-full" />
      
      {/* Stats overlay */}
      <div className="absolute top-2 left-2 bg-gray-900/80 backdrop-blur rounded p-2 text-xs">
        <div className="flex gap-4">
          <div>
            <span className="text-gray-400">Current: </span>
            <span className="text-green-400 font-mono">
              {data.length > 0 ? formatTPS(data[data.length - 1].value) : '0'}
            </span>
          </div>
          <div>
            <span className="text-gray-400">Avg: </span>
            <span className="text-blue-400 font-mono">
              {data.length > 0 ? formatTPS(
                data.reduce((sum, d) => sum + d.value, 0) / data.length
              ) : '0'}
            </span>
          </div>
          <div>
            <span className="text-gray-400">Peak: </span>
            <span className="text-yellow-400 font-mono">
              {data.length > 0 ? formatTPS(
                Math.max(...data.map(d => d.value))
              ) : '0'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function formatTPS(value: number): string {
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toFixed(0);
}