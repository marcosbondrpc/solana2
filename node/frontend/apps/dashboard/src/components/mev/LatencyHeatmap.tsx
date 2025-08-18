'use client';

import React, { useEffect, useRef, useState, useMemo } from 'react';
import { useMEVStore, selectLatencyStats } from '@/stores/mev-store';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import * as d3 from 'd3';
import { 
  Zap, 
  Wifi, 
  Activity,
  Gauge,
  TrendingDown,
  TrendingUp,
  AlertTriangle,
  Server
} from 'lucide-react';
import ReactECharts from 'echarts-for-react';

interface HeatmapData {
  hour: number;
  minute: number;
  value: number;
  category: string;
}

export const LatencyHeatmap: React.FC = () => {
  const { latencyMetrics } = useMEVStore();
  const latencyStats = useMEVStore(selectLatencyStats);
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedMetric, setSelectedMetric] = useState<'rpc' | 'quic' | 'parsing' | 'execution'>('quic');
  const [timeRange, setTimeRange] = useState<'1h' | '6h' | '24h'>('1h');
  const [hoveredCell, setHoveredCell] = useState<HeatmapData | null>(null);
  
  // Generate heatmap data
  const heatmapData = useMemo(() => {
    const now = Date.now();
    const rangeMs = timeRange === '1h' ? 3600000 : timeRange === '6h' ? 21600000 : 86400000;
    const startTime = now - rangeMs;
    
    const filtered = latencyMetrics.filter(m => m.timestamp >= startTime);
    
    // Create time buckets (5-minute intervals)
    const bucketSize = 300000; // 5 minutes
    const buckets = new Map<string, number[]>();
    
    filtered.forEach(metric => {
      const bucketTime = Math.floor(metric.timestamp / bucketSize) * bucketSize;
      const key = `${bucketTime}`;
      
      if (!buckets.has(key)) {
        buckets.set(key, []);
      }
      
      let value = 0;
      switch (selectedMetric) {
        case 'rpc':
          value = metric.rpcLatency;
          break;
        case 'quic':
          value = metric.quicLatency;
          break;
        case 'parsing':
          value = metric.parsingLatency;
          break;
        case 'execution':
          value = metric.executionLatency;
          break;
      }
      
      buckets.get(key)!.push(value);
    });
    
    // Convert to heatmap format
    const data: HeatmapData[] = [];
    buckets.forEach((values, timeStr) => {
      const time = parseInt(timeStr);
      const date = new Date(time);
      const avgValue = values.reduce((a, b) => a + b, 0) / values.length;
      
      data.push({
        hour: date.getHours(),
        minute: Math.floor(date.getMinutes() / 5) * 5,
        value: avgValue,
        category: selectedMetric
      });
    });
    
    return data;
  }, [latencyMetrics, selectedMetric, timeRange]);
  
  // D3 Heatmap rendering
  useEffect(() => {
    if (!svgRef.current || heatmapData.length === 0) return;
    
    const margin = { top: 30, right: 30, bottom: 30, left: 60 };
    const width = containerRef.current?.clientWidth || 800;
    const height = 400;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Clear previous render
    d3.select(svgRef.current).selectAll('*').remove();
    
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);
    
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Scales
    const xScale = d3.scaleBand()
      .domain(Array.from({ length: 12 }, (_, i) => i * 5))
      .range([0, innerWidth])
      .padding(0.05);
    
    const yScale = d3.scaleBand()
      .domain(Array.from({ length: 24 }, (_, i) => i))
      .range([0, innerHeight])
      .padding(0.05);
    
    const colorScale = d3.scaleSequential()
      .interpolator(d3.interpolateInferno)
      .domain([0, d3.max(heatmapData, d => d.value) || 100]);
    
    // Draw cells
    g.selectAll('.cell')
      .data(heatmapData)
      .enter()
      .append('rect')
      .attr('class', 'cell')
      .attr('x', d => xScale(d.minute) || 0)
      .attr('y', d => yScale(d.hour) || 0)
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('fill', d => colorScale(d.value))
      .attr('stroke', 'rgba(0,0,0,0.1)')
      .attr('rx', 2)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this)
          .attr('stroke', '#fff')
          .attr('stroke-width', 2);
        setHoveredCell(d as HeatmapData);
      })
      .on('mouseout', function() {
        d3.select(this)
          .attr('stroke', 'rgba(0,0,0,0.1)')
          .attr('stroke-width', 1);
        setHoveredCell(null);
      });
    
    // X axis
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).tickFormat(d => `${d}m`))
      .append('text')
      .attr('x', innerWidth / 2)
      .attr('y', 25)
      .attr('fill', 'currentColor')
      .style('text-anchor', 'middle')
      .style('font-size', '12px')
      .text('Minute');
    
    // Y axis
    g.append('g')
      .call(d3.axisLeft(yScale).tickFormat(d => `${d}:00`))
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -40)
      .attr('x', -innerHeight / 2)
      .attr('fill', 'currentColor')
      .style('text-anchor', 'middle')
      .style('font-size', '12px')
      .text('Hour');
    
    // Color legend
    const legendWidth = 200;
    const legendHeight = 10;
    
    const legendScale = d3.scaleLinear()
      .domain(colorScale.domain())
      .range([0, legendWidth]);
    
    const legendAxis = d3.axisBottom(legendScale)
      .ticks(5)
      .tickFormat(d => `${d}ms`);
    
    const legend = svg.append('g')
      .attr('transform', `translate(${width - legendWidth - 30},${10})`);
    
    // Create gradient for legend
    const gradientId = 'latency-gradient';
    const gradient = svg.append('defs')
      .append('linearGradient')
      .attr('id', gradientId)
      .attr('x1', '0%')
      .attr('x2', '100%');
    
    const steps = 20;
    for (let i = 0; i <= steps; i++) {
      gradient.append('stop')
        .attr('offset', `${(i / steps) * 100}%`)
        .attr('stop-color', colorScale(i / steps * (colorScale.domain()[1] || 100)));
    }
    
    legend.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', `url(#${gradientId})`);
    
    legend.append('g')
      .attr('transform', `translate(0,${legendHeight})`)
      .call(legendAxis);
    
  }, [heatmapData]);
  
  // Line chart for comparison
  const comparisonChartOptions = useMemo(() => {
    const timestamps = latencyMetrics.slice(-100).map(m => 
      new Date(m.timestamp).toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit',
        hour12: false 
      })
    );
    
    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross'
        }
      },
      legend: {
        data: ['RPC', 'QUIC', 'TCP', 'WebSocket'],
        top: 10
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        top: '15%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: timestamps,
        axisLabel: {
          rotate: 45,
          fontSize: 10
        }
      },
      yAxis: {
        type: 'value',
        name: 'Latency (ms)',
        axisLabel: {
          formatter: '{value}ms'
        }
      },
      series: [
        {
          name: 'RPC',
          type: 'line',
          smooth: true,
          data: latencyMetrics.slice(-100).map(m => m.rpcLatency.toFixed(2)),
          itemStyle: { color: '#3b82f6' }
        },
        {
          name: 'QUIC',
          type: 'line',
          smooth: true,
          data: latencyMetrics.slice(-100).map(m => m.quicLatency.toFixed(2)),
          itemStyle: { color: '#10b981' }
        },
        {
          name: 'TCP',
          type: 'line',
          smooth: true,
          data: latencyMetrics.slice(-100).map(m => m.tcpLatency.toFixed(2)),
          itemStyle: { color: '#f59e0b' }
        },
        {
          name: 'WebSocket',
          type: 'line',
          smooth: true,
          data: latencyMetrics.slice(-100).map(m => m.websocketLatency.toFixed(2)),
          itemStyle: { color: '#8b5cf6' }
        }
      ]
    };
  }, [latencyMetrics]);
  
  // Percentile chart
  const percentileChartOptions = useMemo(() => {
    const recent = latencyMetrics.slice(-60);
    if (recent.length === 0) return {};
    
    return {
      tooltip: {
        trigger: 'axis'
      },
      radar: {
        indicator: [
          { name: 'P50', max: 100 },
          { name: 'P95', max: 200 },
          { name: 'P99', max: 300 },
          { name: 'Jitter', max: 50 },
          { name: 'Min', max: 50 },
          { name: 'Max', max: 500 }
        ]
      },
      series: [{
        type: 'radar',
        data: [{
          value: [
            recent[recent.length - 1]?.p50 || 0,
            recent[recent.length - 1]?.p95 || 0,
            recent[recent.length - 1]?.p99 || 0,
            recent[recent.length - 1]?.jitter || 0,
            Math.min(...recent.map(m => m.rpcLatency)),
            Math.max(...recent.map(m => m.rpcLatency))
          ],
          name: 'Current',
          areaStyle: {
            color: 'rgba(59, 130, 246, 0.2)'
          },
          lineStyle: {
            color: '#3b82f6'
          }
        }]
      }]
    };
  }, [latencyMetrics]);
  
  const getLatencyStatus = (value: number) => {
    if (value < 10) return { color: 'text-green-500', label: 'Excellent', icon: TrendingDown };
    if (value < 50) return { color: 'text-yellow-500', label: 'Good', icon: Activity };
    if (value < 100) return { color: 'text-orange-500', label: 'Fair', icon: AlertTriangle };
    return { color: 'text-red-500', label: 'Poor', icon: TrendingUp };
  };
  
  return (
    <div className="space-y-4">
      {/* Stats Cards */}
      <div className="grid grid-cols-4 gap-4">
        {latencyStats && (
          <>
            <Card className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-gray-500">Avg RPC Latency</div>
                  <div className={`text-2xl font-bold ${getLatencyStatus(latencyStats.avgRpc).color}`}>
                    {latencyStats.avgRpc.toFixed(2)}ms
                  </div>
                  <Badge variant="outline" className="mt-1">
                    {getLatencyStatus(latencyStats.avgRpc).label}
                  </Badge>
                </div>
                <Server className="w-8 h-8 text-blue-500" />
              </div>
            </Card>
            
            <Card className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-gray-500">Avg QUIC Latency</div>
                  <div className={`text-2xl font-bold ${getLatencyStatus(latencyStats.avgQuic).color}`}>
                    {latencyStats.avgQuic.toFixed(2)}ms
                  </div>
                  <Badge variant="outline" className="mt-1">
                    {getLatencyStatus(latencyStats.avgQuic).label}
                  </Badge>
                </div>
                <Zap className="w-8 h-8 text-green-500" />
              </div>
            </Card>
            
            <Card className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-gray-500">QUIC Advantage</div>
                  <div className="text-2xl font-bold text-green-500">
                    {((1 - latencyStats.avgQuic / latencyStats.avgRpc) * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500 mt-1">faster than RPC</div>
                </div>
                <Gauge className="w-8 h-8 text-purple-500" />
              </div>
            </Card>
            
            <Card className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-gray-500">Network Health</div>
                  <div className="text-2xl font-bold">
                    {latencyStats.avgQuic < 20 ? 'Optimal' : 
                     latencyStats.avgQuic < 50 ? 'Good' : 'Degraded'}
                  </div>
                  <Badge 
                    variant={latencyStats.avgQuic < 20 ? 'default' : 'outline'}
                    className="mt-1"
                  >
                    {latencyStats.minQuic.toFixed(0)}-{latencyStats.maxQuic.toFixed(0)}ms range
                  </Badge>
                </div>
                <Wifi className="w-8 h-8 text-cyan-500" />
              </div>
            </Card>
          </>
        )}
      </div>
      
      {/* Controls */}
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">Metric:</label>
              <div className="flex gap-1">
                {(['rpc', 'quic', 'parsing', 'execution'] as const).map(metric => (
                  <Button
                    key={metric}
                    size="sm"
                    variant={selectedMetric === metric ? 'default' : 'outline'}
                    onClick={() => setSelectedMetric(metric)}
                  >
                    {metric.toUpperCase()}
                  </Button>
                ))}
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">Range:</label>
              <div className="flex gap-1">
                {(['1h', '6h', '24h'] as const).map(range => (
                  <Button
                    key={range}
                    size="sm"
                    variant={timeRange === range ? 'default' : 'outline'}
                    onClick={() => setTimeRange(range)}
                  >
                    {range}
                  </Button>
                ))}
              </div>
            </div>
          </div>
          
          {hoveredCell && (
            <div className="text-sm">
              <span className="font-medium">
                {hoveredCell.hour}:{String(hoveredCell.minute).padStart(2, '0')}
              </span>
              <span className="ml-2 text-gray-500">
                {hoveredCell.value.toFixed(2)}ms
              </span>
            </div>
          )}
        </div>
      </Card>
      
      {/* Heatmap */}
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          Latency Heatmap - {selectedMetric.toUpperCase()}
        </h3>
        <div ref={containerRef} className="w-full">
          <svg ref={svgRef}></svg>
        </div>
      </Card>
      
      {/* Comparison Charts */}
      <div className="grid grid-cols-2 gap-4">
        <Card className="p-4">
          <h3 className="text-sm font-semibold mb-2">Protocol Comparison</h3>
          <ReactECharts option={comparisonChartOptions} style={{ height: '300px' }} />
        </Card>
        
        <Card className="p-4">
          <h3 className="text-sm font-semibold mb-2">Latency Percentiles</h3>
          <ReactECharts option={percentileChartOptions} style={{ height: '300px' }} />
        </Card>
      </div>
    </div>
  );
};