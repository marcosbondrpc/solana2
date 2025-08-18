/**
 * Thompson Sampling Bandit Performance Dashboard
 * Real-time visualization of arm performance and budget allocation
 */

import { useEffect, useRef, useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import * as d3 from 'd3';
import { useBanditStore, selectArmPerformance } from '../../stores/bandit-store';
import { WSProtoClient } from '../../lib/ws-proto';

interface ArmMetrics {
  id: string;
  route: string;
  successRate: number;
  expectedValue: number;
  pulls: number;
  profit: number;
  latency: number;
  confidence: number;
  budget: number;
  weekOverWeekGrowth?: number;
}

const ROUTE_COLORS = {
  direct: '#10b981',
  jito: '#8b5cf6',
  hedged: '#f59e0b'
};

export function BanditPerformancePanel() {
  const arms = useBanditStore(selectArmPerformance);
  const updateArm = useBanditStore(state => state.updateArm);
  const recordCanary = useBanditStore(state => state.recordCanary);
  const rebalanceBudget = useBanditStore(state => state.rebalanceBudget);
  const samplingState = useBanditStore(state => state.samplingState);
  
  const [selectedArm, setSelectedArm] = useState<string>('direct');
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');
  const [wsClient, setWsClient] = useState<WSProtoClient | null>(null);
  
  const svgRef = useRef<SVGSVGElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Initialize WebSocket for real-time updates
  useEffect(() => {
    const client = new WSProtoClient({
      url: 'ws://localhost:8001/bandit',
      enableWorker: true,
      binaryMode: true
    });

    client.on('bandit:state', (data: any) => {
      if (data.armId && data.metrics) {
        updateArm(data.armId, data.metrics);
      }
    });

    client.on('canary:result', (data: any) => {
      recordCanary({
        id: data.id || `canary-${Date.now()}`,
        timestamp: data.timestamp || Date.now(),
        armId: data.armId,
        route: data.route,
        amount: data.amount,
        success: data.success,
        latency: data.latency,
        landingSlot: data.landingSlot,
        profit: data.profit,
        gasUsed: data.gasUsed,
        errorReason: data.errorReason
      });
    });

    client.connect();
    setWsClient(client);

    return () => {
      client.disconnect();
    };
  }, [updateArm, recordCanary]);

  // Auto-rebalance budget every hour
  useEffect(() => {
    const interval = setInterval(() => {
      rebalanceBudget();
    }, 3600000);
    
    return () => clearInterval(interval);
  }, [rebalanceBudget]);

  // D3 visualization for arm performance
  useEffect(() => {
    if (!svgRef.current || arms.length === 0) return;

    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;
    
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleBand()
      .domain(arms.map(a => a.route))
      .range([0, innerWidth])
      .padding(0.3);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(arms, a => a.successRate) || 1])
      .range([innerHeight, 0]);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale))
      .attr('color', '#6b7280');

    g.append('g')
      .call(d3.axisLeft(yScale).tickFormat(d => `${(d as number * 100).toFixed(0)}%`))
      .attr('color', '#6b7280');

    // Bars
    g.selectAll('.bar')
      .data(arms)
      .enter().append('rect')
      .attr('class', 'bar')
      .attr('x', d => xScale(d.route)!)
      .attr('y', d => yScale(d.successRate))
      .attr('width', xScale.bandwidth())
      .attr('height', d => innerHeight - yScale(d.successRate))
      .attr('fill', d => ROUTE_COLORS[d.route as keyof typeof ROUTE_COLORS])
      .attr('opacity', 0.8)
      .on('mouseover', function(event, d) {
        d3.select(this).attr('opacity', 1);
      })
      .on('mouseout', function(event, d) {
        d3.select(this).attr('opacity', 0.8);
      });

    // Confidence intervals
    g.selectAll('.confidence')
      .data(arms)
      .enter().append('line')
      .attr('class', 'confidence')
      .attr('x1', d => xScale(d.route)! + xScale.bandwidth() / 2)
      .attr('x2', d => xScale(d.route)! + xScale.bandwidth() / 2)
      .attr('y1', d => yScale(Math.max(0, d.successRate - (1 - d.confidence) / 2)))
      .attr('y2', d => yScale(Math.min(1, d.successRate + (1 - d.confidence) / 2)))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .attr('opacity', 0.5);

  }, [arms]);

  // WebGL canvas for real-time metrics
  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#030712';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = '#1f2937';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 10; i++) {
      const y = (height / 10) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw EV trends
    const arm = arms.find(a => a.id === selectedArm);
    if (arm) {
      ctx.strokeStyle = ROUTE_COLORS[arm.route as keyof typeof ROUTE_COLORS];
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      // Simulate historical data points
      for (let i = 0; i < 100; i++) {
        const x = (width / 100) * i;
        const y = height - (Math.random() * height * 0.8 + height * 0.1);
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      
      ctx.stroke();
    }
  }, [arms, selectedArm]);

  // Calculate summary statistics
  const stats = useMemo(() => {
    const totalPulls = arms.reduce((sum, a) => sum + a.pulls, 0);
    const totalProfit = arms.reduce((sum, a) => sum + a.profit, 0);
    const avgLatency = arms.reduce((sum, a) => sum + a.latency * a.pulls, 0) / Math.max(1, totalPulls);
    const bestArm = arms.reduce((best, a) => a.expectedValue > best.expectedValue ? a : best, arms[0]);
    
    return {
      totalPulls,
      totalProfit,
      avgLatency,
      bestArm: bestArm?.route || 'none'
    };
  }, [arms]);

  return (
    <div className="flex flex-col h-full bg-gray-950 rounded-lg border border-gray-800">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div className="flex items-center space-x-4">
          <h2 className="text-lg font-bold text-white">Bandit Performance</h2>
          <div className="flex items-center space-x-2">
            <div className="px-2 py-1 rounded bg-blue-900/20 text-blue-400 text-xs">
              Epoch {samplingState.epoch}
            </div>
            <div className="px-2 py-1 rounded bg-purple-900/20 text-purple-400 text-xs">
              T={samplingState.temperature.toFixed(2)}
            </div>
            <div className="px-2 py-1 rounded bg-green-900/20 text-green-400 text-xs">
              Best: {stats.bestArm.toUpperCase()}
            </div>
          </div>
        </div>

        {/* Time range selector */}
        <div className="flex items-center space-x-2">
          {(['1h', '24h', '7d', '30d'] as const).map(range => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-3 py-1 text-xs rounded ${
                timeRange === range
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 p-4">
        <div className="grid grid-cols-3 gap-4 h-full">
          {/* Arms performance */}
          <div className="col-span-2 space-y-4">
            {/* Success rate chart */}
            <div className="bg-gray-900 rounded p-4 h-64">
              <h3 className="text-sm font-medium text-gray-400 mb-2">Success Rate by Route</h3>
              <svg ref={svgRef} className="w-full h-48" />
            </div>

            {/* Expected value trends */}
            <div className="bg-gray-900 rounded p-4 h-64">
              <h3 className="text-sm font-medium text-gray-400 mb-2">Expected Value Trends</h3>
              <canvas 
                ref={canvasRef} 
                width={600} 
                height={200}
                className="w-full h-48"
              />
            </div>
          </div>

          {/* Arm details */}
          <div className="space-y-4">
            {/* Arm selector */}
            <div className="bg-gray-900 rounded p-4">
              <h3 className="text-sm font-medium text-gray-400 mb-3">Route Details</h3>
              <div className="space-y-2">
                {arms.map(arm => (
                  <motion.div
                    key={arm.id}
                    whileHover={{ scale: 1.02 }}
                    onClick={() => setSelectedArm(arm.id)}
                    className={`p-3 rounded cursor-pointer border ${
                      selectedArm === arm.id
                        ? 'border-blue-500 bg-blue-900/20'
                        : 'border-gray-700 bg-gray-800/50 hover:bg-gray-800'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-white">
                        {arm.route.toUpperCase()}
                      </span>
                      <span 
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: ROUTE_COLORS[arm.route as keyof typeof ROUTE_COLORS] }}
                      />
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <span className="text-gray-500">Success</span>
                        <div className="text-white font-medium">
                          {(arm.successRate * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div>
                        <span className="text-gray-500">Pulls</span>
                        <div className="text-white font-medium">
                          {arm.pulls.toLocaleString()}
                        </div>
                      </div>
                      <div>
                        <span className="text-gray-500">Profit</span>
                        <div className="text-green-400 font-medium">
                          ${arm.profit.toFixed(2)}
                        </div>
                      </div>
                      <div>
                        <span className="text-gray-500">Latency</span>
                        <div className="text-white font-medium">
                          {arm.latency.toFixed(1)}ms
                        </div>
                      </div>
                    </div>

                    {/* Budget allocation */}
                    <div className="mt-2 pt-2 border-t border-gray-700">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-gray-500">Budget</span>
                        <span className="text-white">${arm.budget.toFixed(2)}</span>
                      </div>
                      <div className="mt-1 w-full h-1 bg-gray-700 rounded">
                        <div 
                          className="h-1 rounded"
                          style={{
                            width: `${(arm.budget / 1000) * 100}%`,
                            backgroundColor: ROUTE_COLORS[arm.route as keyof typeof ROUTE_COLORS]
                          }}
                        />
                      </div>
                    </div>

                    {/* Week over week growth */}
                    {arm.weekOverWeekGrowth !== undefined && (
                      <div className="mt-2 flex items-center justify-between text-xs">
                        <span className="text-gray-500">WoW Growth</span>
                        <span className={arm.weekOverWeekGrowth > 0 ? 'text-green-400' : 'text-red-400'}>
                          {arm.weekOverWeekGrowth > 0 ? '+' : ''}{(arm.weekOverWeekGrowth * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Summary stats */}
            <div className="bg-gray-900 rounded p-4">
              <h3 className="text-sm font-medium text-gray-400 mb-3">Summary</h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-500">Total Pulls</span>
                  <span className="text-white font-medium">{stats.totalPulls.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Total Profit</span>
                  <span className="text-green-400 font-medium">${stats.totalProfit.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Avg Latency</span>
                  <span className="text-white font-medium">{stats.avgLatency.toFixed(1)}ms</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Temperature</span>
                  <span className="text-white font-medium">{samplingState.temperature.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Learning Rate</span>
                  <span className="text-white font-medium">{samplingState.learningRate.toFixed(3)}</span>
                </div>
              </div>
            </div>

            {/* Rebalance button */}
            <button
              onClick={() => rebalanceBudget()}
              className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded transition-colors"
            >
              Rebalance Budget
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}