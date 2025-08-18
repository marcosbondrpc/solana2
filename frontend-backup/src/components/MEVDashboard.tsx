/**
 * Ultra-high-performance MEV Dashboard
 * Real-time monitoring with sub-second updates
 */

import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { DataTable, Column } from './DataTable';
import { LatencyHeatmap } from './charts/LatencyHeatmap';
import { TPSChart } from './charts/TPSChart';
import { wsClient } from '../../lib/ws';
import { useAuth } from '../../lib/auth';
import { formatNumber, formatLatency, formatPercentage } from '../../lib/utils';

interface MEVMetrics {
  bundleLandRate: number;
  avgLatencyP50: number;
  avgLatencyP95: number;
  avgLatencyP99: number;
  totalOpportunities: number;
  totalProfitUSD: number;
  successRate: number;
  tps: number;
  activeModels: number;
  systemHealth: 'healthy' | 'degraded' | 'critical';
}

interface MEVEvent {
  id: string;
  timestamp: number;
  type: 'arbitrage' | 'sandwich' | 'liquidation';
  pool: string;
  profit: number;
  latency: number;
  gasUsed: number;
  status: 'success' | 'failed' | 'pending';
  bundleId?: string;
  modelVersion?: string;
}

interface SystemStatus {
  component: string;
  status: 'online' | 'offline' | 'degraded';
  latency: number;
  throughput: number;
  errors: number;
  lastUpdate: number;
}

export function MEVDashboard() {
  const { hasPermission } = useAuth();
  const [metrics, setMetrics] = useState<MEVMetrics>({
    bundleLandRate: 0,
    avgLatencyP50: 0,
    avgLatencyP95: 0,
    avgLatencyP99: 0,
    totalOpportunities: 0,
    totalProfitUSD: 0,
    successRate: 0,
    tps: 0,
    activeModels: 0,
    systemHealth: 'healthy'
  });
  
  const [events, setEvents] = useState<MEVEvent[]>([]);
  const [systemStatus, setSystemStatus] = useState<SystemStatus[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1m' | '5m' | '15m' | '1h' | '24h'>('5m');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [filterType, setFilterType] = useState<string>('all');
  const [latencyData, setLatencyData] = useState<number[][]>([]);
  const [tpsData, setTpsData] = useState<Array<{ time: number; value: number }>>([]);

  // Subscribe to real-time updates
  useEffect(() => {
    if (!autoRefresh) return;

    // Subscribe to metrics stream
    wsClient.subscribe('metrics.mev');
    wsClient.subscribe('events.mev');
    wsClient.subscribe('system.status');

    const handleMetrics = (message: any) => {
      setMetrics(prev => ({
        ...prev,
        ...message.data
      }));
    };

    const handleEvent = (message: any) => {
      const event: MEVEvent = {
        id: message.data.id,
        timestamp: message.data.timestamp,
        type: message.data.type,
        pool: message.data.pool,
        profit: message.data.profit,
        latency: message.data.latency,
        gasUsed: message.data.gasUsed,
        status: message.data.status,
        bundleId: message.data.bundleId,
        modelVersion: message.data.modelVersion
      };

      setEvents(prev => {
        const newEvents = [event, ...prev].slice(0, 1000); // Keep last 1000 events
        return newEvents;
      });
    };

    const handleSystemStatus = (message: any) => {
      setSystemStatus(message.data.components);
    };

    const handleBatch = (messages: any[]) => {
      messages.forEach(msg => {
        if (msg.topic === 'metrics.mev') handleMetrics(msg);
        else if (msg.topic === 'events.mev') handleEvent(msg);
        else if (msg.topic === 'system.status') handleSystemStatus(msg);
      });
    };

    wsClient.on('message', (msg: any) => {
      if (msg.topic === 'metrics.mev') handleMetrics(msg);
      else if (msg.topic === 'events.mev') handleEvent(msg);
      else if (msg.topic === 'system.status') handleSystemStatus(msg);
    });

    wsClient.on('batch', handleBatch);

    // Generate sample data for charts
    const latencyInterval = setInterval(() => {
      const newLatency = Array(24).fill(0).map(() => 
        Array(60).fill(0).map(() => Math.random() * 20)
      );
      setLatencyData(newLatency);

      const now = Date.now();
      setTpsData(prev => {
        const newData = [...prev, { time: now, value: Math.random() * 50000 + 150000 }];
        return newData.slice(-100); // Keep last 100 points
      });
    }, 1000);

    return () => {
      wsClient.unsubscribe('metrics.mev');
      wsClient.unsubscribe('events.mev');
      wsClient.unsubscribe('system.status');
      wsClient.off('message', handleMetrics);
      wsClient.off('batch', handleBatch);
      clearInterval(latencyInterval);
    };
  }, [autoRefresh]);

  // Table columns for events
  const eventColumns: Column[] = useMemo(() => [
    {
      key: 'timestamp',
      header: 'Time',
      width: 150,
      formatter: (value: number) => new Date(value).toLocaleTimeString()
    },
    {
      key: 'type',
      header: 'Type',
      width: 100,
      formatter: (value: string) => (
        <span className={`
          px-2 py-1 rounded text-xs font-medium
          ${value === 'arbitrage' ? 'bg-blue-900/50 text-blue-300' : ''}
          ${value === 'sandwich' ? 'bg-purple-900/50 text-purple-300' : ''}
          ${value === 'liquidation' ? 'bg-red-900/50 text-red-300' : ''}
        `}>
          {value}
        </span>
      )
    },
    {
      key: 'pool',
      header: 'Pool',
      width: 200,
      className: 'font-mono text-xs'
    },
    {
      key: 'profit',
      header: 'Profit (USD)',
      width: 120,
      align: 'right',
      formatter: (value: number) => (
        <span className={value >= 0 ? 'text-green-400' : 'text-red-400'}>
          ${formatNumber(value, 2)}
        </span>
      )
    },
    {
      key: 'latency',
      header: 'Latency',
      width: 100,
      align: 'right',
      formatter: (value: number) => formatLatency(value)
    },
    {
      key: 'gasUsed',
      header: 'Gas',
      width: 100,
      align: 'right',
      formatter: (value: number) => formatNumber(value)
    },
    {
      key: 'status',
      header: 'Status',
      width: 100,
      formatter: (value: string) => (
        <span className={`
          px-2 py-1 rounded text-xs font-medium
          ${value === 'success' ? 'bg-green-900/50 text-green-300' : ''}
          ${value === 'failed' ? 'bg-red-900/50 text-red-300' : ''}
          ${value === 'pending' ? 'bg-yellow-900/50 text-yellow-300' : ''}
        `}>
          {value}
        </span>
      )
    },
    {
      key: 'bundleId',
      header: 'Bundle',
      width: 150,
      className: 'font-mono text-xs truncate'
    }
  ], []);

  // System status columns
  const statusColumns: Column[] = useMemo(() => [
    {
      key: 'component',
      header: 'Component',
      width: 200
    },
    {
      key: 'status',
      header: 'Status',
      width: 100,
      formatter: (value: string) => (
        <span className={`
          flex items-center gap-2
          ${value === 'online' ? 'text-green-400' : ''}
          ${value === 'offline' ? 'text-red-400' : ''}
          ${value === 'degraded' ? 'text-yellow-400' : ''}
        `}>
          <span className={`
            w-2 h-2 rounded-full
            ${value === 'online' ? 'bg-green-400' : ''}
            ${value === 'offline' ? 'bg-red-400' : ''}
            ${value === 'degraded' ? 'bg-yellow-400' : ''}
          `} />
          {value}
        </span>
      )
    },
    {
      key: 'latency',
      header: 'Latency',
      width: 100,
      align: 'right',
      formatter: (value: number) => formatLatency(value)
    },
    {
      key: 'throughput',
      header: 'Throughput',
      width: 120,
      align: 'right',
      formatter: (value: number) => `${formatNumber(value)}/s`
    },
    {
      key: 'errors',
      header: 'Errors',
      width: 80,
      align: 'right',
      formatter: (value: number) => (
        <span className={value > 0 ? 'text-red-400' : ''}>
          {formatNumber(value)}
        </span>
      )
    }
  ], []);

  // Filter events
  const filteredEvents = useMemo(() => {
    if (filterType === 'all') return events;
    return events.filter(e => e.type === filterType);
  }, [events, filterType]);

  // Calculate health status
  const healthColor = useMemo(() => {
    switch (metrics.systemHealth) {
      case 'healthy': return 'text-green-400 bg-green-900/20';
      case 'degraded': return 'text-yellow-400 bg-yellow-900/20';
      case 'critical': return 'text-red-400 bg-red-900/20';
      default: return 'text-gray-400 bg-gray-900/20';
    }
  }, [metrics.systemHealth]);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4">
      {/* Header */}
      <div className="mb-6 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">MEV Dashboard</h1>
          <p className="text-gray-400 mt-1">Real-time MEV monitoring and analytics</p>
        </div>
        <div className="flex gap-4 items-center">
          <select
            value={selectedTimeRange}
            onChange={(e) => setSelectedTimeRange(e.target.value as any)}
            className="px-3 py-2 bg-gray-900 border border-gray-800 rounded-lg"
          >
            <option value="1m">1 Minute</option>
            <option value="5m">5 Minutes</option>
            <option value="15m">15 Minutes</option>
            <option value="1h">1 Hour</option>
            <option value="24h">24 Hours</option>
          </select>
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              autoRefresh
                ? 'bg-green-900/50 text-green-300 border border-green-800'
                : 'bg-gray-900 text-gray-400 border border-gray-800'
            }`}
          >
            {autoRefresh ? 'Auto-Refresh ON' : 'Auto-Refresh OFF'}
          </button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-5 gap-4 mb-6">
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-1">Bundle Land Rate</div>
          <div className="text-2xl font-bold text-white">{formatPercentage(metrics.bundleLandRate)}</div>
          <div className={`text-xs mt-1 ${metrics.bundleLandRate >= 65 ? 'text-green-400' : 'text-red-400'}`}>
            Target: ≥65%
          </div>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-1">Latency P50</div>
          <div className="text-2xl font-bold text-white">{formatLatency(metrics.avgLatencyP50)}</div>
          <div className={`text-xs mt-1 ${metrics.avgLatencyP50 <= 8 ? 'text-green-400' : 'text-red-400'}`}>
            Target: ≤8ms
          </div>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-1">Latency P99</div>
          <div className="text-2xl font-bold text-white">{formatLatency(metrics.avgLatencyP99)}</div>
          <div className={`text-xs mt-1 ${metrics.avgLatencyP99 <= 20 ? 'text-green-400' : 'text-red-400'}`}>
            Target: ≤20ms
          </div>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-1">Total Profit</div>
          <div className="text-2xl font-bold text-green-400">${formatNumber(metrics.totalProfitUSD)}</div>
          <div className="text-xs mt-1 text-gray-500">Last {selectedTimeRange}</div>
        </div>

        <div className={`rounded-lg p-4 border ${healthColor}`}>
          <div className="text-sm mb-1 opacity-80">System Health</div>
          <div className="text-2xl font-bold uppercase">{metrics.systemHealth}</div>
          <div className="text-xs mt-1 opacity-60">{metrics.activeModels} models active</div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold mb-4">Latency Heatmap</h2>
          <LatencyHeatmap data={latencyData} width={600} height={300} />
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold mb-4">TPS Chart</h2>
          <TPSChart data={tpsData} width={600} height={300} />
        </div>
      </div>

      {/* Event Feed */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">MEV Events</h2>
          <div className="flex gap-2">
            {['all', 'arbitrage', 'sandwich', 'liquidation'].map(type => (
              <button
                key={type}
                onClick={() => setFilterType(type)}
                className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                  filterType === type
                    ? 'bg-blue-900/50 text-blue-300 border border-blue-800'
                    : 'bg-gray-900 text-gray-400 border border-gray-800'
                }`}
              >
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </button>
            ))}
          </div>
        </div>
        
        <DataTable
          data={filteredEvents}
          columns={eventColumns}
          height={400}
          virtualize={true}
          highlightOnHover={true}
          zebra={true}
          className="border-gray-800"
        />
      </div>

      {/* System Status */}
      {hasPermission('system.monitor') && (
        <div>
          <h2 className="text-xl font-semibold mb-4">System Status</h2>
          <DataTable
            data={systemStatus}
            columns={statusColumns}
            height={300}
            virtualize={false}
            highlightOnHover={true}
            className="border-gray-800"
          />
        </div>
      )}
    </div>
  );
}

// Utility functions
function formatNumber(value: number, decimals = 0): string {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  }).format(value);
}

function formatLatency(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
  if (ms < 1000) return `${ms.toFixed(1)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}