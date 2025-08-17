/**
 * Node Overview Page - System health and real-time metrics
 */

import React, { useState, useEffect } from 'react';
import { DataTable } from '../components/DataTable';
import { TPSChart } from '../components/charts/TPSChart';
import { wsClient } from '../../lib/ws';
import { useAuth } from '../../lib/auth';
import { formatNumber, formatLatency, formatBytes, formatDuration, formatPercentage } from '../../lib/utils';

interface NodeMetrics {
  nodeId: string;
  status: 'healthy' | 'degraded' | 'offline';
  cpu: number;
  memory: number;
  disk: number;
  network: {
    in: number;
    out: number;
  };
  uptime: number;
  blockHeight: number;
  peersConnected: number;
  tps: number;
  latency: {
    p50: number;
    p95: number;
    p99: number;
  };
}

interface RecentBlock {
  height: number;
  hash: string;
  timestamp: number;
  txCount: number;
  validator: string;
  rewards: number;
}

export function NodeOverview() {
  const { hasPermission } = useAuth();
  const [nodes, setNodes] = useState<NodeMetrics[]>([]);
  const [recentBlocks, setRecentBlocks] = useState<RecentBlock[]>([]);
  const [tpsHistory, setTpsHistory] = useState<Array<{ time: number; value: number }>>([]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    if (!autoRefresh) return;

    // Subscribe to node metrics
    wsClient.subscribe('node.metrics');
    wsClient.subscribe('blocks.recent');

    const handleNodeMetrics = (message: any) => {
      setNodes(message.data.nodes);
      
      // Update TPS history
      const totalTps = message.data.nodes.reduce((sum: number, node: NodeMetrics) => sum + node.tps, 0);
      setTpsHistory(prev => [...prev, { time: Date.now(), value: totalTps }].slice(-100));
    };

    const handleBlocks = (message: any) => {
      setRecentBlocks(message.data.blocks);
    };

    wsClient.on('message', (msg: any) => {
      if (msg.topic === 'node.metrics') handleNodeMetrics(msg);
      else if (msg.topic === 'blocks.recent') handleBlocks(msg);
    });

    // Generate sample data
    const sampleInterval = setInterval(() => {
      const sampleNodes: NodeMetrics[] = [
        {
          nodeId: 'node-001',
          status: 'healthy',
          cpu: Math.random() * 100,
          memory: Math.random() * 100,
          disk: Math.random() * 100,
          network: {
            in: Math.random() * 1e9,
            out: Math.random() * 1e9
          },
          uptime: Date.now() - 86400000 * Math.random() * 30,
          blockHeight: 250000000 + Math.floor(Math.random() * 1000),
          peersConnected: Math.floor(Math.random() * 100),
          tps: Math.random() * 50000 + 150000,
          latency: {
            p50: Math.random() * 10,
            p95: Math.random() * 20,
            p99: Math.random() * 30
          }
        },
        {
          nodeId: 'node-002',
          status: Math.random() > 0.9 ? 'degraded' : 'healthy',
          cpu: Math.random() * 100,
          memory: Math.random() * 100,
          disk: Math.random() * 100,
          network: {
            in: Math.random() * 1e9,
            out: Math.random() * 1e9
          },
          uptime: Date.now() - 86400000 * Math.random() * 30,
          blockHeight: 250000000 + Math.floor(Math.random() * 1000),
          peersConnected: Math.floor(Math.random() * 100),
          tps: Math.random() * 50000 + 150000,
          latency: {
            p50: Math.random() * 10,
            p95: Math.random() * 20,
            p99: Math.random() * 30
          }
        }
      ];
      setNodes(sampleNodes);

      const sampleBlocks: RecentBlock[] = Array(10).fill(0).map((_, i) => ({
        height: 250000000 - i,
        hash: `0x${Math.random().toString(16).substr(2, 16)}`,
        timestamp: Date.now() - i * 400,
        txCount: Math.floor(Math.random() * 3000),
        validator: `Validator${Math.floor(Math.random() * 10)}`,
        rewards: Math.random() * 10
      }));
      setRecentBlocks(sampleBlocks);

      setTpsHistory(prev => [
        ...prev, 
        { time: Date.now(), value: Math.random() * 50000 + 150000 }
      ].slice(-100));
    }, 1000);

    return () => {
      wsClient.unsubscribe('node.metrics');
      wsClient.unsubscribe('blocks.recent');
      clearInterval(sampleInterval);
    };
  }, [autoRefresh]);

  const nodeColumns = [
    {
      key: 'nodeId',
      header: 'Node ID',
      width: 150
    },
    {
      key: 'status',
      header: 'Status',
      width: 100,
      formatter: (value: string) => (
        <span className={`
          flex items-center gap-2
          ${value === 'healthy' ? 'text-green-400' : ''}
          ${value === 'degraded' ? 'text-yellow-400' : ''}
          ${value === 'offline' ? 'text-red-400' : ''}
        `}>
          <span className={`
            w-2 h-2 rounded-full
            ${value === 'healthy' ? 'bg-green-400' : ''}
            ${value === 'degraded' ? 'bg-yellow-400' : ''}
            ${value === 'offline' ? 'bg-red-400' : ''}
          `} />
          {value}
        </span>
      )
    },
    {
      key: 'cpu',
      header: 'CPU %',
      width: 100,
      align: 'right' as const,
      formatter: (value: number) => (
        <span className={value > 80 ? 'text-red-400' : value > 60 ? 'text-yellow-400' : ''}>
          {formatPercentage(value / 100)}
        </span>
      )
    },
    {
      key: 'memory',
      header: 'Memory %',
      width: 100,
      align: 'right' as const,
      formatter: (value: number) => (
        <span className={value > 80 ? 'text-red-400' : value > 60 ? 'text-yellow-400' : ''}>
          {formatPercentage(value / 100)}
        </span>
      )
    },
    {
      key: 'tps',
      header: 'TPS',
      width: 120,
      align: 'right' as const,
      formatter: (value: number) => formatNumber(value)
    },
    {
      key: 'blockHeight',
      header: 'Block Height',
      width: 150,
      align: 'right' as const,
      formatter: (value: number) => formatNumber(value)
    },
    {
      key: 'peersConnected',
      header: 'Peers',
      width: 80,
      align: 'right' as const
    },
    {
      key: 'uptime',
      header: 'Uptime',
      width: 120,
      formatter: (value: number) => formatDuration(Date.now() - value)
    }
  ];

  const blockColumns = [
    {
      key: 'height',
      header: 'Height',
      width: 120,
      formatter: (value: number) => formatNumber(value)
    },
    {
      key: 'hash',
      header: 'Hash',
      width: 200,
      className: 'font-mono text-xs',
      formatter: (value: string) => value.substring(0, 10) + '...'
    },
    {
      key: 'timestamp',
      header: 'Time',
      width: 150,
      formatter: (value: number) => new Date(value).toLocaleTimeString()
    },
    {
      key: 'txCount',
      header: 'Transactions',
      width: 120,
      align: 'right' as const,
      formatter: (value: number) => formatNumber(value)
    },
    {
      key: 'validator',
      header: 'Validator',
      width: 150
    },
    {
      key: 'rewards',
      header: 'Rewards (SOL)',
      width: 120,
      align: 'right' as const,
      formatter: (value: number) => value.toFixed(4)
    }
  ];

  const selectedNodeData = selectedNode ? nodes.find(n => n.nodeId === selectedNode) : null;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4">
      <div className="mb-6 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">Node Overview</h1>
          <p className="text-gray-400 mt-1">Real-time Solana node monitoring</p>
        </div>
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

      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-1">Total Nodes</div>
          <div className="text-2xl font-bold text-white">{nodes.length}</div>
          <div className="text-xs mt-1 text-green-400">
            {nodes.filter(n => n.status === 'healthy').length} healthy
          </div>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-1">Network TPS</div>
          <div className="text-2xl font-bold text-white">
            {formatNumber(nodes.reduce((sum, n) => sum + n.tps, 0))}
          </div>
          <div className="text-xs mt-1 text-gray-500">Combined throughput</div>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-1">Latest Block</div>
          <div className="text-2xl font-bold text-white">
            {recentBlocks[0] ? formatNumber(recentBlocks[0].height) : '—'}
          </div>
          <div className="text-xs mt-1 text-gray-500">
            {recentBlocks[0] ? `${recentBlocks[0].txCount} transactions` : ''}
          </div>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-1">Avg Latency</div>
          <div className="text-2xl font-bold text-white">
            {formatLatency(
              nodes.reduce((sum, n) => sum + n.latency.p50, 0) / (nodes.length || 1)
            )}
          </div>
          <div className="text-xs mt-1 text-gray-500">P50 across all nodes</div>
        </div>
      </div>

      {/* TPS Chart */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
        <h2 className="text-lg font-semibold mb-4">Network TPS</h2>
        <TPSChart data={tpsHistory} height={300} theme="dark" />
      </div>

      {/* Node Status Table */}
      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-4">Node Status</h2>
        <DataTable
          data={nodes}
          columns={nodeColumns}
          height={300}
          onRowClick={(node) => setSelectedNode(node.nodeId)}
          highlightOnHover={true}
          className="border-gray-800"
        />
      </div>

      {/* Selected Node Details */}
      {selectedNodeData && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
          <h3 className="text-lg font-semibold mb-3">Node Details: {selectedNodeData.nodeId}</h3>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-gray-400 text-sm">Network I/O</div>
              <div className="text-white">
                ↓ {formatBytes(selectedNodeData.network.in)}/s | 
                ↑ {formatBytes(selectedNodeData.network.out)}/s
              </div>
            </div>
            <div>
              <div className="text-gray-400 text-sm">Disk Usage</div>
              <div className="text-white">{formatPercentage(selectedNodeData.disk / 100)}</div>
            </div>
            <div>
              <div className="text-gray-400 text-sm">Latency (P50/P95/P99)</div>
              <div className="text-white">
                {formatLatency(selectedNodeData.latency.p50)} / 
                {formatLatency(selectedNodeData.latency.p95)} / 
                {formatLatency(selectedNodeData.latency.p99)}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Recent Blocks */}
      <div>
        <h2 className="text-xl font-semibold mb-4">Recent Blocks</h2>
        <DataTable
          data={recentBlocks}
          columns={blockColumns}
          height={300}
          highlightOnHover={true}
          className="border-gray-800"
        />
      </div>
    </div>
  );
}