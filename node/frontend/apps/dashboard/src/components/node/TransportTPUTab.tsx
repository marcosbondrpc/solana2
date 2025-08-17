import { motion } from 'framer-motion';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import { 
  Network, 
  Shield,
  Activity,
  Server,
  Gauge,
  Users,
  AlertTriangle,
  CheckCircle,
  Wifi,
  GitBranch,
  Radio
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  AreaChart,
  Area,
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid,
  Tooltip, 
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Treemap
} from 'recharts';
import type { NodeStatus } from '../../pages/Node';

interface TransportTPUTabProps {
  nodeStatus: NodeStatus;
}

export default function TransportTPUTab({ nodeStatus }: TransportTPUTabProps) {
  // Transform open streams data for visualization
  const streamData = Object.entries(nodeStatus.transport.quic.openStreamsPerPeer)
    .slice(0, 10)
    .map(([peer, streams]) => ({
      peer: peer.slice(0, 8) + '...',
      streams
    }));

  // PPS rate limiting history
  const ppsData = nodeStatus.transport.quic.ppsRateLimiting.map((rate, i) => ({
    time: i,
    rate
  }));

  // Retransmits data
  const retransmitData = nodeStatus.transport.gossip.retransmits.map((count, i) => ({
    time: i,
    retransmits: count
  }));

  // Whitelisted RPCs for treemap
  const rpcTreemapData = nodeStatus.transport.qos.whitelistedRpcs.map(rpc => ({
    name: rpc.address.slice(0, 12) + '...',
    size: rpc.stake,
    value: rpc.stake
  }));

  // QUIC health metrics for radar chart
  const quicHealthData = [
    { metric: 'Handshake', value: nodeStatus.transport.quic.handshakeSuccessRate },
    { metric: 'Connections', value: Math.min((nodeStatus.transport.quic.concurrentConnections / 1000) * 100, 100) },
    { metric: 'Throttling', value: 100 - Math.min(nodeStatus.transport.quic.throttlingEvents, 100) },
    { metric: 'PPS Limit', value: 100 - (ppsData[ppsData.length - 1]?.rate || 0) },
    { metric: 'QoS', value: nodeStatus.transport.qos.pinPeeringStatus ? 100 : 50 }
  ];

  return (
    <div className="space-y-6">
      {/* QUIC Health Dashboard */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card className="bg-gray-900/50 backdrop-blur-sm border-gray-800 p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Network className="w-5 h-5 text-green-400" />
              QUIC Health Dashboard
            </h3>
            <div className="flex items-center gap-3">
              <Badge className={`${nodeStatus.transport.quic.handshakeSuccessRate > 90 ? 'bg-green-500/20 text-green-400 border-green-500' : 'bg-yellow-500/20 text-yellow-400 border-yellow-500'}`}>
                {nodeStatus.transport.quic.handshakeSuccessRate.toFixed(1)}% Success
              </Badge>
              {nodeStatus.transport.quic.throttlingEvents > 0 && (
                <Badge className="bg-red-500/20 text-red-400 border-red-500">
                  <AlertTriangle className="w-3 h-3 mr-1" />
                  Error 15 Active
                </Badge>
              )}
            </div>
          </div>

          <div className="grid grid-cols-3 gap-6">
            {/* Key Metrics */}
            <div className="space-y-4">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-400">Handshake Success Rate</span>
                  <span className={`text-lg font-bold ${nodeStatus.transport.quic.handshakeSuccessRate > 95 ? 'text-green-400' : nodeStatus.transport.quic.handshakeSuccessRate > 85 ? 'text-yellow-400' : 'text-red-400'}`}>
                    {nodeStatus.transport.quic.handshakeSuccessRate.toFixed(1)}%
                  </span>
                </div>
                <Progress value={nodeStatus.transport.quic.handshakeSuccessRate} className="h-2" />
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-400">Concurrent Connections</span>
                  <span className="text-lg font-bold text-blue-400">
                    {nodeStatus.transport.quic.concurrentConnections.toLocaleString()}
                  </span>
                </div>
                <div className="h-12">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={[{ value: nodeStatus.transport.quic.concurrentConnections }]}>
                      <Bar dataKey="value" fill="#3b82f6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-500">Throttling Events</span>
                  <span className={`text-lg font-bold ${nodeStatus.transport.quic.throttlingEvents === 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {nodeStatus.transport.quic.throttlingEvents}
                  </span>
                </div>
                {nodeStatus.transport.quic.throttlingEvents > 0 && (
                  <p className="text-xs text-red-400 mt-1">Error code 15 detected</p>
                )}
              </div>
            </div>

            {/* Open Streams Per Peer */}
            <div>
              <p className="text-sm text-gray-400 mb-3">Open Streams per Peer</p>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={streamData} layout="horizontal">
                    <XAxis type="number" stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                    <YAxis type="category" dataKey="peer" stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 10 }} width={60} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                      labelStyle={{ color: '#9ca3af' }}
                    />
                    <Bar dataKey="streams" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* PPS Rate Limiting */}
            <div>
              <p className="text-sm text-gray-400 mb-3">PPS Rate Limiting</p>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={ppsData}>
                    <defs>
                      <linearGradient id="ppsGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="time" stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                    <YAxis stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                      labelStyle={{ color: '#9ca3af' }}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="rate" 
                      stroke="#ef4444" 
                      fill="url(#ppsGradient)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* QUIC Health Radar */}
          <div className="mt-6 pt-6 border-t border-gray-800">
            <p className="text-sm text-gray-400 mb-3">Overall QUIC Health</p>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={quicHealthData}>
                  <PolarGrid stroke="#374151" />
                  <PolarAngleAxis dataKey="metric" tick={{ fill: '#9ca3af', fontSize: 12 }} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: '#9ca3af', fontSize: 10 }} />
                  <Radar name="Health" dataKey="value" stroke="#10b981" fill="#10b981" fillOpacity={0.6} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Card>
      </motion.div>

      <div className="grid grid-cols-2 gap-6">
        {/* Stake-weighted QoS Peering View */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card className="bg-gray-900/50 backdrop-blur-sm border-gray-800 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <Shield className="w-5 h-5 text-purple-400" />
                Stake-weighted QoS Peering
              </h3>
              {nodeStatus.transport.qos.pinPeeringStatus ? (
                <Badge className="bg-green-500/20 text-green-400 border-green-500">
                  PIN PEERING ON
                </Badge>
              ) : (
                <Badge className="bg-gray-500/20 text-gray-400 border-gray-500">
                  PIN PEERING OFF
                </Badge>
              )}
            </div>

            <div className="space-y-4">
              {/* Whitelisted RPCs Visualization */}
              <div>
                <p className="text-sm text-gray-400 mb-3">Whitelisted RPCs by Stake</p>
                {rpcTreemapData.length > 0 ? (
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <Treemap
                        data={rpcTreemapData}
                        dataKey="size"
                        aspectRatio={4/3}
                        stroke="#374151"
                        fill="#8b5cf6"
                      >
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                          formatter={(value: number) => `${(value / 1e9).toFixed(2)}B SOL`}
                        />
                      </Treemap>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="bg-gray-800/50 rounded-lg p-8 text-center">
                    <AlertTriangle className="w-8 h-8 text-yellow-400 mx-auto mb-2" />
                    <p className="text-sm text-gray-400">No whitelisted RPCs configured</p>
                  </div>
                )}
              </div>

              {/* QoS Metrics */}
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">Virtual Stake</p>
                  <p className="text-lg font-bold text-purple-400">
                    {(nodeStatus.transport.qos.virtualStake / 1e9).toFixed(2)}B SOL
                  </p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">Leader TPU Port</p>
                  <p className="text-lg font-bold text-cyan-400">
                    {nodeStatus.transport.qos.leaderTpuPort}
                  </p>
                </div>
              </div>

              {/* Whitelisted RPCs Table */}
              <div>
                <p className="text-sm text-gray-400 mb-2">Top RPCs</p>
                <div className="space-y-2">
                  {nodeStatus.transport.qos.whitelistedRpcs.slice(0, 3).map((rpc, i) => (
                    <div key={i} className="bg-gray-800/50 rounded-lg p-2 flex justify-between items-center">
                      <span className="text-xs font-mono text-gray-400">
                        {rpc.address.slice(0, 16)}...
                      </span>
                      <Badge className="bg-purple-500/20 text-purple-400 border-purple-500 text-xs">
                        {(rpc.stake / 1e9).toFixed(2)}B
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Card>
        </motion.div>

        {/* Gossip/Repair/Broadcast Metrics */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <Card className="bg-gray-900/50 backdrop-blur-sm border-gray-800 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <Radio className="w-5 h-5 text-blue-400" />
                Gossip/Repair/Broadcast
              </h3>
              <Badge className="bg-blue-500/20 text-blue-400 border-blue-500">
                {nodeStatus.transport.gossip.peers} PEERS
              </Badge>
            </div>

            <div className="space-y-4">
              {/* Peer Connections */}
              <div className="grid grid-cols-3 gap-3">
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Users className="w-4 h-4 text-green-400" />
                    <p className="text-xs text-gray-500">Peers</p>
                  </div>
                  <p className="text-xl font-bold text-white">
                    {nodeStatus.transport.gossip.peers}
                  </p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">Shreds Sent</p>
                  <p className="text-lg font-bold text-blue-400">
                    {(nodeStatus.transport.gossip.shredsSent / 1e6).toFixed(2)}M
                  </p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">Shreds Received</p>
                  <p className="text-lg font-bold text-purple-400">
                    {(nodeStatus.transport.gossip.shredsReceived / 1e6).toFixed(2)}M
                  </p>
                </div>
              </div>

              {/* Retransmits Graph */}
              <div>
                <p className="text-sm text-gray-400 mb-3">Retransmits History</p>
                <div className="h-32">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={retransmitData}>
                      <XAxis dataKey="time" stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                      <YAxis stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                        labelStyle={{ color: '#9ca3af' }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="retransmits" 
                        stroke="#f59e0b" 
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Shard Tree Heatmap */}
              <div>
                <p className="text-sm text-gray-400 mb-3">Network Activity Heatmap</p>
                <div className="grid grid-cols-8 gap-1">
                  {Array.from({ length: 32 }).map((_, i) => {
                    const intensity = Math.random();
                    return (
                      <div
                        key={i}
                        className="h-6 rounded"
                        style={{
                          backgroundColor: `rgba(59, 130, 246, ${intensity})`,
                        }}
                        title={`Shard ${i}: ${(intensity * 100).toFixed(0)}%`}
                      />
                    );
                  })}
                </div>
              </div>

              {/* Health Summary */}
              <div className="bg-gradient-to-r from-blue-900/20 to-purple-900/20 rounded-lg p-4 border border-blue-800/30">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {nodeStatus.transport.gossip.peers > 50 ? (
                      <CheckCircle className="w-5 h-5 text-green-400" />
                    ) : (
                      <AlertTriangle className="w-5 h-5 text-yellow-400" />
                    )}
                    <span className="text-sm text-gray-400">Network Health</span>
                  </div>
                  <span className={`text-sm font-bold ${nodeStatus.transport.gossip.peers > 50 ? 'text-green-400' : 'text-yellow-400'}`}>
                    {nodeStatus.transport.gossip.peers > 50 ? 'HEALTHY' : 'LIMITED PEERS'}
                  </span>
                </div>
              </div>
            </div>
          </Card>
        </motion.div>
      </div>

      {/* Additional Transport Metrics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <Card className="bg-gray-900/50 backdrop-blur-sm border-gray-800 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Activity className="w-5 h-5 text-green-400" />
              Transport Performance Summary
            </h3>
          </div>

          <div className="grid grid-cols-4 gap-4">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Gauge className="w-4 h-4 text-green-400" />
                <p className="text-sm text-gray-400">QUIC Efficiency</p>
              </div>
              <p className="text-2xl font-bold text-white">
                {((nodeStatus.transport.quic.handshakeSuccessRate + (100 - nodeStatus.transport.quic.throttlingEvents)) / 2).toFixed(1)}%
              </p>
              <Progress 
                value={(nodeStatus.transport.quic.handshakeSuccessRate + (100 - nodeStatus.transport.quic.throttlingEvents)) / 2} 
                className="h-1 mt-2"
              />
            </div>

            <div className="bg-gray-800/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <GitBranch className="w-4 h-4 text-purple-400" />
                <p className="text-sm text-gray-400">QoS Coverage</p>
              </div>
              <p className="text-2xl font-bold text-white">
                {nodeStatus.transport.qos.whitelistedRpcs.length}
              </p>
              <p className="text-xs text-gray-500 mt-1">Whitelisted nodes</p>
            </div>

            <div className="bg-gray-800/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Wifi className="w-4 h-4 text-blue-400" />
                <p className="text-sm text-gray-400">Gossip Ratio</p>
              </div>
              <p className="text-2xl font-bold text-white">
                {nodeStatus.transport.gossip.shredsSent > 0 
                  ? (nodeStatus.transport.gossip.shredsReceived / nodeStatus.transport.gossip.shredsSent).toFixed(2)
                  : '0.00'}
              </p>
              <p className="text-xs text-gray-500 mt-1">Received/Sent</p>
            </div>

            <div className="bg-gray-800/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Server className="w-4 h-4 text-cyan-400" />
                <p className="text-sm text-gray-400">Stream Load</p>
              </div>
              <p className="text-2xl font-bold text-white">
                {Object.values(nodeStatus.transport.quic.openStreamsPerPeer).reduce((a, b) => a + b, 0)}
              </p>
              <p className="text-xs text-gray-500 mt-1">Total open streams</p>
            </div>
          </div>
        </Card>
      </motion.div>
    </div>
  );
}