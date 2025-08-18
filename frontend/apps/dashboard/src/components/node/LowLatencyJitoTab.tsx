import { motion } from 'framer-motion';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { Progress } from '../ui/Progress';
import { 
  Zap, 
  Clock, 
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  XCircle,
  ArrowRight,
  Activity,
  Gauge,
  Send,
  Package
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
  RadialBarChart,
  RadialBar,
  Legend
} from 'recharts';
import type { NodeStatus } from '../../pages/Node';

interface LowLatencyJitoTabProps {
  nodeStatus: NodeStatus;
}

export default function LowLatencyJitoTab({ nodeStatus }: LowLatencyJitoTabProps) {
  // Transform rejection reasons for pie chart
  const rejectionData = Object.entries(nodeStatus.jito.bundleMetrics.rejectionReasons).map(([reason, count]) => ({
    name: reason,
    value: count,
    color: reason === 'simulation_failed' ? '#ef4444' : 
           reason === 'insufficient_tip' ? '#f59e0b' :
           reason === 'bundle_too_large' ? '#8b5cf6' :
           reason === 'duplicate_transaction' ? '#3b82f6' : '#6b7280'
  }));

  // Landing delay distribution data
  const landingDelayData = nodeStatus.jito.bundleMetrics.landingDelay.map((delay, i) => ({
    slot: i,
    delay,
    color: delay < 100 ? '#10b981' : delay < 200 ? '#f59e0b' : '#ef4444'
  }));

  // PPS rate limiting data
  const ppsData = nodeStatus.transport.quic.ppsRateLimiting.map((rate, i) => ({
    time: i,
    rate
  }));

  // Tip percentiles calculation
  const sortedTips = [...nodeStatus.jito.tipFeed].sort((a, b) => a - b);
  const p50Tip = sortedTips[Math.floor(sortedTips.length * 0.5)] || 0;
  const p90Tip = sortedTips[Math.floor(sortedTips.length * 0.9)] || 0;
  const p99Tip = sortedTips[Math.floor(sortedTips.length * 0.99)] || 0;

  // Waterfall timing stages
  const waterfallStages = [
    { name: 'Client', time: 5, color: '#8b5cf6' },
    { name: 'RPC', time: 12, color: '#3b82f6' },
    { name: 'BE Ingress', time: 8, color: '#06b6d4' },
    { name: 'Auction', time: nodeStatus.jito.auctionTick, color: '#10b981' },
    { name: 'Relay', time: 15, color: '#f59e0b' },
    { name: 'Leader Landing', time: 25, color: '#ef4444' },
  ];

  const totalTime = waterfallStages.reduce((sum, stage) => sum + stage.time, 0);

  return (
    <div className="space-y-6">
      {/* End-to-end Timing Waterfall */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card className="bg-gray-900/50 backdrop-blur-sm border-gray-800 p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Clock className="w-5 h-5 text-cyan-400" />
              End-to-End Timing Waterfall
            </h3>
            <Badge className="bg-cyan-500/20 text-cyan-400 border-cyan-500">
              Total: {totalTime}ms
            </Badge>
          </div>

          <div className="space-y-4">
            {/* Visual Waterfall */}
            <div className="relative">
              <div className="flex items-center gap-1">
                {waterfallStages.map((stage, i) => {
                  const width = (stage.time / totalTime) * 100;
                  return (
                    <motion.div
                      key={stage.name}
                      initial={{ width: 0 }}
                      animate={{ width: `${width}%` }}
                      transition={{ delay: i * 0.1, duration: 0.5 }}
                      className="relative group"
                    >
                      <div 
                        className="h-12 rounded flex items-center justify-center overflow-hidden"
                        style={{ backgroundColor: stage.color }}
                      >
                        <span className="text-xs text-white font-medium px-2 truncate">
                          {stage.name}
                        </span>
                      </div>
                      <div className="absolute -bottom-6 left-0 right-0 text-center">
                        <span className="text-xs text-gray-400">{stage.time}ms</span>
                      </div>
                      
                      {/* Connector Arrow */}
                      {i < waterfallStages.length - 1 && (
                        <ArrowRight className="absolute -right-3 top-1/2 -translate-y-1/2 w-6 h-6 text-gray-600 z-10" />
                      )}
                    </motion.div>
                  );
                })}
              </div>
            </div>

            {/* Stage Details */}
            <div className="grid grid-cols-6 gap-3 mt-8">
              {waterfallStages.map((stage) => (
                <div key={stage.name} className="bg-gray-800/50 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: stage.color }} />
                    <p className="text-xs text-gray-400">{stage.name}</p>
                  </div>
                  <p className="text-sm font-bold text-white">{stage.time}ms</p>
                  <p className="text-xs text-gray-500">
                    {((stage.time / totalTime) * 100).toFixed(1)}%
                  </p>
                </div>
              ))}
            </div>
          </div>
        </Card>
      </motion.div>

      <div className="grid grid-cols-2 gap-6">
        {/* Tip Intelligence Panel */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card className="bg-gray-900/50 backdrop-blur-sm border-gray-800 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-green-400" />
                Tip Intelligence
              </h3>
              <Badge className="bg-green-500/20 text-green-400 border-green-500">
                LIVE FEED
              </Badge>
            </div>

            <div className="space-y-4">
              {/* Real-time Tip Feed */}
              <div>
                <p className="text-sm text-gray-400 mb-2">Real-time Tip Feed</p>
                <div className="h-32">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={nodeStatus.jito.tipFeed.map((tip, i) => ({ time: i, tip }))}>
                      <Line 
                        type="monotone" 
                        dataKey="tip" 
                        stroke="#10b981" 
                        strokeWidth={2}
                        dot={false}
                      />
                      <YAxis 
                        stroke="#6b7280" 
                        tick={{ fill: '#9ca3af', fontSize: 10 }}
                        width={40}
                      />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                        labelStyle={{ color: '#9ca3af' }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Tip Percentiles */}
              <div>
                <p className="text-sm text-gray-400 mb-3">Rolling Percentiles</p>
                <div className="grid grid-cols-3 gap-3">
                  <div className="bg-gray-800/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500 mb-1">p50</p>
                    <p className="text-lg font-bold text-cyan-400">{p50Tip}</p>
                    <p className="text-xs text-gray-500">lamports</p>
                  </div>
                  <div className="bg-gray-800/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500 mb-1">p90</p>
                    <p className="text-lg font-bold text-yellow-400">{p90Tip}</p>
                    <p className="text-xs text-gray-500">lamports</p>
                  </div>
                  <div className="bg-gray-800/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500 mb-1">p99</p>
                    <p className="text-lg font-bold text-orange-400">{p99Tip}</p>
                    <p className="text-xs text-gray-500">lamports</p>
                  </div>
                </div>
              </div>

              {/* Tip/Compute Efficiency */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-400">Tip/CU Efficiency Ratio</span>
                  <span className="text-sm font-bold text-purple-400">
                    {(p50Tip / 1400000).toFixed(6)}
                  </span>
                </div>
                <Progress value={Math.min((p50Tip / 1000000) * 100, 100)} className="h-2" />
              </div>

              <div className="bg-gradient-to-r from-green-900/20 to-emerald-900/20 rounded-lg p-3 border border-green-800/30">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">Current Min Tip</span>
                  <span className="text-lg font-bold text-green-400">
                    {nodeStatus.jito.minTip} lamports
                  </span>
                </div>
              </div>
            </div>
          </Card>
        </motion.div>

        {/* Bundle Success Metrics */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <Card className="bg-gray-900/50 backdrop-blur-sm border-gray-800 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <Package className="w-5 h-5 text-purple-400" />
                Bundle Success Metrics
              </h3>
              <Badge className={`${nodeStatus.jito.bundleMetrics.acceptanceRate > 80 ? 'bg-green-500/20 text-green-400 border-green-500' : 'bg-yellow-500/20 text-yellow-400 border-yellow-500'}`}>
                {nodeStatus.jito.bundleMetrics.acceptanceRate.toFixed(1)}% Success
              </Badge>
            </div>

            <div className="space-y-4">
              {/* Acceptance Rate Gauge */}
              <div>
                <p className="text-sm text-gray-400 mb-3">Acceptance Rate</p>
                <div className="h-32">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadialBarChart 
                      cx="50%" 
                      cy="50%" 
                      innerRadius="60%" 
                      outerRadius="90%" 
                      data={[{ value: nodeStatus.jito.bundleMetrics.acceptanceRate, fill: '#8b5cf6' }]}
                      startAngle={180} 
                      endAngle={0}
                    >
                      <RadialBar dataKey="value" cornerRadius={10} />
                      <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" className="text-3xl font-bold fill-white">
                        {nodeStatus.jito.bundleMetrics.acceptanceRate.toFixed(0)}%
                      </text>
                    </RadialBarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Rejection Reasons */}
              <div>
                <p className="text-sm text-gray-400 mb-3">Rejection Reasons</p>
                {rejectionData.length > 0 ? (
                  <div className="h-32">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={rejectionData}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          outerRadius={50}
                          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        >
                          {rejectionData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="bg-gray-800/50 rounded-lg p-4 text-center">
                    <CheckCircle className="w-8 h-8 text-green-400 mx-auto mb-2" />
                    <p className="text-sm text-gray-400">No rejections</p>
                  </div>
                )}
              </div>

              {/* Multi-region Failover */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-400">Multi-region Failover Rate</span>
                  <span className={`text-sm font-bold ${nodeStatus.jito.bundleMetrics.multiRegionFailoverRate > 95 ? 'text-green-400' : 'text-yellow-400'}`}>
                    {nodeStatus.jito.bundleMetrics.multiRegionFailoverRate.toFixed(1)}%
                  </span>
                </div>
                <Progress value={nodeStatus.jito.bundleMetrics.multiRegionFailoverRate} className="h-2" />
              </div>
            </div>
          </Card>
        </motion.div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Landing Delay Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <Card className="bg-gray-900/50 backdrop-blur-sm border-gray-800 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <Activity className="w-5 h-5 text-blue-400" />
                Landing Delay Distribution
              </h3>
              <Badge className="bg-blue-500/20 text-blue-400 border-blue-500">
                Last 100 Bundles
              </Badge>
            </div>

            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={landingDelayData}>
                  <XAxis 
                    dataKey="slot" 
                    stroke="#6b7280" 
                    tick={{ fill: '#9ca3af', fontSize: 10 }}
                  />
                  <YAxis 
                    stroke="#6b7280" 
                    tick={{ fill: '#9ca3af', fontSize: 10 }}
                    label={{ value: 'Delay (ms)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                    labelStyle={{ color: '#9ca3af' }}
                  />
                  <Bar dataKey="delay">
                    {landingDelayData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-3 gap-3 mt-4">
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-2 h-2 rounded-full bg-green-400" />
                  <p className="text-xs text-gray-500">Fast</p>
                </div>
                <p className="text-sm font-bold text-white">&lt;100ms</p>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-2 h-2 rounded-full bg-yellow-400" />
                  <p className="text-xs text-gray-500">Medium</p>
                </div>
                <p className="text-sm font-bold text-white">100-200ms</p>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-2 h-2 rounded-full bg-red-400" />
                  <p className="text-xs text-gray-500">Slow</p>
                </div>
                <p className="text-sm font-bold text-white">&gt;200ms</p>
              </div>
            </div>
          </Card>
        </motion.div>

        {/* ShredStream Latency */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <Card className="bg-gray-900/50 backdrop-blur-sm border-gray-800 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <Send className="w-5 h-5 text-cyan-400" />
                ShredStream Latency
              </h3>
              {nodeStatus.jito.shredstreamProxy ? (
                <Badge className="bg-green-500/20 text-green-400 border-green-500">
                  ACTIVE
                </Badge>
              ) : (
                <Badge className="bg-gray-500/20 text-gray-400 border-gray-500">
                  INACTIVE
                </Badge>
              )}
            </div>

            <div className="space-y-4">
              {/* Packets Per Second */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-400">Packets/sec</span>
                  <span className="text-2xl font-bold text-cyan-400">
                    {nodeStatus.jito.shredstream.packetsPerSec.toLocaleString()}
                  </span>
                </div>
                <div className="h-20">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={[{ name: 'PPS', value: nodeStatus.jito.shredstream.packetsPerSec }]}>
                      <Bar dataKey="value" fill="#06b6d4" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Stream Health Metrics */}
              <div className="grid grid-cols-3 gap-3">
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">Gaps</p>
                  <p className={`text-lg font-bold ${nodeStatus.jito.shredstream.gaps === 0 ? 'text-green-400' : 'text-yellow-400'}`}>
                    {nodeStatus.jito.shredstream.gaps}
                  </p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">Reorders</p>
                  <p className={`text-lg font-bold ${nodeStatus.jito.shredstream.reorders < 10 ? 'text-green-400' : 'text-orange-400'}`}>
                    {nodeStatus.jito.shredstream.reorders}
                  </p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">Decode Time</p>
                  <p className={`text-lg font-bold ${nodeStatus.jito.shredstream.decodeTime < 5 ? 'text-green-400' : 'text-yellow-400'}`}>
                    {nodeStatus.jito.shredstream.decodeTime}ms
                  </p>
                </div>
              </div>

              {/* Visual Health Indicator */}
              <div className="bg-gradient-to-r from-cyan-900/20 to-blue-900/20 rounded-lg p-4 border border-cyan-800/30">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {nodeStatus.jito.shredstream.gaps === 0 && nodeStatus.jito.shredstream.reorders < 10 ? (
                      <CheckCircle className="w-5 h-5 text-green-400" />
                    ) : (
                      <AlertTriangle className="w-5 h-5 text-yellow-400" />
                    )}
                    <span className="text-sm text-gray-400">Stream Health</span>
                  </div>
                  <span className={`text-sm font-bold ${nodeStatus.jito.shredstream.gaps === 0 && nodeStatus.jito.shredstream.reorders < 10 ? 'text-green-400' : 'text-yellow-400'}`}>
                    {nodeStatus.jito.shredstream.gaps === 0 && nodeStatus.jito.shredstream.reorders < 10 ? 'OPTIMAL' : 'DEGRADED'}
                  </span>
                </div>
              </div>
            </div>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}