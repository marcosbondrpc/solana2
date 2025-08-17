import { motion } from 'framer-motion';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import { 
  Server, 
  Shield, 
  Activity,
  Zap,
  Wifi,
  Database,
  TrendingUp,
  Clock,
  AlertCircle,
  CheckCircle,
  XCircle
} from 'lucide-react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import type { NodeStatus } from '../../pages/Node';

interface OverviewTabProps {
  nodeStatus: NodeStatus;
}

export default function OverviewTab({ nodeStatus }: OverviewTabProps) {
  const blockProductionRate = nodeStatus.consensus.recentBlockProduction.total > 0
    ? (nodeStatus.consensus.recentBlockProduction.success / nodeStatus.consensus.recentBlockProduction.total) * 100
    : 0;

  const tpsData = nodeStatus.cluster.tps.slice(-60).map((value, index) => ({
    time: index,
    tps: value,
  }));

  const tipFeedData = nodeStatus.jito.tipFeed.slice(-30).map((value, index) => ({
    time: index,
    tip: value,
  }));

  return (
    <div className="grid grid-cols-12 gap-6">
      {/* Node Summary Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="col-span-4"
      >
        <Card className="bg-gray-900/50 backdrop-blur-sm border-gray-800 p-6 h-full">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Server className="w-5 h-5 text-purple-400" />
              Node Summary
            </h3>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              <span className="text-xs text-gray-400">ACTIVE</span>
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-400">Client Version</span>
                <span className="text-sm font-mono text-cyan-400">{nodeStatus.client.version}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-400">Commit</span>
                <span className="text-sm font-mono text-purple-400">{nodeStatus.client.commit}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-400">Cluster</span>
                <Badge className="bg-blue-500/20 text-blue-400 border-blue-500">
                  {nodeStatus.client.cluster}
                </Badge>
              </div>
            </div>

            <div className="h-px bg-gray-800" />

            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-400">Identity</span>
                <span className="text-xs font-mono text-gray-500">
                  {nodeStatus.client.identity.slice(0, 8)}...
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-400">Vote Pubkey</span>
                <span className="text-xs font-mono text-gray-500">
                  {nodeStatus.client.votePubkey.slice(0, 8)}...
                </span>
              </div>
            </div>

            <div className="h-px bg-gray-800" />

            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-400">Ledger Height</span>
                <span className="text-sm font-bold text-white">
                  {nodeStatus.consensus.ledgerHeight.toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-400">Current Slot</span>
                <span className="text-sm font-bold text-yellow-400">
                  {nodeStatus.consensus.slot.toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-400">Distance to Tip</span>
                <span className={`text-sm font-bold ${nodeStatus.consensus.distanceToTip < 10 ? 'text-green-400' : 'text-orange-400'}`}>
                  {nodeStatus.consensus.distanceToTip}
                </span>
              </div>
            </div>

            <div className="h-px bg-gray-800" />

            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-400">Epoch Progress</span>
                <span className="text-sm font-bold text-white">
                  {nodeStatus.consensus.epochProgress.toFixed(1)}%
                </span>
              </div>
              <Progress value={nodeStatus.consensus.epochProgress} className="h-2" />
            </div>

            {nodeStatus.consensus.upcomingLeaderSlots.length > 0 && (
              <>
                <div className="h-px bg-gray-800" />
                <div>
                  <p className="text-sm text-gray-400 mb-2">Upcoming Leader Slots</p>
                  <div className="flex flex-wrap gap-1">
                    {nodeStatus.consensus.upcomingLeaderSlots.slice(0, 5).map((slot, i) => (
                      <Badge key={i} className="bg-purple-500/20 text-purple-400 border-purple-500 text-xs">
                        {slot.toLocaleString()}
                      </Badge>
                    ))}
                    {nodeStatus.consensus.upcomingLeaderSlots.length > 5 && (
                      <Badge className="bg-gray-700/50 text-gray-400 border-gray-600 text-xs">
                        +{nodeStatus.consensus.upcomingLeaderSlots.length - 5} more
                      </Badge>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        </Card>
      </motion.div>

      {/* Consensus Health Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="col-span-4"
      >
        <Card className="bg-gray-900/50 backdrop-blur-sm border-gray-800 p-6 h-full">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Shield className="w-5 h-5 text-green-400" />
              Consensus Health
            </h3>
            {nodeStatus.consensus.delinquent ? (
              <Badge className="bg-red-500/20 text-red-400 border-red-500">
                <XCircle className="w-3 h-3 mr-1" />
                DELINQUENT
              </Badge>
            ) : (
              <Badge className="bg-green-500/20 text-green-400 border-green-500">
                <CheckCircle className="w-3 h-3 mr-1" />
                HEALTHY
              </Badge>
            )}
          </div>

          <div className="space-y-4">
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-gray-400">Vote Credits</span>
                <span className="text-2xl font-bold text-white">
                  {nodeStatus.consensus.voteCredits.toLocaleString()}
                </span>
              </div>
            </div>

            <div className="h-px bg-gray-800" />

            <div>
              <p className="text-sm text-gray-400 mb-3">Block Production</p>
              <div className="grid grid-cols-3 gap-3">
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">Success</p>
                  <p className="text-lg font-bold text-green-400">
                    {nodeStatus.consensus.recentBlockProduction.success}
                  </p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">Skipped</p>
                  <p className="text-lg font-bold text-yellow-400">
                    {nodeStatus.consensus.recentBlockProduction.skipped}
                  </p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">Total</p>
                  <p className="text-lg font-bold text-white">
                    {nodeStatus.consensus.recentBlockProduction.total}
                  </p>
                </div>
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-gray-400">Production Rate</span>
                <span className={`text-sm font-bold ${blockProductionRate > 90 ? 'text-green-400' : blockProductionRate > 75 ? 'text-yellow-400' : 'text-red-400'}`}>
                  {blockProductionRate.toFixed(1)}%
                </span>
              </div>
              <Progress 
                value={blockProductionRate} 
                className="h-2"
              />
            </div>

            <div className="h-px bg-gray-800" />

            <div className="bg-gradient-to-r from-purple-900/20 to-blue-900/20 rounded-lg p-4 border border-purple-800/30">
              <p className="text-xs text-gray-400 mb-2">Performance Metrics</p>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-xs text-gray-500">Skip Rate</p>
                  <p className="text-sm font-bold text-yellow-400">
                    {nodeStatus.consensus.recentBlockProduction.total > 0 
                      ? ((nodeStatus.consensus.recentBlockProduction.skipped / nodeStatus.consensus.recentBlockProduction.total) * 100).toFixed(1)
                      : 0}%
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Efficiency</p>
                  <p className="text-sm font-bold text-cyan-400">
                    {blockProductionRate > 95 ? 'Excellent' : blockProductionRate > 85 ? 'Good' : 'Poor'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </Card>
      </motion.div>

      {/* Cluster Performance Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="col-span-4"
      >
        <Card className="bg-gray-900/50 backdrop-blur-sm border-gray-800 p-6 h-full">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-400" />
              Cluster Performance
            </h3>
            <Badge className="bg-blue-500/20 text-blue-400 border-blue-500">
              LIVE
            </Badge>
          </div>

          <div className="space-y-4">
            <div>
              <p className="text-sm text-gray-400 mb-2">Recent TPS (60s)</p>
              <div className="h-24">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={tpsData}>
                    <defs>
                      <linearGradient id="tpsGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <Area 
                      type="monotone" 
                      dataKey="tps" 
                      stroke="#3b82f6" 
                      fill="url(#tpsGradient)"
                      strokeWidth={2}
                    />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                      labelStyle={{ color: '#9ca3af' }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="flex justify-between items-center mt-2">
                <span className="text-xs text-gray-500">Current</span>
                <span className="text-lg font-bold text-blue-400">
                  {nodeStatus.cluster.tps[nodeStatus.cluster.tps.length - 1]?.toLocaleString() || 0} TPS
                </span>
              </div>
            </div>

            <div className="h-px bg-gray-800" />

            <div>
              <p className="text-sm text-gray-400 mb-3">Confirmation Times</p>
              <div className="grid grid-cols-3 gap-3">
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">p50</p>
                  <p className="text-lg font-bold text-green-400">
                    {nodeStatus.cluster.confirmationTime.p50}ms
                  </p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">p90</p>
                  <p className="text-lg font-bold text-yellow-400">
                    {nodeStatus.cluster.confirmationTime.p90}ms
                  </p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">p99</p>
                  <p className="text-lg font-bold text-orange-400">
                    {nodeStatus.cluster.confirmationTime.p99}ms
                  </p>
                </div>
              </div>
            </div>

            <div className="h-px bg-gray-800" />

            <div>
              <p className="text-sm text-gray-400 mb-3">Network Metrics</p>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-500">Total Stake</span>
                  <span className="text-sm font-mono text-white">
                    {(nodeStatus.cluster.networkMetrics.totalStake / 1e9).toFixed(2)}B SOL
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-500">Active Validators</span>
                  <span className="text-sm font-mono text-purple-400">
                    {nodeStatus.cluster.networkMetrics.activeValidators.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-500">Epoch Time Remaining</span>
                  <span className="text-sm font-mono text-cyan-400">
                    {Math.floor(nodeStatus.cluster.networkMetrics.epochTimeRemaining / 3600)}h {Math.floor((nodeStatus.cluster.networkMetrics.epochTimeRemaining % 3600) / 60)}m
                  </span>
                </div>
              </div>
            </div>
          </div>
        </Card>
      </motion.div>

      {/* Jito Status Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="col-span-6"
      >
        <Card className="bg-gray-900/50 backdrop-blur-sm border-gray-800 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Zap className="w-5 h-5 text-cyan-400" />
              Jito Status
            </h3>
            <div className="flex items-center gap-2">
              {nodeStatus.jito.shredstreamProxy && (
                <Badge className="bg-cyan-500/20 text-cyan-400 border-cyan-500">
                  SHREDSTREAM ACTIVE
                </Badge>
              )}
              <Badge className="bg-green-500/20 text-green-400 border-green-500">
                {nodeStatus.jito.regions.length} REGIONS
              </Badge>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-400 mb-2">BE Regions Connected</p>
                <div className="flex flex-wrap gap-2">
                  {nodeStatus.jito.regions.map((region, i) => (
                    <Badge key={i} className="bg-cyan-500/20 text-cyan-400 border-cyan-500">
                      {region}
                    </Badge>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-xs text-gray-500">Min Tip</p>
                  <p className="text-lg font-bold text-green-400">
                    {nodeStatus.jito.minTip} lamports
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Auction Tick</p>
                  <p className="text-lg font-bold text-purple-400">
                    {nodeStatus.jito.auctionTick}ms
                  </p>
                </div>
              </div>

              <div>
                <p className="text-xs text-gray-500 mb-1">Bundle Auth UUID</p>
                <p className="text-xs font-mono text-gray-400 bg-gray-800/50 p-2 rounded">
                  {nodeStatus.jito.bundleAuthUuid || 'Not configured'}
                </p>
              </div>
            </div>

            <div>
              <p className="text-sm text-gray-400 mb-2">Mini Tip Feed</p>
              <div className="h-32">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={tipFeedData}>
                    <Line 
                      type="monotone" 
                      dataKey="tip" 
                      stroke="#00ffff" 
                      strokeWidth={2}
                      dot={false}
                    />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                      labelStyle={{ color: '#9ca3af' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </Card>
      </motion.div>

      {/* RPC Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="col-span-6"
      >
        <Card className="bg-gray-900/50 backdrop-blur-sm border-gray-800 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Wifi className="w-5 h-5 text-purple-400" />
              RPC Performance
            </h3>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400">Subscriptions</span>
                <Badge className="bg-purple-500/20 text-purple-400 border-purple-500">
                  {nodeStatus.rpc.subscriptions.count}
                </Badge>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400">Lag</span>
                <Badge className={`${nodeStatus.rpc.subscriptions.lag < 100 ? 'bg-green-500/20 text-green-400 border-green-500' : 'bg-yellow-500/20 text-yellow-400 border-yellow-500'}`}>
                  {nodeStatus.rpc.subscriptions.lag}ms
                </Badge>
              </div>
            </div>
          </div>

          <div>
            <p className="text-sm text-gray-400 mb-3">Method Latency Heatmap</p>
            <div className="grid grid-cols-4 gap-2">
              {Object.entries(nodeStatus.rpc.methods).map(([method, latency]) => {
                const avgLatency = (latency.p50 + latency.p99) / 2;
                const intensity = Math.min(avgLatency / 200, 1);
                return (
                  <div
                    key={method}
                    className="bg-gray-800/50 rounded-lg p-3 border border-gray-700"
                    style={{
                      background: `linear-gradient(135deg, rgba(147, 51, 234, ${intensity * 0.3}), rgba(59, 130, 246, ${intensity * 0.3}))`,
                      borderColor: `rgba(147, 51, 234, ${intensity})`,
                    }}
                  >
                    <p className="text-xs text-gray-400 mb-1">{method}</p>
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-green-400">p50: {latency.p50}ms</span>
                      <span className="text-xs text-orange-400">p99: {latency.p99}ms</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </Card>
      </motion.div>
    </div>
  );
}