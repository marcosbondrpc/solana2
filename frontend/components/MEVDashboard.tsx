/**
 * Legendary MEV Dashboard
 * Ultra-high-performance dashboard for MEV operations
 */

import React, { useMemo, useCallback } from 'react';
import { useMEVDashboard } from '../hooks/useMEV';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Activity, AlertTriangle, TrendingUp, Zap, 
  DollarSign, Target, Shield, BarChart3,
  PlayCircle, StopCircle, AlertCircle
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import numeral from 'numeral';

const MEVDashboard: React.FC = () => {
  const {
    opportunities,
    isOpportunitiesConnected,
    executeOpportunity,
    isExecuting,
    metrics,
    isMetricsConnected,
    banditStats,
    resetBandit,
    riskStatus,
    emergencyStop,
    resumeOperations,
    setThrottle,
    mevStats,
    executions,
    isExecutionsConnected,
    isLoading,
  } = useMEVDashboard();

  // Sort opportunities by profit
  const sortedOpportunities = useMemo(() => {
    return [...(opportunities || [])].sort((a, b) => b.expectedProfit - a.expectedProfit);
  }, [opportunities]);

  // Calculate total potential profit
  const totalPotentialProfit = useMemo(() => {
    return opportunities?.reduce((sum, opp) => sum + opp.expectedProfit, 0) || 0;
  }, [opportunities]);

  const handleExecute = useCallback((opportunityId: string) => {
    executeOpportunity({ opportunityId, priority: 1 });
  }, [executeOpportunity]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-black">
        <div className="text-green-400 text-2xl animate-pulse">Loading MEV System...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white p-4">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-4xl font-bold text-green-400 mb-2">
          🚀 LEGENDARY MEV DASHBOARD
        </h1>
        <p className="text-gray-400">Built for billions. Optimized for microseconds.</p>
      </div>

      {/* Connection Status */}
      <div className="flex gap-4 mb-6">
        <StatusBadge 
          label="Opportunities" 
          connected={isOpportunitiesConnected} 
        />
        <StatusBadge 
          label="Metrics" 
          connected={isMetricsConnected} 
        />
        <StatusBadge 
          label="Executions" 
          connected={isExecutionsConnected} 
        />
      </div>

      {/* Risk Controls */}
      <div className="bg-gray-900 rounded-lg p-4 mb-6 border border-gray-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Shield className="text-yellow-400" size={24} />
            <span className="text-lg font-semibold">Risk Management</span>
            {riskStatus && (
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-400">
                  Exposure: ${numeral(riskStatus.currentExposure).format('0,0')} / ${numeral(riskStatus.maxExposure).format('0,0')}
                </span>
                {riskStatus.killSwitchEnabled && (
                  <span className="text-red-500 font-bold animate-pulse">KILL SWITCH ACTIVE</span>
                )}
              </div>
            )}
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setThrottle(50)}
              className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded transition-colors"
            >
              Throttle 50%
            </button>
            {riskStatus?.killSwitchEnabled ? (
              <button
                onClick={resumeOperations}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded transition-colors flex items-center gap-2"
              >
                <PlayCircle size={18} />
                Resume
              </button>
            ) : (
              <button
                onClick={emergencyStop}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded transition-colors flex items-center gap-2"
              >
                <StopCircle size={18} />
                Emergency Stop
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
        <MetricCard
          title="Latency P50"
          value={metrics?.latencyP50 ? `${metrics.latencyP50.toFixed(1)}ms` : '-'}
          icon={<Zap className="text-yellow-400" size={20} />}
          target="≤8ms"
          good={metrics?.latencyP50 ? metrics.latencyP50 <= 8 : false}
        />
        <MetricCard
          title="Latency P99"
          value={metrics?.latencyP99 ? `${metrics.latencyP99.toFixed(1)}ms` : '-'}
          icon={<Zap className="text-yellow-400" size={20} />}
          target="≤20ms"
          good={metrics?.latencyP99 ? metrics.latencyP99 <= 20 : false}
        />
        <MetricCard
          title="Land Rate"
          value={metrics?.bundleLandRate ? `${(metrics.bundleLandRate * 100).toFixed(1)}%` : '-'}
          icon={<Target className="text-blue-400" size={20} />}
          target="≥65%"
          good={metrics?.bundleLandRate ? metrics.bundleLandRate >= 0.65 : false}
        />
        <MetricCard
          title="Ingestion"
          value={metrics?.ingestionRate ? `${numeral(metrics.ingestionRate).format('0,0')}/s` : '-'}
          icon={<Activity className="text-green-400" size={20} />}
          target="≥200k/s"
          good={metrics?.ingestionRate ? metrics.ingestionRate >= 200000 : false}
        />
        <MetricCard
          title="Total Profit"
          value={mevStats?.totalProfit ? `$${numeral(mevStats.totalProfit).format('0,0.00')}` : '-'}
          icon={<DollarSign className="text-green-400" size={20} />}
        />
        <MetricCard
          title="Success Rate"
          value={mevStats?.successRate ? `${(mevStats.successRate * 100).toFixed(1)}%` : '-'}
          icon={<BarChart3 className="text-purple-400" size={20} />}
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Opportunities Feed */}
        <div className="lg:col-span-2">
          <div className="bg-gray-900 rounded-lg border border-gray-800 h-[600px] flex flex-col">
            <div className="p-4 border-b border-gray-800 flex items-center justify-between">
              <h2 className="text-xl font-semibold flex items-center gap-2">
                <TrendingUp className="text-green-400" size={20} />
                MEV Opportunities
              </h2>
              <div className="text-sm text-gray-400">
                Potential: ${numeral(totalPotentialProfit).format('0,0.00')}
              </div>
            </div>
            <div className="flex-1 overflow-auto p-4">
              <AnimatePresence>
                {sortedOpportunities.map((opp) => (
                  <motion.div
                    key={opp.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    className="mb-3 p-3 bg-gray-800 rounded-lg hover:bg-gray-700 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className={`px-2 py-1 rounded text-xs font-semibold ${
                            opp.type === 'arbitrage' ? 'bg-blue-600' :
                            opp.type === 'sandwich' ? 'bg-purple-600' :
                            opp.type === 'liquidation' ? 'bg-red-600' :
                            'bg-gray-600'
                          }`}>
                            {opp.type.toUpperCase()}
                          </span>
                          <span className="text-sm text-gray-400">
                            {opp.dexes.join(' → ')}
                          </span>
                        </div>
                        <div className="flex items-center gap-4 text-sm">
                          <span className="text-green-400 font-semibold">
                            +${numeral(opp.expectedProfit).format('0,0.00')}
                          </span>
                          <span className="text-gray-400">
                            {(opp.confidence * 100).toFixed(0)}% confidence
                          </span>
                          <span className="text-gray-500">
                            Gas: {opp.gasEstimate}
                          </span>
                        </div>
                      </div>
                      <button
                        onClick={() => handleExecute(opp.id)}
                        disabled={isExecuting || riskStatus?.killSwitchEnabled}
                        className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded transition-colors"
                      >
                        Execute
                      </button>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="space-y-6">
          {/* Thompson Sampling Stats */}
          <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Thompson Sampling</h3>
              <button
                onClick={resetBandit}
                className="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
              >
                Reset
              </button>
            </div>
            {banditStats?.arms.map((arm) => (
              <div key={arm.name} className="mb-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">{arm.name}</span>
                  <span className="text-green-400">
                    EV: {arm.ev.toFixed(3)}
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2 mt-1">
                  <div
                    className="bg-green-500 h-2 rounded-full transition-all"
                    style={{ width: `${(arm.ev * 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Recent Executions */}
          <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
            <h3 className="text-lg font-semibold mb-4">Recent Executions</h3>
            <div className="space-y-2 max-h-64 overflow-auto">
              {executions.slice(0, 10).map((exec, idx) => (
                <div
                  key={`${exec.bundleId}-${idx}`}
                  className={`p-2 rounded text-sm ${
                    exec.success ? 'bg-green-900/20' : 'bg-red-900/20'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className={exec.success ? 'text-green-400' : 'text-red-400'}>
                      {exec.success ? '✓' : '✗'} {exec.bundleId.slice(0, 8)}...
                    </span>
                    {exec.actualProfit && (
                      <span className="text-green-400">
                        +${numeral(exec.actualProfit).format('0,0.00')}
                      </span>
                    )}
                  </div>
                  {exec.landedSlot && (
                    <div className="text-xs text-gray-500">
                      Slot: {exec.landedSlot}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Helper Components
const StatusBadge: React.FC<{ label: string; connected: boolean }> = ({ label, connected }) => (
  <div className={`px-3 py-1 rounded-full text-sm flex items-center gap-2 ${
    connected ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'
  }`}>
    <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400' : 'bg-red-400'} animate-pulse`} />
    {label}: {connected ? 'Connected' : 'Disconnected'}
  </div>
);

const MetricCard: React.FC<{
  title: string;
  value: string;
  icon?: React.ReactNode;
  target?: string;
  good?: boolean;
}> = ({ title, value, icon, target, good }) => (
  <div className="bg-gray-900 rounded-lg border border-gray-800 p-3">
    <div className="flex items-center justify-between mb-1">
      {icon}
      {target && (
        <span className={`text-xs ${good ? 'text-green-400' : 'text-yellow-400'}`}>
          {target}
        </span>
      )}
    </div>
    <div className="text-xs text-gray-400 mb-1">{title}</div>
    <div className={`text-lg font-bold ${good !== undefined ? (good ? 'text-green-400' : 'text-yellow-400') : 'text-white'}`}>
      {value}
    </div>
  </div>
);

export default MEVDashboard;