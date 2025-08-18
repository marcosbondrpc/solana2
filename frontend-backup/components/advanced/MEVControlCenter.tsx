/**
 * MEV Control Center - Ultimate command and control interface
 * Real-time monitoring and control with sub-10ms latency
 */

'use client';

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useControlStore } from '../../stores/controlStore';
import { useBanditStore } from '../../stores/banditStore';
import { apiService } from '../../lib/api-service';
import { mevWebSocket } from '../../lib/enhanced-websocket';
import { toast } from 'sonner';

interface ControlMetric {
  label: string;
  value: string | number;
  status: 'good' | 'warning' | 'critical';
  trend?: 'up' | 'down' | 'stable';
}

export const MEVControlCenter: React.FC = () => {
  const {
    systemStatus,
    controlPlaneLatency,
    emergencyStopActive,
    throttleLevel,
    activeModel,
    deployedModels,
    killSwitches,
    commandHistory,
    policies,
    sendCommand,
    emergencyStop,
    setThrottle,
    swapModel,
    triggerKillSwitch
  } = useControlStore();

  const {
    performance: banditPerformance,
    arms,
    bestArm,
    isConverged
  } = useBanditStore();

  const [selectedModule, setSelectedModule] = useState<string>('overview');
  const [commandInput, setCommandInput] = useState<string>('');
  const [isExecuting, setIsExecuting] = useState(false);
  const [realTimeMetrics, setRealTimeMetrics] = useState<any>({});
  const [wsConnected, setWsConnected] = useState(false);

  // WebSocket connection
  useEffect(() => {
    const handleMessage = (data: any) => {
      if (data.type === 'metrics') {
        setRealTimeMetrics(data.payload);
      }
    };

    const handleOpen = () => setWsConnected(true);
    const handleClose = () => setWsConnected(false);

    mevWebSocket.on('metrics', handleMessage);
    mevWebSocket.on('open', handleOpen);
    mevWebSocket.on('close', handleClose);
    mevWebSocket.connect();

    return () => {
      mevWebSocket.off('metrics', handleMessage);
      mevWebSocket.off('open', handleOpen);
      mevWebSocket.off('close', handleClose);
    };
  }, []);

  // Calculate system metrics
  const systemMetrics = useMemo<ControlMetric[]>(() => {
    const latencyStatus = controlPlaneLatency < 10 ? 'good' : 
                         controlPlaneLatency < 20 ? 'warning' : 'critical';
    
    const landRate = realTimeMetrics.bundleLandRate || 0;
    const landRateStatus = landRate >= 65 ? 'good' : 
                          landRate >= 55 ? 'warning' : 'critical';

    return [
      {
        label: 'System Status',
        value: systemStatus.toUpperCase(),
        status: systemStatus === 'operational' ? 'good' : 
                systemStatus === 'degraded' ? 'warning' : 'critical'
      },
      {
        label: 'Control Latency',
        value: `${controlPlaneLatency.toFixed(1)}ms`,
        status: latencyStatus,
        trend: controlPlaneLatency < 10 ? 'down' : 'up'
      },
      {
        label: 'Bundle Land Rate',
        value: `${landRate.toFixed(1)}%`,
        status: landRateStatus,
        trend: landRate > 65 ? 'up' : 'down'
      },
      {
        label: 'Throttle Level',
        value: `${throttleLevel}%`,
        status: throttleLevel === 100 ? 'good' : 
                throttleLevel >= 50 ? 'warning' : 'critical'
      },
      {
        label: 'Active Model',
        value: activeModel || 'None',
        status: activeModel ? 'good' : 'warning'
      },
      {
        label: 'Bandit Convergence',
        value: isConverged ? 'Converged' : 'Exploring',
        status: isConverged ? 'good' : 'warning',
        trend: banditPerformance.convergence_metric > 0.8 ? 'up' : 'stable'
      }
    ];
  }, [systemStatus, controlPlaneLatency, throttleLevel, activeModel, isConverged, 
      banditPerformance.convergence_metric, realTimeMetrics]);

  // Execute command
  const executeCommand = useCallback(async () => {
    if (!commandInput.trim()) return;

    setIsExecuting(true);
    try {
      const parts = commandInput.split(' ');
      const module = parts[0];
      const action = parts[1];
      const params = parts.slice(2).join(' ');

      await sendCommand({
        module,
        action,
        params: params ? JSON.parse(params) : {},
        priority: 1
      });

      toast.success('Command executed successfully');
      setCommandInput('');
    } catch (error: any) {
      toast.error(`Command failed: ${error.message}`);
    } finally {
      setIsExecuting(false);
    }
  }, [commandInput, sendCommand]);

  // Quick actions
  const handleEmergencyStop = useCallback(async () => {
    if (confirm('Are you sure you want to trigger emergency stop?')) {
      try {
        await emergencyStop();
        toast.success('Emergency stop activated');
      } catch (error: any) {
        toast.error(`Failed to activate emergency stop: ${error.message}`);
      }
    }
  }, [emergencyStop]);

  const handleThrottleChange = useCallback(async (value: number) => {
    try {
      await setThrottle(value);
      toast.success(`Throttle set to ${value}%`);
    } catch (error: any) {
      toast.error(`Failed to set throttle: ${error.message}`);
    }
  }, [setThrottle]);

  const handleModelSwap = useCallback(async (modelId: string) => {
    const model = deployedModels.get(modelId);
    if (!model) return;

    try {
      await swapModel({
        model_id: modelId,
        model_path: model.path,
        model_type: model.type as any,
        version: model.version,
        metadata: model.metadata
      });
      toast.success(`Model swapped to ${modelId}`);
    } catch (error: any) {
      toast.error(`Failed to swap model: ${error.message}`);
    }
  }, [deployedModels, swapModel]);

  return (
    <div className="w-full bg-black/90 backdrop-blur-xl rounded-lg border border-cyan-500/20 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <h2 className="text-2xl font-bold text-cyan-400">MEV Control Center</h2>
          <div className={`px-3 py-1 rounded-full text-xs font-semibold ${
            wsConnected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
          }`}>
            {wsConnected ? 'CONNECTED' : 'DISCONNECTED'}
          </div>
          {emergencyStopActive && (
            <div className="px-3 py-1 rounded-full bg-red-500/20 text-red-400 text-xs font-semibold animate-pulse">
              EMERGENCY STOP ACTIVE
            </div>
          )}
        </div>

        {/* Quick Actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => handleThrottleChange(100)}
            className="px-4 py-2 bg-green-500/20 text-green-400 rounded hover:bg-green-500/30 transition-colors"
            disabled={throttleLevel === 100}
          >
            Full Speed
          </button>
          <button
            onClick={() => handleThrottleChange(50)}
            className="px-4 py-2 bg-yellow-500/20 text-yellow-400 rounded hover:bg-yellow-500/30 transition-colors"
          >
            50% Throttle
          </button>
          <button
            onClick={() => handleThrottleChange(10)}
            className="px-4 py-2 bg-orange-500/20 text-orange-400 rounded hover:bg-orange-500/30 transition-colors"
          >
            10% Throttle
          </button>
          <button
            onClick={handleEmergencyStop}
            className="px-4 py-2 bg-red-500/20 text-red-400 rounded hover:bg-red-500/30 transition-colors font-semibold"
          >
            EMERGENCY STOP
          </button>
        </div>
      </div>

      {/* System Metrics Grid */}
      <div className="grid grid-cols-6 gap-4 mb-6">
        {systemMetrics.map((metric, idx) => (
          <motion.div
            key={metric.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.05 }}
            className={`p-4 rounded-lg border ${
              metric.status === 'good' ? 'bg-green-500/10 border-green-500/30' :
              metric.status === 'warning' ? 'bg-yellow-500/10 border-yellow-500/30' :
              'bg-red-500/10 border-red-500/30'
            }`}
          >
            <div className="text-xs text-gray-400 mb-1">{metric.label}</div>
            <div className="flex items-center justify-between">
              <div className={`text-lg font-bold ${
                metric.status === 'good' ? 'text-green-400' :
                metric.status === 'warning' ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                {metric.value}
              </div>
              {metric.trend && (
                <div className={`text-xs ${
                  metric.trend === 'up' ? 'text-green-400' :
                  metric.trend === 'down' ? 'text-red-400' :
                  'text-gray-400'
                }`}>
                  {metric.trend === 'up' ? '↑' : metric.trend === 'down' ? '↓' : '→'}
                </div>
              )}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Control Tabs */}
      <div className="flex gap-2 mb-4 border-b border-cyan-500/20">
        {['overview', 'commands', 'models', 'policies', 'killswitches', 'bandit'].map(tab => (
          <button
            key={tab}
            onClick={() => setSelectedModule(tab)}
            className={`px-4 py-2 capitalize transition-colors ${
              selectedModule === tab
                ? 'text-cyan-400 border-b-2 border-cyan-400'
                : 'text-gray-400 hover:text-cyan-300'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={selectedModule}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.2 }}
          className="min-h-[400px]"
        >
          {selectedModule === 'overview' && (
            <div className="space-y-4">
              {/* Real-time Metrics */}
              <div className="grid grid-cols-4 gap-4">
                <div className="p-4 bg-gray-900/50 rounded-lg">
                  <div className="text-xs text-gray-400 mb-2">Decision Latency P50</div>
                  <div className="text-2xl font-bold text-cyan-400">
                    {realTimeMetrics.decisionLatencyP50 || 0}ms
                  </div>
                </div>
                <div className="p-4 bg-gray-900/50 rounded-lg">
                  <div className="text-xs text-gray-400 mb-2">Decision Latency P99</div>
                  <div className="text-2xl font-bold text-cyan-400">
                    {realTimeMetrics.decisionLatencyP99 || 0}ms
                  </div>
                </div>
                <div className="p-4 bg-gray-900/50 rounded-lg">
                  <div className="text-xs text-gray-400 mb-2">Ingestion Rate</div>
                  <div className="text-2xl font-bold text-cyan-400">
                    {((realTimeMetrics.ingestionRate || 0) / 1000).toFixed(1)}k/s
                  </div>
                </div>
                <div className="p-4 bg-gray-900/50 rounded-lg">
                  <div className="text-xs text-gray-400 mb-2">Active Opportunities</div>
                  <div className="text-2xl font-bold text-cyan-400">
                    {realTimeMetrics.activeOpportunities || 0}
                  </div>
                </div>
              </div>

              {/* Kill Switches Status */}
              {killSwitches.size > 0 && (
                <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
                  <h3 className="text-lg font-semibold text-red-400 mb-2">Active Kill Switches</h3>
                  <div className="space-y-2">
                    {Array.from(killSwitches.entries()).map(([target, state]) => (
                      <div key={target} className="flex items-center justify-between">
                        <span className="text-sm text-gray-300">
                          {target}: {state.reason}
                        </span>
                        <span className="text-xs text-gray-400">
                          Expires in {Math.round((state.expires_at! - Date.now()) / 1000)}s
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {selectedModule === 'commands' && (
            <div className="space-y-4">
              {/* Command Input */}
              <div className="flex gap-2">
                <input
                  type="text"
                  value={commandInput}
                  onChange={(e) => setCommandInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && executeCommand()}
                  placeholder="Enter command (e.g., throttle set {percent: 50})"
                  className="flex-1 px-4 py-2 bg-gray-900/50 border border-cyan-500/30 rounded text-cyan-300 placeholder-gray-500 focus:outline-none focus:border-cyan-400"
                  disabled={isExecuting}
                />
                <button
                  onClick={executeCommand}
                  disabled={isExecuting || !commandInput.trim()}
                  className="px-6 py-2 bg-cyan-500/20 text-cyan-400 rounded hover:bg-cyan-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isExecuting ? 'Executing...' : 'Execute'}
                </button>
              </div>

              {/* Command History */}
              <div className="space-y-2">
                <h3 className="text-lg font-semibold text-cyan-400">Command History</h3>
                <div className="max-h-96 overflow-y-auto space-y-2">
                  {commandHistory.slice(0, 20).map((cmd) => (
                    <div
                      key={cmd.id}
                      className={`p-3 rounded-lg border ${
                        cmd.status === 'executed' ? 'bg-green-500/10 border-green-500/30' :
                        cmd.status === 'failed' ? 'bg-red-500/10 border-red-500/30' :
                        'bg-yellow-500/10 border-yellow-500/30'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-mono text-cyan-300">
                          {cmd.command.module}.{cmd.command.action}
                        </span>
                        <span className={`text-xs ${
                          cmd.status === 'executed' ? 'text-green-400' :
                          cmd.status === 'failed' ? 'text-red-400' :
                          'text-yellow-400'
                        }`}>
                          {cmd.status.toUpperCase()}
                        </span>
                      </div>
                      {cmd.command.params && Object.keys(cmd.command.params).length > 0 && (
                        <div className="text-xs text-gray-400 font-mono">
                          {JSON.stringify(cmd.command.params)}
                        </div>
                      )}
                      {cmd.error && (
                        <div className="text-xs text-red-400 mt-1">
                          Error: {cmd.error}
                        </div>
                      )}
                      <div className="text-xs text-gray-500 mt-1">
                        {new Date(cmd.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {selectedModule === 'models' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-cyan-400">Deployed Models</h3>
              <div className="grid grid-cols-2 gap-4">
                {Array.from(deployedModels.entries()).map(([id, model]) => (
                  <div
                    key={id}
                    className={`p-4 rounded-lg border cursor-pointer transition-all ${
                      id === activeModel
                        ? 'bg-cyan-500/20 border-cyan-500/50'
                        : 'bg-gray-900/50 border-gray-700 hover:border-cyan-500/30'
                    }`}
                    onClick={() => handleModelSwap(id)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-cyan-300">{id}</span>
                      <span className={`text-xs px-2 py-1 rounded ${
                        model.status === 'ready' ? 'bg-green-500/20 text-green-400' :
                        model.status === 'loading' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        {model.status.toUpperCase()}
                      </span>
                    </div>
                    <div className="space-y-1 text-xs text-gray-400">
                      <div>Type: {model.type}</div>
                      <div>Version: {model.version}</div>
                      <div>Inference P50: {model.performance.inference_p50_us}μs</div>
                      <div>Inference P99: {model.performance.inference_p99_us}μs</div>
                      <div>Throughput: {model.performance.throughput_rps} req/s</div>
                    </div>
                    {id === activeModel && (
                      <div className="mt-2 text-xs text-cyan-400 font-semibold">
                        ACTIVE MODEL
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {selectedModule === 'policies' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-cyan-400">Active Policies</h3>
              <div className="space-y-2">
                {Array.from(policies.entries()).map(([id, policy]) => (
                  <div
                    key={id}
                    className={`p-4 rounded-lg border ${
                      policy.enabled
                        ? 'bg-green-500/10 border-green-500/30'
                        : 'bg-gray-900/50 border-gray-700'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-cyan-300">{id}</span>
                      <span className={`text-xs px-2 py-1 rounded ${
                        policy.enabled
                          ? 'bg-green-500/20 text-green-400'
                          : 'bg-gray-500/20 text-gray-400'
                      }`}>
                        {policy.enabled ? 'ENABLED' : 'DISABLED'}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-xs text-gray-400">
                      <div>
                        <div className="font-semibold text-gray-300 mb-1">Thresholds</div>
                        {Object.entries(policy.thresholds).map(([key, value]) => (
                          <div key={key}>{key}: {value}</div>
                        ))}
                      </div>
                      <div>
                        <div className="font-semibold text-gray-300 mb-1">Rules</div>
                        {Object.entries(policy.rules).map(([key, value]) => (
                          <div key={key}>{key}: {value}</div>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {selectedModule === 'bandit' && (
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-4">
                <div className="p-4 bg-gray-900/50 rounded-lg">
                  <div className="text-xs text-gray-400 mb-2">Total Pulls</div>
                  <div className="text-2xl font-bold text-cyan-400">
                    {banditPerformance.total_pulls}
                  </div>
                </div>
                <div className="p-4 bg-gray-900/50 rounded-lg">
                  <div className="text-xs text-gray-400 mb-2">Convergence</div>
                  <div className="text-2xl font-bold text-cyan-400">
                    {(banditPerformance.convergence_metric * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-4 bg-gray-900/50 rounded-lg">
                  <div className="text-xs text-gray-400 mb-2">Regret</div>
                  <div className="text-2xl font-bold text-cyan-400">
                    {banditPerformance.regret.toFixed(2)}
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                <h3 className="text-lg font-semibold text-cyan-400">Bandit Arms</h3>
                <div className="space-y-2">
                  {Array.from(arms.entries()).slice(0, 10).map(([id, arm]) => (
                    <div
                      key={id}
                      className={`p-3 rounded-lg border ${
                        id === bestArm
                          ? 'bg-cyan-500/20 border-cyan-500/50'
                          : 'bg-gray-900/50 border-gray-700'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-semibold text-cyan-300">
                          {arm.route}
                        </span>
                        <div className="flex items-center gap-4 text-xs text-gray-400">
                          <span>Pulls: {arm.pulls}</span>
                          <span>Rewards: {arm.rewards}</span>
                          <span>EV: {(arm.expected_value * 100).toFixed(1)}%</span>
                          <span>UCB: {arm.ucb_score.toFixed(3)}</span>
                        </div>
                      </div>
                      {id === bestArm && (
                        <div className="mt-1 text-xs text-cyan-400 font-semibold">
                          BEST ARM
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
};