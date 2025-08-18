'use client';

import { useEffect, useState } from 'react';
import { useSnapshot } from 'valtio';
import { mevStore } from '../../stores/mevStore';

interface SystemMetric {
  name: string;
  value: number;
  unit: string;
  status: 'good' | 'warning' | 'critical';
  target?: number;
}

export default function SystemStatus() {
  const store = useSnapshot(mevStore);
  const [metrics, setMetrics] = useState<SystemMetric[]>([]);
  const [uptime, setUptime] = useState(0);
  const [dnaVerified, setDnaVerified] = useState(0);

  useEffect(() => {
    // Update metrics from store
    const health = store.health;
    const newMetrics: SystemMetric[] = [
      {
        name: 'Ingestion',
        value: health.ingestionRate / 1000,
        unit: 'k/s',
        status: health.ingestionRate >= 200000 ? 'good' : health.ingestionRate >= 150000 ? 'warning' : 'critical',
        target: 200,
      },
      {
        name: 'P50 Latency',
        value: health.decisionLatencyP50,
        unit: 'ms',
        status: health.decisionLatencyP50 <= 8 ? 'good' : health.decisionLatencyP50 <= 12 ? 'warning' : 'critical',
        target: 8,
      },
      {
        name: 'P99 Latency',
        value: health.decisionLatencyP99,
        unit: 'ms',
        status: health.decisionLatencyP99 <= 20 ? 'good' : health.decisionLatencyP99 <= 30 ? 'warning' : 'critical',
        target: 20,
      },
      {
        name: 'Land Rate',
        value: health.bundleLandRate,
        unit: '%',
        status: health.bundleLandRate >= 65 ? 'good' : health.bundleLandRate >= 55 ? 'warning' : 'critical',
        target: 65,
      },
      {
        name: 'Model Inference',
        value: health.modelInferenceTime * 1000,
        unit: 'μs',
        status: health.modelInferenceTime <= 0.1 ? 'good' : health.modelInferenceTime <= 0.2 ? 'warning' : 'critical',
        target: 100,
      },
      {
        name: 'Active Opps',
        value: store.activeOpportunities,
        unit: '',
        status: store.activeOpportunities > 0 ? 'good' : 'warning',
      },
    ];

    setMetrics(newMetrics);
  }, [store.health, store.activeOpportunities]);

  // Update uptime
  useEffect(() => {
    const startTime = Date.now();
    const interval = setInterval(() => {
      setUptime(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Update DNA verification count
  useEffect(() => {
    const verified = Array.from(store.dnaFingerprints.values()).filter(dna => dna.verified).length;
    const total = store.dnaFingerprints.size;
    setDnaVerified(total > 0 ? (verified / total) * 100 : 0);
  }, [store.dnaFingerprints]);

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const getStatusColor = (status: SystemMetric['status']) => {
    switch (status) {
      case 'good': return 'bg-green-500';
      case 'warning': return 'bg-yellow-500';
      case 'critical': return 'bg-red-500';
    }
  };

  const getStatusTextColor = (status: SystemMetric['status']) => {
    switch (status) {
      case 'good': return 'text-green-400';
      case 'warning': return 'text-yellow-400';
      case 'critical': return 'text-red-400';
    }
  };

  return (
    <div className="bg-zinc-950 border border-zinc-900 rounded-lg p-3">
      <div className="flex items-center justify-between">
        {/* System Metrics */}
        <div className="flex items-center gap-6">
          {metrics.map(metric => (
            <div key={metric.name} className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${getStatusColor(metric.status)}`} />
              <div className="flex flex-col">
                <span className="text-[10px] text-zinc-600">{metric.name}</span>
                <div className="flex items-center gap-1">
                  <span className={`text-sm font-mono ${getStatusTextColor(metric.status)}`}>
                    {metric.value.toFixed(metric.unit === '%' ? 1 : 0)}{metric.unit}
                  </span>
                  {metric.target && (
                    <span className="text-[10px] text-zinc-600">
                      /{metric.target}{metric.unit}
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Right Side Indicators */}
        <div className="flex items-center gap-6">
          {/* DNA Tracking */}
          <div className="flex items-center gap-2">
            <div className="flex flex-col">
              <span className="text-[10px] text-zinc-600">DNA Verified</span>
              <div className="flex items-center gap-1">
                <div className="w-24 h-1 bg-zinc-900 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-purple-500 transition-all duration-300"
                    style={{ width: `${dnaVerified}%` }}
                  />
                </div>
                <span className="text-[10px] font-mono text-purple-400">
                  {dnaVerified.toFixed(0)}%
                </span>
              </div>
            </div>
          </div>

          {/* Connection Status */}
          <div className="flex items-center gap-2">
            <div className="flex flex-col">
              <span className="text-[10px] text-zinc-600">Connections</span>
              <div className="flex items-center gap-1">
                <div className={`w-2 h-2 rounded-full ${
                  store.wsStatus === 'connected' ? 'bg-green-500' : 
                  store.wsStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' : 
                  'bg-red-500'
                }`} />
                <span className="text-[10px] text-zinc-500">WS</span>
                <div className={`w-2 h-2 rounded-full ${
                  store.clickhouseStatus === 'connected' ? 'bg-green-500' : 
                  store.clickhouseStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' : 
                  'bg-red-500'
                }`} />
                <span className="text-[10px] text-zinc-500">CH</span>
                <div className={`w-2 h-2 rounded-full ${
                  store.backendStatus === 'connected' ? 'bg-green-500' : 
                  store.backendStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' : 
                  'bg-red-500'
                }`} />
                <span className="text-[10px] text-zinc-500">API</span>
              </div>
            </div>
          </div>

          {/* Uptime */}
          <div className="flex flex-col">
            <span className="text-[10px] text-zinc-600">Uptime</span>
            <span className="text-sm font-mono text-cyan-400">{formatUptime(uptime)}</span>
          </div>

          {/* System Health */}
          <div className="flex items-center gap-2">
            <div className={`px-2 py-1 rounded text-xs font-bold ${
              store.health.isHealthy 
                ? 'bg-green-900/30 text-green-400 border border-green-800' 
                : 'bg-red-900/30 text-red-400 border border-red-800 animate-pulse'
            }`}>
              {store.health.isHealthy ? 'HEALTHY' : 'DEGRADED'}
            </div>
          </div>
        </div>
      </div>

      {/* Alerts Bar */}
      {store.health.alerts.length > 0 && (
        <div className="mt-2 pt-2 border-t border-zinc-800">
          <div className="flex items-center gap-2 overflow-x-auto">
            {store.health.alerts.slice(0, 3).map((alert, idx) => (
              <div 
                key={idx}
                className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] whitespace-nowrap ${
                  alert.level === 'error' ? 'bg-red-900/20 text-red-400' :
                  alert.level === 'warning' ? 'bg-yellow-900/20 text-yellow-400' :
                  'bg-blue-900/20 text-blue-400'
                }`}
              >
                <span>{alert.level === 'error' ? '⚠️' : alert.level === 'warning' ? '⚡' : 'ℹ️'}</span>
                <span>{alert.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}