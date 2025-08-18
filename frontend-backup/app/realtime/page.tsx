'use client';

import { useSnapshot } from 'valtio';
import { mevStore, mevActions } from '../../stores/mevStore';
import { useEffect, useState, useCallback } from 'react';
import { clsx } from 'clsx';
import numeral from 'numeral';
import { motion } from 'framer-motion';
import { useHotkeys } from 'react-hotkeys-hook';
import dynamic from 'next/dynamic';
import { ClickHouseQueries } from '../../lib/clickhouse';

// Dynamic imports for performance
const TransactionFeed = dynamic(() => import('../../components/TransactionFeed'), { ssr: false });

function MetricTile({ 
  label, 
  value, 
  unit = '', 
  trend, 
  status 
}: {
  label: string;
  value: string | number;
  unit?: string;
  trend?: 'up' | 'down' | 'stable';
  status?: 'good' | 'warning' | 'critical';
}) {
  return (
    <motion.div 
      className="glass rounded-lg p-3"
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.2 }}
    >
      <div className="text-xs text-zinc-500 mb-1">{label}</div>
      <div className="flex items-baseline gap-1">
        <span className={clsx(
          "text-xl font-bold",
          status === 'good' && "text-green-400",
          status === 'warning' && "text-yellow-400",
          status === 'critical' && "text-red-400",
          !status && "text-white"
        )}>
          {value}
        </span>
        {unit && <span className="text-sm text-zinc-500">{unit}</span>}
      </div>
      {trend && (
        <div className={clsx(
          "text-xs mt-1",
          trend === 'up' && "text-green-400",
          trend === 'down' && "text-red-400",
          trend === 'stable' && "text-zinc-400"
        )}>
          {trend === 'up' && '↑'}
          {trend === 'down' && '↓'}
          {trend === 'stable' && '→'}
        </div>
      )}
    </motion.div>
  );
}

function OpportunityCard({ 
  type, 
  count, 
  profit, 
  successRate 
}: {
  type: string;
  count: number;
  profit: number;
  successRate: number;
}) {
  return (
    <div className="glass rounded-lg p-4 hover:border-zinc-700 transition-all">
      <div className="flex items-center justify-between mb-2">
        <span className={clsx(
          "text-xs px-2 py-0.5 rounded-full uppercase",
          type === 'arbitrage' && "bg-blue-500/10 text-blue-400",
          type === 'liquidation' && "bg-orange-500/10 text-orange-400",
          type === 'sandwich' && "bg-purple-500/10 text-purple-400",
          type === 'jit' && "bg-cyan-500/10 text-cyan-400"
        )}>
          {type}
        </span>
        <span className="text-xs text-zinc-500">{count} ops</span>
      </div>
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <span className="text-xs text-zinc-600">Profit</span>
          <span className="text-sm font-medium text-green-400">
            ${numeral(profit).format('0,0.00')}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-xs text-zinc-600">Success</span>
          <span className={clsx(
            "text-sm font-medium",
            successRate >= 65 ? "text-green-400" : 
            successRate >= 55 ? "text-yellow-400" : "text-red-400"
          )}>
            {successRate.toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
}

export default function RealtimePage() {
  const snap = useSnapshot(mevStore);
  const [arbitrageByDEX, setArbitrageByDEX] = useState<any[]>([]);
  const [isPaused, setIsPaused] = useState(false);
  
  // Keyboard shortcuts
  useHotkeys('space', () => setIsPaused(prev => !prev), []);
  useHotkeys('r', () => mevActions.reset(), []);
  useHotkeys('s', () => mevActions.updateSettings({ autoScroll: !snap.settings.autoScroll }), [snap.settings.autoScroll]);
  
  // Fetch DEX arbitrage data
  useEffect(() => {
    const fetchData = async () => {
      if (isPaused) return;
      
      try {
        const dexData = await ClickHouseQueries.getArbitrageByDEX();
        setArbitrageByDEX(dexData);
      } catch (err) {
        console.error('Failed to fetch DEX data:', err);
      }
    };
    
    fetchData();
    const interval = setInterval(fetchData, 3000);
    return () => clearInterval(interval);
  }, [isPaused]);
  
  const handleEmergencyStop = useCallback(() => {
    console.log('EMERGENCY STOP TRIGGERED');
    // This would trigger the actual emergency stop via API
    fetch('/api/emergency-stop', { method: 'POST' });
  }, []);
  
  return (
    <div className="space-y-6">
      {/* Header with controls */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Real-Time Monitor</h1>
          <p className="text-sm text-zinc-500 mt-1">
            Live MEV opportunity tracking and execution
          </p>
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsPaused(!isPaused)}
            className={clsx(
              "btn",
              isPaused ? "bg-yellow-500/10 text-yellow-400" : "btn-ghost"
            )}
          >
            {isPaused ? 'Resume' : 'Pause'}
          </button>
          
          <button
            onClick={handleEmergencyStop}
            className="btn btn-danger"
          >
            Emergency Stop
          </button>
        </div>
      </div>
      
      {/* Real-time metrics grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3">
        <MetricTile 
          label="Latency P50"
          value={snap.health.decisionLatencyP50.toFixed(1)}
          unit="ms"
          status={snap.health.decisionLatencyP50 <= 8 ? 'good' : snap.health.decisionLatencyP50 <= 15 ? 'warning' : 'critical'}
        />
        <MetricTile 
          label="Latency P99"
          value={snap.health.decisionLatencyP99.toFixed(1)}
          unit="ms"
          status={snap.health.decisionLatencyP99 <= 20 ? 'good' : snap.health.decisionLatencyP99 <= 30 ? 'warning' : 'critical'}
        />
        <MetricTile 
          label="Land Rate"
          value={snap.currentLandRate.toFixed(1)}
          unit="%"
          status={snap.currentLandRate >= 65 ? 'good' : snap.currentLandRate >= 55 ? 'warning' : 'critical'}
        />
        <MetricTile 
          label="Ingestion"
          value={numeral(snap.health.ingestionRate).format('0a')}
          unit="/s"
          status={snap.health.ingestionRate >= 200000 ? 'good' : snap.health.ingestionRate >= 150000 ? 'warning' : 'critical'}
        />
        <MetricTile 
          label="Model Time"
          value={(snap.health.modelInferenceTime * 1000).toFixed(0)}
          unit="μs"
          status={snap.health.modelInferenceTime <= 0.0001 ? 'good' : 'warning'}
        />
        <MetricTile 
          label="Active Ops"
          value={snap.activeOpportunities}
          trend={snap.activeOpportunities > 10 ? 'up' : snap.activeOpportunities > 5 ? 'stable' : 'down'}
        />
        <MetricTile 
          label="Buffer"
          value={snap.transactionBuffer.length}
          status={snap.transactionBuffer.length < 100 ? 'good' : snap.transactionBuffer.length < 500 ? 'warning' : 'critical'}
        />
        <MetricTile 
          label="Profit/Hr"
          value={numeral(snap.profitLast24h / 24).format('$0.0a')}
          trend="up"
        />
      </div>
      
      {/* Main content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Transaction feed - 2 columns */}
        <div className="lg:col-span-2">
          <TransactionFeed />
        </div>
        
        {/* Right sidebar */}
        <div className="space-y-4">
          {/* Opportunity breakdown */}
          <div className="glass rounded-xl p-4">
            <h3 className="text-sm font-medium text-zinc-400 mb-3">Active Opportunities</h3>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(snap.opportunityTypes).map(([type, count]) => (
                <OpportunityCard
                  key={type}
                  type={type}
                  count={count}
                  profit={0} // Would need to calculate from transactions
                  successRate={65} // Would need to calculate
                />
              ))}
            </div>
          </div>
          
          {/* DEX Performance */}
          <div className="glass rounded-xl p-4">
            <h3 className="text-sm font-medium text-zinc-400 mb-3">DEX Performance</h3>
            <div className="space-y-2">
              {arbitrageByDEX.slice(0, 5).map((dex, idx) => (
                <div key={idx} className="flex items-center justify-between text-xs">
                  <code className="text-zinc-500 truncate max-w-[100px]">
                    {dex.dex.slice(0, 8)}...
                  </code>
                  <div className="flex items-center gap-3">
                    <span className="text-green-400">
                      ${numeral(dex.total_profit).format('0.0a')}
                    </span>
                    <span className="text-zinc-600">
                      {dex.opportunity_count}
                    </span>
                    <span className={clsx(
                      dex.success_rate >= 65 ? "text-green-400" : "text-yellow-400"
                    )}>
                      {dex.success_rate.toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          {/* System Status */}
          <div className={clsx(
            "rounded-xl p-4",
            snap.health.isHealthy 
              ? "bg-green-500/10 border border-green-500/20" 
              : "bg-red-500/10 border border-red-500/20"
          )}>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium">System Status</h3>
              <div className={clsx(
                "w-3 h-3 rounded-full",
                snap.health.isHealthy ? "bg-green-500 pulse-success" : "bg-red-500 pulse-error"
              )} />
            </div>
            
            {snap.health.alerts.length > 0 ? (
              <div className="space-y-1">
                {snap.health.alerts.slice(0, 3).map((alert, idx) => (
                  <div key={idx} className="text-xs text-zinc-400">
                    • {alert.message}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-xs text-green-400">
                All systems operational
              </div>
            )}
          </div>
          
          {/* Keyboard shortcuts help */}
          <div className="glass rounded-xl p-4">
            <h3 className="text-sm font-medium text-zinc-400 mb-3">Keyboard Shortcuts</h3>
            <div className="space-y-1 text-xs">
              <div className="flex items-center justify-between">
                <kbd className="px-2 py-0.5 bg-zinc-800 rounded">Space</kbd>
                <span className="text-zinc-500">Pause/Resume</span>
              </div>
              <div className="flex items-center justify-between">
                <kbd className="px-2 py-0.5 bg-zinc-800 rounded">R</kbd>
                <span className="text-zinc-500">Reset Data</span>
              </div>
              <div className="flex items-center justify-between">
                <kbd className="px-2 py-0.5 bg-zinc-800 rounded">S</kbd>
                <span className="text-zinc-500">Toggle Auto-Scroll</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}