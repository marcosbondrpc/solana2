'use client';

import { useSnapshot } from 'valtio';
import { mevStore } from '../stores/mevStore';
import { useEffect, useRef, useState } from 'react';
import { clsx } from 'clsx';
import numeral from 'numeral';
import { motion, AnimatePresence } from 'framer-motion';
import { ClickHouseQueries } from '../lib/clickhouse';
import { usePerformanceMonitor } from '../hooks/usePerformance';
import dynamic from 'next/dynamic';

// Dynamically import heavy components
const TransactionFeed = dynamic(() => import('../components/TransactionFeed'), { 
  ssr: false,
  loading: () => <div className="h-96 glass rounded-xl animate-pulse" />
});

const ProfitChart = dynamic(() => import('../components/charts/ProfitChart'), { 
  ssr: false,
  loading: () => <div className="h-64 glass rounded-xl animate-pulse" />
});

const LatencyHeatmap = dynamic(() => import('../components/charts/LatencyHeatmap'), { 
  ssr: false,
  loading: () => <div className="h-64 glass rounded-xl animate-pulse" />
});

const BundleSuccessRate = dynamic(() => import('../components/charts/BundleSuccessRate'), { 
  ssr: false,
  loading: () => <div className="h-64 glass rounded-xl animate-pulse" />
});

function MetricCard({ 
  title, 
  value, 
  change, 
  unit = '', 
  status = 'neutral',
  icon 
}: {
  title: string;
  value: string | number;
  change?: number;
  unit?: string;
  status?: 'success' | 'error' | 'warning' | 'neutral';
  icon?: string;
}) {
  return (
    <motion.div 
      className="metric-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs text-zinc-500 uppercase tracking-wide">{title}</span>
        {icon && <span className="text-lg">{icon}</span>}
      </div>
      <div className="flex items-baseline gap-1">
        <span className={clsx(
          "text-2xl font-bold",
          status === 'success' && "text-green-400",
          status === 'error' && "text-red-400",
          status === 'warning' && "text-yellow-400",
          status === 'neutral' && "text-white"
        )}>
          {value}
        </span>
        {unit && <span className="text-sm text-zinc-500">{unit}</span>}
      </div>
      {change !== undefined && (
        <div className={clsx(
          "text-xs mt-1",
          change > 0 ? "text-green-400" : "text-red-400"
        )}>
          {change > 0 ? '↑' : '↓'} {Math.abs(change).toFixed(1)}%
        </div>
      )}
    </motion.div>
  );
}

function DNAFingerprint({ dna, verified }: { dna: string; verified: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <code className="dna-fingerprint">{dna.slice(0, 8)}...{dna.slice(-8)}</code>
      {verified ? (
        <span className="text-green-400 text-xs">✓</span>
      ) : (
        <span className="text-yellow-400 text-xs">⏳</span>
      )}
    </div>
  );
}

function AlertPanel() {
  const snap = useSnapshot(mevStore);
  const alerts = snap.health.alerts;
  
  if (alerts.length === 0) return null;
  
  return (
    <AnimatePresence>
      <motion.div 
        className="glass-dark rounded-xl p-4 border-red-500/20"
        initial={{ opacity: 0, x: 100 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: 100 }}
      >
        <h3 className="text-sm font-medium text-red-400 mb-2">System Alerts</h3>
        <div className="space-y-1">
          {alerts.slice(0, 5).map((alert, idx) => (
            <div key={idx} className={clsx(
              "text-xs p-2 rounded",
              alert.level === 'error' && "bg-red-500/10 text-red-400",
              alert.level === 'warning' && "bg-yellow-500/10 text-yellow-400",
              alert.level === 'info' && "bg-blue-500/10 text-blue-400"
            )}>
              {alert.message}
            </div>
          ))}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

function TopRoutesPanel() {
  const [routes, setRoutes] = useState<any[]>([]);
  
  useEffect(() => {
    const fetchRoutes = async () => {
      try {
        const data = await ClickHouseQueries.getTopRoutes(5);
        setRoutes(data);
      } catch (err) {
        console.error('Failed to fetch routes:', err);
      }
    };
    
    fetchRoutes();
    const interval = setInterval(fetchRoutes, 10000);
    return () => clearInterval(interval);
  }, []);
  
  return (
    <div className="glass rounded-xl p-4">
      <h3 className="text-sm font-medium text-zinc-400 mb-3">Top Profitable Routes</h3>
      <div className="space-y-2">
        {routes.map((route, idx) => (
          <div key={idx} className="flex items-center justify-between text-xs">
            <div className="flex-1 truncate">
              <code className="text-zinc-300">{route.route}</code>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-green-400 font-medium">
                ${numeral(route.total_profit).format('0,0.00')}
              </span>
              <span className={clsx(
                "px-1.5 py-0.5 rounded",
                route.success_rate >= 65 ? "bg-green-500/10 text-green-400" : "bg-yellow-500/10 text-yellow-400"
              )}>
                {route.success_rate.toFixed(0)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const snap = useSnapshot(mevStore);
  const { measureRender, mark, measure } = usePerformanceMonitor('Dashboard');
  
  useEffect(() => {
    mark('mount');
    
    // Fetch initial data
    const fetchData = async () => {
      mark('fetch-start');
      try {
        const [transactions, metrics, stats] = await Promise.all([
          ClickHouseQueries.getRecentTransactions(100),
          ClickHouseQueries.getSystemMetrics('1m'),
          ClickHouseQueries.getBundleStats('1m'),
        ]);
        
        // Update store with fetched data
        if (transactions.length > 0) {
          transactions.forEach(tx => mevStore.transactions.push(tx));
        }
        if (metrics.length > 0) {
          mevStore.systemMetrics = metrics;
        }
        if (stats.length > 0) {
          mevStore.bundleStats = stats;
        }
      } catch (err) {
        console.error('Failed to fetch initial data:', err);
      }
      measure('fetch-time', 'fetch-start');
    };
    
    fetchData();
    
    // Set up polling for real-time updates
    const interval = setInterval(fetchData, 5000);
    
    measure('mount-time', 'mount');
    
    return () => clearInterval(interval);
  }, [mark, measure]);
  
  const endRender = measureRender();
  useEffect(() => {
    endRender();
  });
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
            MEV Command Center
          </h1>
          <p className="text-sm text-zinc-500 mt-1">
            Real-time monitoring of Solana MEV infrastructure
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className={clsx(
            "px-4 py-2 rounded-lg text-sm font-medium",
            snap.health.isHealthy 
              ? "bg-green-500/10 text-green-400 border border-green-500/20" 
              : "bg-red-500/10 text-red-400 border border-red-500/20 pulse-error"
          )}>
            {snap.health.isHealthy ? 'System Healthy' : 'System Degraded'}
          </div>
        </div>
      </div>
      
      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <MetricCard 
          title="Total Profit (24h)"
          value={numeral(snap.profitLast24h).format('$0,0.00')}
          status="success"
          icon="💰"
        />
        <MetricCard 
          title="Active Opportunities"
          value={snap.activeOpportunities}
          status="neutral"
          icon="🎯"
        />
        <MetricCard 
          title="Land Rate"
          value={snap.currentLandRate.toFixed(1)}
          unit="%"
          status={snap.currentLandRate >= 65 ? 'success' : snap.currentLandRate >= 55 ? 'warning' : 'error'}
          icon="🎯"
        />
        <MetricCard 
          title="Latency P50"
          value={snap.health.decisionLatencyP50.toFixed(1)}
          unit="ms"
          status={snap.health.decisionLatencyP50 <= 8 ? 'success' : snap.health.decisionLatencyP50 <= 15 ? 'warning' : 'error'}
          icon="⚡"
        />
        <MetricCard 
          title="Ingestion Rate"
          value={numeral(snap.health.ingestionRate).format('0,0')}
          unit="/s"
          status={snap.health.ingestionRate >= 200000 ? 'success' : snap.health.ingestionRate >= 150000 ? 'warning' : 'error'}
          icon="📊"
        />
        <MetricCard 
          title="Transactions"
          value={snap.transactions.length}
          status="neutral"
          icon="📝"
        />
      </div>
      
      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Transaction Feed */}
        <div className="lg:col-span-2 space-y-4">
          <TransactionFeed />
          
          <div className="grid grid-cols-2 gap-4">
            <ProfitChart />
            <BundleSuccessRate />
          </div>
        </div>
        
        {/* Right Column - Status and Controls */}
        <div className="space-y-4">
          {/* System Health Alerts */}
          {snap.health.alerts.length > 0 && <AlertPanel />}
          
          {/* Top Routes */}
          <TopRoutesPanel />
          
          {/* Latency Distribution */}
          <LatencyHeatmap />
          
          {/* Opportunity Types */}
          <div className="glass rounded-xl p-4">
            <h3 className="text-sm font-medium text-zinc-400 mb-3">Opportunity Distribution</h3>
            <div className="space-y-2">
              {Object.entries(snap.opportunityTypes).map(([type, count]) => (
                <div key={type} className="flex items-center justify-between">
                  <span className="text-xs text-zinc-300 capitalize">{type}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-24 h-1 bg-zinc-800 rounded-full overflow-hidden">
                      <motion.div 
                        className="h-full bg-gradient-to-r from-cyan-500 to-purple-500"
                        initial={{ width: 0 }}
                        animate={{ width: `${(count / Math.max(...Object.values(snap.opportunityTypes))) * 100}%` }}
                        transition={{ duration: 0.5 }}
                      />
                    </div>
                    <span className="text-xs text-zinc-500 w-8 text-right">{count}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          {/* Recent DNA Fingerprints */}
          <div className="glass rounded-xl p-4">
            <h3 className="text-sm font-medium text-zinc-400 mb-3">Recent DNA Fingerprints</h3>
            <div className="space-y-1">
              {Array.from(snap.dnaFingerprints.entries()).slice(0, 5).map(([dna, info]) => (
                <DNAFingerprint key={dna} dna={dna} verified={info.verified} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}