import { proxy, subscribe, snapshot } from 'valtio';
import { subscribeKey } from 'valtio/utils';
import { MEVTransaction, SystemMetrics, BundleStats } from '../lib/clickhouse';

// Connection status
export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

// MEV store state
interface MEVStore {
  // Connection status
  wsStatus: ConnectionStatus;
  clickhouseStatus: ConnectionStatus;
  backendStatus: ConnectionStatus;
  
  // Real-time data
  transactions: MEVTransaction[];
  transactionBuffer: MEVTransaction[]; // Buffer for batch updates
  lastTransactionTime: number;
  
  // Metrics
  systemMetrics: SystemMetrics[];
  bundleStats: BundleStats[];
  currentLatency: number;
  currentLandRate: number;
  totalProfit: number;
  profitLast24h: number;
  
  // Opportunities
  activeOpportunities: number;
  opportunityTypes: Record<string, number>;
  topRoutes: Array<{
    route: string;
    profit: number;
    count: number;
    successRate: number;
  }>;
  
  // System health
  health: {
    ingestionRate: number;
    decisionLatencyP50: number;
    decisionLatencyP99: number;
    modelInferenceTime: number;
    bundleLandRate: number;
    isHealthy: boolean;
    alerts: Array<{
      level: 'info' | 'warning' | 'error';
      message: string;
      timestamp: number;
    }>;
  };
  
  // DNA tracking
  dnaFingerprints: Map<string, {
    timestamp: number;
    verified: boolean;
    merkleProof?: string;
  }>;
  
  // Settings
  settings: {
    autoScroll: boolean;
    soundEnabled: boolean;
    maxTransactions: number;
    updateInterval: number;
    theme: 'dark' | 'midnight' | 'matrix';
  };
}

// Create the store
export const mevStore = proxy<MEVStore>({
  // Connection status
  wsStatus: 'disconnected',
  clickhouseStatus: 'disconnected',
  backendStatus: 'disconnected',
  
  // Real-time data
  transactions: [],
  transactionBuffer: [],
  lastTransactionTime: 0,
  
  // Metrics
  systemMetrics: [],
  bundleStats: [],
  currentLatency: 0,
  currentLandRate: 0,
  totalProfit: 0,
  profitLast24h: 0,
  
  // Opportunities
  activeOpportunities: 0,
  opportunityTypes: {},
  topRoutes: [],
  
  // System health
  health: {
    ingestionRate: 0,
    decisionLatencyP50: 0,
    decisionLatencyP99: 0,
    modelInferenceTime: 0,
    bundleLandRate: 0,
    isHealthy: true,
    alerts: [],
  },
  
  // DNA tracking
  dnaFingerprints: new Map(),
  
  // Settings
  settings: {
    autoScroll: true,
    soundEnabled: true,
    maxTransactions: 1000,
    updateInterval: 100, // ms
    theme: 'dark',
  },
});

// Actions
export const mevActions = {
  // Add new transaction
  addTransaction(tx: MEVTransaction) {
    // Add to buffer for batching
    mevStore.transactionBuffer.push(tx);
    
    // Track DNA fingerprint
    if (tx.decision_dna) {
      mevStore.dnaFingerprints.set(tx.decision_dna, {
        timestamp: Date.now(),
        verified: false,
      });
    }
    
    // Update opportunity types
    if (tx.transaction_type) {
      mevStore.opportunityTypes[tx.transaction_type] = 
        (mevStore.opportunityTypes[tx.transaction_type] || 0) + 1;
    }
    
    // Update metrics
    if (tx.profit_amount > 0) {
      mevStore.totalProfit += tx.profit_amount;
    }
    
    mevStore.lastTransactionTime = Date.now();
  },
  
  // Flush transaction buffer (for batch updates)
  flushTransactionBuffer() {
    if (mevStore.transactionBuffer.length === 0) return;
    
    const newTransactions = [...mevStore.transactionBuffer, ...mevStore.transactions];
    
    // Keep only the latest transactions based on settings
    mevStore.transactions = newTransactions.slice(0, mevStore.settings.maxTransactions);
    
    // Clear buffer
    mevStore.transactionBuffer = [];
  },
  
  // Update system metrics
  updateSystemMetrics(metrics: SystemMetrics[]) {
    mevStore.systemMetrics = metrics;
    
    // Extract key metrics
    const latencyMetric = metrics.find(m => m.metric_name === 'decision_latency');
    if (latencyMetric) {
      mevStore.health.decisionLatencyP50 = latencyMetric.p50;
      mevStore.health.decisionLatencyP99 = latencyMetric.p99;
      mevStore.currentLatency = latencyMetric.metric_value;
    }
    
    const inferenceMetric = metrics.find(m => m.metric_name === 'model_inference');
    if (inferenceMetric) {
      mevStore.health.modelInferenceTime = inferenceMetric.metric_value;
    }
    
    const ingestionMetric = metrics.find(m => m.metric_name === 'ingestion_rate');
    if (ingestionMetric) {
      mevStore.health.ingestionRate = ingestionMetric.metric_value;
    }
  },
  
  // Update bundle stats
  updateBundleStats(stats: BundleStats[]) {
    mevStore.bundleStats = stats;
    
    if (stats.length > 0) {
      const latest = stats[0];
      mevStore.currentLandRate = latest.land_rate;
      mevStore.health.bundleLandRate = latest.land_rate;
      
      // Calculate 24h profit
      const last24h = stats.filter(s => 
        new Date(s.timestamp).getTime() > Date.now() - 24 * 60 * 60 * 1000
      );
      mevStore.profitLast24h = last24h.reduce((sum, s) => sum + s.total_profit, 0);
    }
  },
  
  // Update health status
  updateHealth() {
    const health = mevStore.health;
    
    // Check if system is healthy based on SLOs
    health.isHealthy = 
      health.decisionLatencyP50 <= 8 &&
      health.decisionLatencyP99 <= 20 &&
      health.bundleLandRate >= 55 &&
      health.ingestionRate >= 200000 &&
      health.modelInferenceTime <= 0.1;
    
    // Generate alerts
    const alerts = [];
    
    if (health.decisionLatencyP50 > 8) {
      alerts.push({
        level: 'warning' as const,
        message: `P50 latency ${health.decisionLatencyP50.toFixed(1)}ms exceeds 8ms target`,
        timestamp: Date.now(),
      });
    }
    
    if (health.decisionLatencyP99 > 20) {
      alerts.push({
        level: 'error' as const,
        message: `P99 latency ${health.decisionLatencyP99.toFixed(1)}ms exceeds 20ms target`,
        timestamp: Date.now(),
      });
    }
    
    if (health.bundleLandRate < 55) {
      alerts.push({
        level: 'error' as const,
        message: `Bundle land rate ${health.bundleLandRate.toFixed(1)}% below 55% threshold`,
        timestamp: Date.now(),
      });
    }
    
    if (health.ingestionRate < 200000) {
      alerts.push({
        level: 'warning' as const,
        message: `Ingestion rate ${(health.ingestionRate / 1000).toFixed(0)}k/s below 200k/s target`,
        timestamp: Date.now(),
      });
    }
    
    // Keep only recent alerts (last 5 minutes)
    const recentAlerts = health.alerts.filter(a => 
      a.timestamp > Date.now() - 5 * 60 * 1000
    );
    
    health.alerts = [...alerts, ...recentAlerts].slice(0, 10);
  },
  
  // Update connection status
  setConnectionStatus(service: 'ws' | 'clickhouse' | 'backend', status: ConnectionStatus) {
    switch (service) {
      case 'ws':
        mevStore.wsStatus = status;
        break;
      case 'clickhouse':
        mevStore.clickhouseStatus = status;
        break;
      case 'backend':
        mevStore.backendStatus = status;
        break;
    }
  },
  
  // Update settings
  updateSettings(settings: Partial<MEVStore['settings']>) {
    Object.assign(mevStore.settings, settings);
  },
  
  // Clear old transactions
  pruneTransactions() {
    const maxAge = 60 * 60 * 1000; // 1 hour
    const cutoff = Date.now() - maxAge;
    
    mevStore.transactions = mevStore.transactions.filter(tx => 
      new Date(tx.timestamp).getTime() > cutoff
    );
  },
  
  // Reset store
  reset() {
    mevStore.transactions = [];
    mevStore.transactionBuffer = [];
    mevStore.systemMetrics = [];
    mevStore.bundleStats = [];
    mevStore.totalProfit = 0;
    mevStore.profitLast24h = 0;
    mevStore.activeOpportunities = 0;
    mevStore.opportunityTypes = {};
    mevStore.topRoutes = [];
    mevStore.dnaFingerprints.clear();
    mevStore.health.alerts = [];
  },
};

// Subscriptions for side effects
let flushInterval: NodeJS.Timeout;

// Start batch flushing
export function startBatchFlushing() {
  if (flushInterval) return;
  
  flushInterval = setInterval(() => {
    if (mevStore.transactionBuffer.length > 0) {
      mevActions.flushTransactionBuffer();
    }
  }, mevStore.settings.updateInterval);
}

// Stop batch flushing
export function stopBatchFlushing() {
  if (flushInterval) {
    clearInterval(flushInterval);
    flushInterval = undefined as any;
  }
}

// Subscribe to health changes
subscribe(mevStore.health, () => {
  mevActions.updateHealth();
});

// Auto-prune old data every minute
setInterval(() => {
  mevActions.pruneTransactions();
}, 60 * 1000);

// Export snapshot for read-only access
export function getMEVSnapshot() {
  return snapshot(mevStore);
}