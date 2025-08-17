import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { devtools } from 'zustand/middleware';
import { enableMapSet } from 'immer';
import EventEmitter from 'eventemitter3';

// Enable Map/Set support in Immer
enableMapSet();

export interface ArbitrageOpportunity {
  id: string;
  timestamp: number;
  dexA: string;
  dexB: string;
  tokenIn: string;
  tokenOut: string;
  amountIn: number;
  expectedProfit: number;
  actualProfit?: number;
  gasUsed?: number;
  slippage: number;
  path: string[];
  confidence: number;
  status: 'pending' | 'executing' | 'completed' | 'failed';
  txHash?: string;
  blockHeight?: number;
  latency?: number;
}

export interface JitoBundle {
  id: string;
  timestamp: number;
  bundleId: string;
  transactions: string[];
  tip: number;
  landingSlot?: number;
  status: 'pending' | 'landed' | 'failed' | 'expired';
  relayUrl: string;
  submitLatency: number;
  confirmationTime?: number;
  profitability?: number;
  mevType: 'arb' | 'liquidation' | 'sandwich' | 'jit';
}

export interface LatencyMetrics {
  timestamp: number;
  rpcLatency: number;
  quicLatency: number;
  tcpLatency: number;
  websocketLatency: number;
  parsingLatency: number;
  calculationLatency: number;
  executionLatency: number;
  confirmationLatency: number;
  p50: number;
  p95: number;
  p99: number;
  jitter: number;
}

export interface DEXFlow {
  timestamp: number;
  flows: Map<string, {
    from: string;
    to: string;
    token: string;
    volume: number;
    count: number;
    avgPrice: number;
  }[]>;
  totalVolume: number;
  uniqueTokens: Set<string>;
  arbitragePaths: string[][];
}

export interface ProfitMetrics {
  timestamp: number;
  totalProfit: number;
  dailyProfit: number;
  hourlyProfit: number;
  successRate: number;
  avgProfit: number;
  maxProfit: number;
  minProfit: number;
  gasSpent: number;
  netProfit: number;
  roi: number;
  profitByDex: Map<string, number>;
  profitByToken: Map<string, number>;
  cumulativeProfit: number[];
}

export interface SystemPerformance {
  cpuUsage: number[];
  memoryUsage: number;
  networkIn: number;
  networkOut: number;
  diskRead: number;
  diskWrite: number;
  threadCount: number;
  handleCount: number;
  goroutines?: number;
  gcPauses?: number[];
  heapSize?: number;
  timestamp: number;
}

export interface Alert {
  id: string;
  timestamp: number;
  type: 'info' | 'warning' | 'error' | 'critical';
  category: 'arbitrage' | 'bundle' | 'latency' | 'system' | 'profit';
  message: string;
  details?: any;
  acknowledged: boolean;
  autoResolve?: boolean;
  resolveTime?: number;
}

interface MEVState {
  // Real-time data
  arbitrageOpportunities: Map<string, ArbitrageOpportunity>;
  jitoBundles: Map<string, JitoBundle>;
  latencyMetrics: LatencyMetrics[];
  dexFlows: DEXFlow[];
  profitMetrics: ProfitMetrics;
  systemPerformance: SystemPerformance[];
  alerts: Alert[];
  
  // Configuration
  autoExecute: boolean;
  minProfitThreshold: number;
  maxSlippage: number;
  maxGasPrice: number;
  selectedDexes: Set<string>;
  enabledStrategies: Set<string>;
  
  // Statistics
  totalOpportunities: number;
  executedOpportunities: number;
  successfulTrades: number;
  failedTrades: number;
  totalVolume: number;
  
  // Connection status
  isConnected: boolean;
  connectionLatency: number;
  lastHeartbeat: number;
  nodeHealth: 'healthy' | 'degraded' | 'unhealthy';
  
  // Actions
  addArbitrageOpportunity: (opp: ArbitrageOpportunity) => void;
  updateArbitrageStatus: (id: string, status: ArbitrageOpportunity['status'], data?: Partial<ArbitrageOpportunity>) => void;
  addJitoBundle: (bundle: JitoBundle) => void;
  updateBundleStatus: (id: string, status: JitoBundle['status'], data?: Partial<JitoBundle>) => void;
  addLatencyMetric: (metric: LatencyMetrics) => void;
  updateDEXFlow: (flow: DEXFlow) => void;
  updateProfitMetrics: (metrics: Partial<ProfitMetrics>) => void;
  updateSystemPerformance: (perf: SystemPerformance) => void;
  addAlert: (alert: Omit<Alert, 'id' | 'timestamp' | 'acknowledged'>) => void;
  acknowledgeAlert: (id: string) => void;
  clearOldData: (before: number) => void;
  setConfiguration: (config: Partial<{
    autoExecute: boolean;
    minProfitThreshold: number;
    maxSlippage: number;
    maxGasPrice: number;
  }>) => void;
  toggleDex: (dex: string) => void;
  toggleStrategy: (strategy: string) => void;
  setConnectionStatus: (connected: boolean, latency?: number) => void;
  updateNodeHealth: (health: 'healthy' | 'degraded' | 'unhealthy') => void;
  reset: () => void;
}

// Event emitter for high-frequency updates
export const mevEvents = new EventEmitter();

// Performance monitoring
const performanceMonitor = {
  updateCount: 0,
  lastReset: Date.now(),
  resetIfNeeded() {
    const now = Date.now();
    if (now - this.lastReset > 1000) {
      console.log(`MEV Store: ${this.updateCount} updates/sec`);
      this.updateCount = 0;
      this.lastReset = now;
    }
  }
};

const initialState = {
  arbitrageOpportunities: new Map(),
  jitoBundles: new Map(),
  latencyMetrics: [],
  dexFlows: [],
  profitMetrics: {
    timestamp: Date.now(),
    totalProfit: 0,
    dailyProfit: 0,
    hourlyProfit: 0,
    successRate: 0,
    avgProfit: 0,
    maxProfit: 0,
    minProfit: 0,
    gasSpent: 0,
    netProfit: 0,
    roi: 0,
    profitByDex: new Map(),
    profitByToken: new Map(),
    cumulativeProfit: []
  },
  systemPerformance: [],
  alerts: [],
  autoExecute: false,
  minProfitThreshold: 0.01,
  maxSlippage: 0.02,
  maxGasPrice: 100,
  selectedDexes: new Set(['raydium', 'orca', 'phoenix', 'meteora']),
  enabledStrategies: new Set(['arb', 'jit', 'sandwich']),
  totalOpportunities: 0,
  executedOpportunities: 0,
  successfulTrades: 0,
  failedTrades: 0,
  totalVolume: 0,
  isConnected: false,
  connectionLatency: 0,
  lastHeartbeat: Date.now(),
  nodeHealth: 'healthy' as const
};

export const useMEVStore = create<MEVState>()(
  devtools(
    subscribeWithSelector(
      immer((set, get) => ({
        ...initialState,
        
        addArbitrageOpportunity: (opp) => {
          set((state) => {
            state.arbitrageOpportunities.set(opp.id, opp);
            state.totalOpportunities++;
            
            // Keep only last 10000 opportunities for memory efficiency
            if (state.arbitrageOpportunities.size > 10000) {
              const firstKey = state.arbitrageOpportunities.keys().next().value;
              state.arbitrageOpportunities.delete(firstKey);
            }
          });
          
          mevEvents.emit('arbitrage:new', opp);
          performanceMonitor.updateCount++;
          performanceMonitor.resetIfNeeded();
        },
        
        updateArbitrageStatus: (id, status, data) => {
          set((state) => {
            const opp = state.arbitrageOpportunities.get(id);
            if (opp) {
              opp.status = status;
              if (data) Object.assign(opp, data);
              
              if (status === 'executing') {
                state.executedOpportunities++;
              } else if (status === 'completed') {
                state.successfulTrades++;
                if (opp.actualProfit) {
                  state.profitMetrics.totalProfit += opp.actualProfit;
                  state.profitMetrics.netProfit += opp.actualProfit - (opp.gasUsed || 0);
                }
              } else if (status === 'failed') {
                state.failedTrades++;
              }
            }
          });
          
          mevEvents.emit('arbitrage:update', { id, status });
        },
        
        addJitoBundle: (bundle) => {
          set((state) => {
            state.jitoBundles.set(bundle.id, bundle);
            
            // Keep only last 5000 bundles
            if (state.jitoBundles.size > 5000) {
              const firstKey = state.jitoBundles.keys().next().value;
              state.jitoBundles.delete(firstKey);
            }
          });
          
          mevEvents.emit('bundle:new', bundle);
        },
        
        updateBundleStatus: (id, status, data) => {
          set((state) => {
            const bundle = state.jitoBundles.get(id);
            if (bundle) {
              bundle.status = status;
              if (data) Object.assign(bundle, data);
            }
          });
          
          mevEvents.emit('bundle:update', { id, status });
        },
        
        addLatencyMetric: (metric) => {
          set((state) => {
            state.latencyMetrics.push(metric);
            
            // Keep only last 1000 metrics (about 16 minutes at 1/sec)
            if (state.latencyMetrics.length > 1000) {
              state.latencyMetrics.shift();
            }
          });
          
          mevEvents.emit('latency:update', metric);
        },
        
        updateDEXFlow: (flow) => {
          set((state) => {
            state.dexFlows.push(flow);
            
            // Keep only last 100 flow snapshots
            if (state.dexFlows.length > 100) {
              state.dexFlows.shift();
            }
            
            state.totalVolume += flow.totalVolume;
          });
          
          mevEvents.emit('flow:update', flow);
        },
        
        updateProfitMetrics: (metrics) => {
          set((state) => {
            Object.assign(state.profitMetrics, metrics);
            state.profitMetrics.timestamp = Date.now();
            
            // Update cumulative profit array
            if (state.profitMetrics.cumulativeProfit.length > 1000) {
              state.profitMetrics.cumulativeProfit.shift();
            }
            state.profitMetrics.cumulativeProfit.push(state.profitMetrics.totalProfit);
          });
          
          mevEvents.emit('profit:update', metrics);
        },
        
        updateSystemPerformance: (perf) => {
          set((state) => {
            state.systemPerformance.push(perf);
            
            // Keep only last 500 performance snapshots
            if (state.systemPerformance.length > 500) {
              state.systemPerformance.shift();
            }
          });
          
          mevEvents.emit('performance:update', perf);
        },
        
        addAlert: (alert) => {
          const fullAlert: Alert = {
            ...alert,
            id: `alert-${Date.now()}-${Math.random()}`,
            timestamp: Date.now(),
            acknowledged: false
          };
          
          set((state) => {
            state.alerts.unshift(fullAlert);
            
            // Keep only last 100 alerts
            if (state.alerts.length > 100) {
              state.alerts.pop();
            }
          });
          
          mevEvents.emit('alert:new', fullAlert);
        },
        
        acknowledgeAlert: (id) => {
          set((state) => {
            const alert = state.alerts.find(a => a.id === id);
            if (alert) {
              alert.acknowledged = true;
              if (alert.autoResolve) {
                alert.resolveTime = Date.now();
              }
            }
          });
        },
        
        clearOldData: (before) => {
          set((state) => {
            // Clear old arbitrage opportunities
            for (const [id, opp] of state.arbitrageOpportunities) {
              if (opp.timestamp < before) {
                state.arbitrageOpportunities.delete(id);
              }
            }
            
            // Clear old bundles
            for (const [id, bundle] of state.jitoBundles) {
              if (bundle.timestamp < before) {
                state.jitoBundles.delete(id);
              }
            }
            
            // Clear old metrics
            state.latencyMetrics = state.latencyMetrics.filter(m => m.timestamp >= before);
            state.dexFlows = state.dexFlows.filter(f => f.timestamp >= before);
            state.systemPerformance = state.systemPerformance.filter(p => p.timestamp >= before);
            state.alerts = state.alerts.filter(a => a.timestamp >= before);
          });
        },
        
        setConfiguration: (config) => {
          set((state) => {
            Object.assign(state, config);
          });
          
          mevEvents.emit('config:update', config);
        },
        
        toggleDex: (dex) => {
          set((state) => {
            if (state.selectedDexes.has(dex)) {
              state.selectedDexes.delete(dex);
            } else {
              state.selectedDexes.add(dex);
            }
          });
        },
        
        toggleStrategy: (strategy) => {
          set((state) => {
            if (state.enabledStrategies.has(strategy)) {
              state.enabledStrategies.delete(strategy);
            } else {
              state.enabledStrategies.add(strategy);
            }
          });
        },
        
        setConnectionStatus: (connected, latency) => {
          set((state) => {
            state.isConnected = connected;
            if (latency !== undefined) {
              state.connectionLatency = latency;
            }
            state.lastHeartbeat = Date.now();
          });
          
          mevEvents.emit('connection:status', { connected, latency });
        },
        
        updateNodeHealth: (health) => {
          set((state) => {
            state.nodeHealth = health;
          });
          
          mevEvents.emit('node:health', health);
        },
        
        reset: () => {
          set(initialState);
          mevEvents.removeAllListeners();
        }
      }))
    ),
    {
      name: 'mev-store'
    }
  )
);

// Selectors for optimized re-renders
export const selectArbitrageOpportunities = (state: MEVState) => 
  Array.from(state.arbitrageOpportunities.values())
    .sort((a, b) => b.timestamp - a.timestamp);

export const selectActiveOpportunities = (state: MEVState) =>
  Array.from(state.arbitrageOpportunities.values())
    .filter(o => o.status === 'pending' || o.status === 'executing');

export const selectRecentBundles = (state: MEVState) =>
  Array.from(state.jitoBundles.values())
    .sort((a, b) => b.timestamp - a.timestamp)
    .slice(0, 100);

export const selectLatencyStats = (state: MEVState) => {
  const recent = state.latencyMetrics.slice(-60);
  if (recent.length === 0) return null;
  
  const rpcLatencies = recent.map(m => m.rpcLatency);
  const quicLatencies = recent.map(m => m.quicLatency);
  
  return {
    avgRpc: rpcLatencies.reduce((a, b) => a + b, 0) / rpcLatencies.length,
    avgQuic: quicLatencies.reduce((a, b) => a + b, 0) / quicLatencies.length,
    minRpc: Math.min(...rpcLatencies),
    maxRpc: Math.max(...rpcLatencies),
    minQuic: Math.min(...quicLatencies),
    maxQuic: Math.max(...quicLatencies)
  };
};

export const selectProfitTrend = (state: MEVState) => {
  return state.profitMetrics.cumulativeProfit.map((profit, i) => ({
    x: i,
    y: profit,
    timestamp: Date.now() - (state.profitMetrics.cumulativeProfit.length - i) * 60000
  }));
};

export const selectSystemHealth = (state: MEVState) => {
  const recent = state.systemPerformance.slice(-10);
  if (recent.length === 0) return 'unknown';
  
  const avgCpu = recent.reduce((sum, p) => 
    sum + p.cpuUsage.reduce((a, b) => a + b, 0) / p.cpuUsage.length, 0) / recent.length;
  const avgMemory = recent.reduce((sum, p) => sum + p.memoryUsage, 0) / recent.length;
  
  if (avgCpu > 90 || avgMemory > 90) return 'critical';
  if (avgCpu > 70 || avgMemory > 70) return 'warning';
  return 'healthy';
};