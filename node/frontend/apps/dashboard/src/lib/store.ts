import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { devtools } from 'zustand/middleware';

export interface NodeMetrics {
  slot: number;
  blockHeight: number;
  transactionCount: number;
  slotTime: number;
  leaderSchedule: string[];
  tps: number;
  skipRate: number;
  timestamp: number;
}

export interface SystemMetrics {
  cpuUsage: number;
  cpuTemp: number;
  memoryUsed: number;
  memoryTotal: number;
  memoryPercent: number;
  diskUsed: number;
  diskTotal: number;
  diskPercent: number;
  networkRx: number;
  networkTx: number;
  processCount: number;
  loadAverage: number[];
  uptime: number;
  timestamp: number;
}

export interface JitoMetrics {
  bundlesReceived: number;
  bundlesForwarded: number;
  bundlesLanded: number;
  bundlesPending: number;
  bundlesFailed: number;
  bundleSuccessRate: number;
  avgBundleSize: number;
  totalTips: number;
  avgTip: number;
  blockEngineConnected: boolean;
  relayConnected: boolean;
  mempoolSize: number;
  timestamp: number;
}

export interface RPCMetrics {
  requestsPerSecond: number;
  avgResponseTime: number;
  p50ResponseTime: number;
  p95ResponseTime: number;
  p99ResponseTime: number;
  errorRate: number;
  activeConnections: number;
  totalRequests: number;
  totalErrors: number;
  requestsByMethod: Record<string, number>;
  healthScore: number;
  timestamp: number;
}

export interface LogEntry {
  timestamp: Date;
  level: 'info' | 'warn' | 'error' | 'debug';
  message: string;
  source: string;
  metadata?: Record<string, any>;
}

export interface HistoricalDataPoint {
  timestamp: number;
  value: number;
}

export interface NetworkData {
  timestamp: number;
  rx: number;
  tx: number;
}

interface NodeState {
  // Connection status
  connected: boolean;
  connectionError: string | null;
  lastHeartbeat: number;
  
  // Real-time metrics
  nodeMetrics: NodeMetrics | null;
  systemMetrics: SystemMetrics | null;
  jitoMetrics: JitoMetrics | null;
  rpcMetrics: RPCMetrics | null;
  
  // Historical data for charts (limited to last 100 points for performance)
  slotHistory: HistoricalDataPoint[];
  cpuHistory: HistoricalDataPoint[];
  memoryHistory: HistoricalDataPoint[];
  networkHistory: NetworkData[];
  tpsHistory: HistoricalDataPoint[];
  
  // Logs
  logs: LogEntry[];
  maxLogs: number;
  
  // Settings
  updateInterval: number;
  enableNotifications: boolean;
  alertThresholds: {
    cpu: number;
    memory: number;
    diskSpace: number;
    errorRate: number;
  };
  
  // Actions
  setConnected: (connected: boolean) => void;
  setConnectionError: (error: string | null) => void;
  updateNodeMetrics: (metrics: Partial<NodeMetrics>) => void;
  updateSystemMetrics: (metrics: Partial<SystemMetrics>) => void;
  updateJitoMetrics: (metrics: Partial<JitoMetrics>) => void;
  updateRPCMetrics: (metrics: Partial<RPCMetrics>) => void;
  addLog: (log: Omit<LogEntry, 'timestamp'> | LogEntry) => void;
  clearLogs: () => void;
  addSlotData: (slot: number) => void;
  addCpuData: (usage: number) => void;
  addMemoryData: (usage: number) => void;
  addNetworkData: (rx: number, tx: number) => void;
  addTpsData: (tps: number) => void;
  updateSettings: (settings: Partial<NodeState['alertThresholds']>) => void;
  clearHistoricalData: () => void;
  reset: () => void;
}

const MAX_HISTORY_POINTS = 100;

const initialState = {
  connected: false,
  connectionError: null,
  lastHeartbeat: Date.now(),
  nodeMetrics: null,
  systemMetrics: null,
  jitoMetrics: null,
  rpcMetrics: null,
  slotHistory: [],
  cpuHistory: [],
  memoryHistory: [],
  networkHistory: [],
  tpsHistory: [],
  logs: [],
  maxLogs: 1000,
  updateInterval: 1000,
  enableNotifications: true,
  alertThresholds: {
    cpu: 90,
    memory: 90,
    diskSpace: 90,
    errorRate: 5,
  },
};

export const useNodeStore = create<NodeState>()(
  devtools(
    subscribeWithSelector(
      immer((set, get) => ({
        ...initialState,
        
        setConnected: (connected) => {
          set((state) => {
            state.connected = connected;
            state.lastHeartbeat = Date.now();
            if (!connected) {
              state.connectionError = 'Connection lost';
            } else {
              state.connectionError = null;
            }
          });
        },
        
        setConnectionError: (error) => {
          set((state) => {
            state.connectionError = error;
            if (error) {
              state.connected = false;
            }
          });
        },
        
        updateNodeMetrics: (metrics) => {
          set((state) => {
            if (!state.nodeMetrics) {
              state.nodeMetrics = {
                slot: 0,
                blockHeight: 0,
                transactionCount: 0,
                slotTime: 0,
                leaderSchedule: [],
                tps: 0,
                skipRate: 0,
                timestamp: Date.now(),
              };
            }
            Object.assign(state.nodeMetrics, metrics);
            state.nodeMetrics.timestamp = Date.now();
            state.lastHeartbeat = Date.now();
          });
        },
        
        updateSystemMetrics: (metrics) => {
          set((state) => {
            if (!state.systemMetrics) {
              state.systemMetrics = {
                cpuUsage: 0,
                cpuTemp: 0,
                memoryUsed: 0,
                memoryTotal: 0,
                memoryPercent: 0,
                diskUsed: 0,
                diskTotal: 0,
                diskPercent: 0,
                networkRx: 0,
                networkTx: 0,
                processCount: 0,
                loadAverage: [0, 0, 0],
                uptime: 0,
                timestamp: Date.now(),
              };
            }
            Object.assign(state.systemMetrics, metrics);
            state.systemMetrics.timestamp = Date.now();
          });
        },
        
        updateJitoMetrics: (metrics) => {
          set((state) => {
            if (!state.jitoMetrics) {
              state.jitoMetrics = {
                bundlesReceived: 0,
                bundlesForwarded: 0,
                bundlesLanded: 0,
                bundlesPending: 0,
                bundlesFailed: 0,
                bundleSuccessRate: 0,
                avgBundleSize: 0,
                totalTips: 0,
                avgTip: 0,
                blockEngineConnected: false,
                relayConnected: false,
                mempoolSize: 0,
                timestamp: Date.now(),
              };
            }
            Object.assign(state.jitoMetrics, metrics);
            state.jitoMetrics.timestamp = Date.now();
          });
        },
        
        updateRPCMetrics: (metrics) => {
          set((state) => {
            if (!state.rpcMetrics) {
              state.rpcMetrics = {
                requestsPerSecond: 0,
                avgResponseTime: 0,
                p50ResponseTime: 0,
                p95ResponseTime: 0,
                p99ResponseTime: 0,
                errorRate: 0,
                activeConnections: 0,
                totalRequests: 0,
                totalErrors: 0,
                requestsByMethod: {},
                healthScore: 100,
                timestamp: Date.now(),
              };
            }
            Object.assign(state.rpcMetrics, metrics);
            state.rpcMetrics.timestamp = Date.now();
          });
        },
        
        addLog: (log) => {
          set((state) => {
            const incoming: any = log as any;
            const logEntry: LogEntry = {
              ...(incoming as any),
              timestamp: incoming?.timestamp instanceof Date ? incoming.timestamp : new Date(),
            };
            
            state.logs.unshift(logEntry);
            
            if (state.logs.length > state.maxLogs) {
              state.logs = state.logs.slice(0, state.maxLogs);
            }
          });
        },
        
        clearLogs: () => {
          set((state) => {
            state.logs = [];
          });
        },
        
        addSlotData: (slot) => {
          set((state) => {
            const dataPoint = { timestamp: Date.now(), value: slot };
            state.slotHistory.push(dataPoint);
            
            if (state.slotHistory.length > MAX_HISTORY_POINTS) {
              state.slotHistory.shift();
            }
          });
        },
        
        addCpuData: (usage) => {
          set((state) => {
            const dataPoint = { timestamp: Date.now(), value: usage };
            state.cpuHistory.push(dataPoint);
            
            if (state.cpuHistory.length > MAX_HISTORY_POINTS) {
              state.cpuHistory.shift();
            }
            
            // Check alert threshold
            if (usage > state.alertThresholds.cpu && state.enableNotifications) {
              state.logs.unshift({
                timestamp: new Date(),
                level: 'warn',
                message: `High CPU usage detected: ${usage.toFixed(1)}%`,
                source: 'system',
                metadata: { cpuUsage: usage },
              });
            }
          });
        },
        
        addMemoryData: (usage) => {
          set((state) => {
            const dataPoint = { timestamp: Date.now(), value: usage };
            state.memoryHistory.push(dataPoint);
            
            if (state.memoryHistory.length > MAX_HISTORY_POINTS) {
              state.memoryHistory.shift();
            }
            
            // Check alert threshold
            if (usage > state.alertThresholds.memory && state.enableNotifications) {
              state.logs.unshift({
                timestamp: new Date(),
                level: 'warn',
                message: `High memory usage detected: ${usage.toFixed(1)}%`,
                source: 'system',
                metadata: { memoryUsage: usage },
              });
            }
          });
        },
        
        addNetworkData: (rx, tx) => {
          set((state) => {
            const dataPoint = { timestamp: Date.now(), rx, tx };
            state.networkHistory.push(dataPoint);
            
            if (state.networkHistory.length > MAX_HISTORY_POINTS) {
              state.networkHistory.shift();
            }
          });
        },
        
        addTpsData: (tps) => {
          set((state) => {
            const dataPoint = { timestamp: Date.now(), value: tps };
            state.tpsHistory.push(dataPoint);
            
            if (state.tpsHistory.length > MAX_HISTORY_POINTS) {
              state.tpsHistory.shift();
            }
          });
        },
        
        updateSettings: (settings) => {
          set((state) => {
            if (settings) {
              Object.assign(state.alertThresholds, settings);
            }
          });
        },
        
        clearHistoricalData: () => {
          set((state) => {
            state.slotHistory = [];
            state.cpuHistory = [];
            state.memoryHistory = [];
            state.networkHistory = [];
            state.tpsHistory = [];
          });
        },
        
        reset: () => {
          set(initialState);
        },
      }))
    ),
    {
      name: 'node-store',
    }
  )
);

// Selectors for optimized re-renders
export const selectNodeHealth = (state: NodeState) => {
  if (!state.connected) return 'disconnected';
  if (!state.nodeMetrics || !state.systemMetrics) return 'unknown';
  
  const timeSinceLastUpdate = Date.now() - state.lastHeartbeat;
  if (timeSinceLastUpdate > 30000) return 'stale';
  
  const cpu = state.systemMetrics.cpuUsage;
  const memory = state.systemMetrics.memoryPercent;
  const errorRate = state.rpcMetrics?.errorRate || 0;
  
  if (cpu > 90 || memory > 90 || errorRate > 10) return 'critical';
  if (cpu > 70 || memory > 70 || errorRate > 5) return 'warning';
  
  return 'healthy';
};

export const selectRecentLogs = (state: NodeState, level?: LogEntry['level']) => {
  if (level) {
    return state.logs.filter(log => log.level === level).slice(0, 100);
  }
  return state.logs.slice(0, 100);
};

export const selectMetricsSummary = (state: NodeState) => ({
  node: {
    slot: state.nodeMetrics?.slot || 0,
    tps: state.nodeMetrics?.tps || 0,
    skipRate: state.nodeMetrics?.skipRate || 0,
  },
  system: {
    cpu: state.systemMetrics?.cpuUsage || 0,
    memory: state.systemMetrics?.memoryPercent || 0,
    disk: state.systemMetrics?.diskPercent || 0,
  },
  jito: {
    successRate: state.jitoMetrics?.bundleSuccessRate || 0,
    totalTips: state.jitoMetrics?.totalTips || 0,
    connected: state.jitoMetrics?.blockEngineConnected || false,
  },
  rpc: {
    rps: state.rpcMetrics?.requestsPerSecond || 0,
    errorRate: state.rpcMetrics?.errorRate || 0,
    healthScore: state.rpcMetrics?.healthScore || 0,
  },
});

export const selectChartData = (state: NodeState) => ({
  slot: state.slotHistory,
  cpu: state.cpuHistory,
  memory: state.memoryHistory,
  network: state.networkHistory,
  tps: state.tpsHistory,
});