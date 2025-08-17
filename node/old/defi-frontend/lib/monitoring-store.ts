import { create } from 'zustand';
import { subscribeWithSelector, devtools } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import {
  ConsensusMetrics,
  PerformanceMetrics,
  RPCLayerMetrics,
  NetworkMetrics,
  OSMetrics,
  JitoMEVMetrics,
  GeyserMetrics,
  SecurityMetrics,
  HealthScore,
  AlertConfig,
  User,
  MonitoringSnapshot,
  DashboardConfig,
} from '@/types/monitoring';

interface HistoricalData<T> {
  data: Array<{ timestamp: Date; value: T }>;
  maxPoints: number;
}

interface MonitoringState {
  // Connection State
  connectionStatus: 'connected' | 'connecting' | 'disconnected' | 'error';
  lastHeartbeat: Date | null;
  reconnectAttempts: number;
  
  // Real-time Metrics
  consensus: ConsensusMetrics | null;
  performance: PerformanceMetrics | null;
  rpcLayer: RPCLayerMetrics | null;
  network: NetworkMetrics | null;
  os: OSMetrics | null;
  jito: JitoMEVMetrics | null;
  geyser: GeyserMetrics | null;
  security: SecurityMetrics | null;
  health: HealthScore | null;
  
  // Historical Data
  consensusHistory: HistoricalData<Partial<ConsensusMetrics>>;
  performanceHistory: HistoricalData<Partial<PerformanceMetrics>>;
  rpcHistory: HistoricalData<Partial<RPCLayerMetrics>>;
  networkHistory: HistoricalData<Partial<NetworkMetrics>>;
  osHistory: HistoricalData<Partial<OSMetrics>>;
  jitoHistory: HistoricalData<Partial<JitoMEVMetrics>>;
  
  // Snapshots
  snapshots: MonitoringSnapshot[];
  maxSnapshots: number;
  
  // Alerts & Notifications
  activeAlerts: SecurityMetrics['alerts'];
  alertConfig: AlertConfig;
  unreadAlertCount: number;
  
  // User & Auth
  currentUser: User | null;
  isAuthenticated: boolean;
  authToken: string | null;
  
  // Dashboard Configuration
  config: DashboardConfig;
  
  // WebSocket Management
  wsMessageQueue: Array<{ type: string; payload: any; timestamp: Date }>;
  wsReconnectTimer: NodeJS.Timeout | null;
  
  // Performance Optimization
  renderOptimization: {
    skipNextUpdate: boolean;
    batchedUpdates: any[];
    updateThrottle: number;
  };
  
  // Actions
  setConnectionStatus: (status: MonitoringState['connectionStatus']) => void;
  updateConsensus: (metrics: Partial<ConsensusMetrics>) => void;
  updatePerformance: (metrics: Partial<PerformanceMetrics>) => void;
  updateRPCLayer: (metrics: Partial<RPCLayerMetrics>) => void;
  updateNetwork: (metrics: Partial<NetworkMetrics>) => void;
  updateOS: (metrics: Partial<OSMetrics>) => void;
  updateJito: (metrics: Partial<JitoMEVMetrics>) => void;
  updateGeyser: (metrics: Partial<GeyserMetrics>) => void;
  updateSecurity: (metrics: Partial<SecurityMetrics>) => void;
  updateHealth: (health: HealthScore) => void;
  
  // Historical Data Actions
  addHistoricalPoint: <T extends keyof MonitoringState>(
    metric: T,
    value: any
  ) => void;
  clearHistory: (metric?: keyof MonitoringState) => void;
  
  // Snapshot Actions
  createSnapshot: () => void;
  deleteSnapshot: (timestamp: Date) => void;
  exportSnapshots: () => Promise<void>;
  
  // Alert Actions
  addAlert: (alert: SecurityMetrics['alerts'][0]) => void;
  acknowledgeAlert: (alertId: string) => void;
  clearAlerts: () => void;
  updateAlertConfig: (config: Partial<AlertConfig>) => void;
  
  // Auth Actions
  login: (user: User, token: string) => void;
  logout: () => void;
  updateUser: (user: Partial<User>) => void;
  
  // Config Actions
  updateConfig: (config: Partial<DashboardConfig>) => void;
  resetConfig: () => void;
  
  // WebSocket Actions
  queueWSMessage: (type: string, payload: any) => void;
  processWSQueue: () => void;
  clearWSQueue: () => void;
  
  // Performance Actions
  enableRenderOptimization: () => void;
  disableRenderOptimization: () => void;
  batchUpdate: (updates: Array<() => void>) => void;
}

const DEFAULT_CONFIG: DashboardConfig = {
  refreshIntervals: {
    consensus: 5000,
    performance: 2000,
    rpc: 3000,
    network: 5000,
    system: 2000,
    jito: 10000,
    geyser: 10000,
    security: 30000,
  },
  retention: {
    metrics: 86400000, // 24 hours
    logs: 604800000, // 7 days
    alerts: 2592000000, // 30 days
    snapshots: 7776000000, // 90 days
  },
  features: {
    darkMode: true,
    autoRefresh: true,
    notifications: true,
    advancedMode: false,
    exportData: true,
  },
};

const DEFAULT_ALERT_CONFIG: AlertConfig = {
  enabled: true,
  thresholds: {
    cpuUsage: 90,
    memoryUsage: 85,
    diskUsage: 90,
    skipRate: 5,
    delinquency: 10,
    rpcLatency: 1000,
    packetLoss: 1,
    temperature: 80,
  },
  channels: {
    email: false,
    slack: false,
    telegram: false,
    webhook: '',
  },
  cooldown: 300000, // 5 minutes
};

const MAX_HISTORY_POINTS = 1000;
const MAX_QUEUE_SIZE = 100;

export const useMonitoringStore = create<MonitoringState>()(
  devtools(
    immer(
      subscribeWithSelector((set, get) => ({
        // Initial State
        connectionStatus: 'disconnected',
        lastHeartbeat: null,
        reconnectAttempts: 0,
        
        consensus: null,
        performance: null,
        rpcLayer: null,
        network: null,
        os: null,
        jito: null,
        geyser: null,
        security: null,
        health: null,
        
        consensusHistory: { data: [], maxPoints: MAX_HISTORY_POINTS },
        performanceHistory: { data: [], maxPoints: MAX_HISTORY_POINTS },
        rpcHistory: { data: [], maxPoints: MAX_HISTORY_POINTS },
        networkHistory: { data: [], maxPoints: MAX_HISTORY_POINTS },
        osHistory: { data: [], maxPoints: MAX_HISTORY_POINTS },
        jitoHistory: { data: [], maxPoints: MAX_HISTORY_POINTS },
        
        snapshots: [],
        maxSnapshots: 100,
        
        activeAlerts: [],
        alertConfig: DEFAULT_ALERT_CONFIG,
        unreadAlertCount: 0,
        
        currentUser: null,
        isAuthenticated: false,
        authToken: null,
        
        config: DEFAULT_CONFIG,
        
        wsMessageQueue: [],
        wsReconnectTimer: null,
        
        renderOptimization: {
          skipNextUpdate: false,
          batchedUpdates: [],
          updateThrottle: 16, // ~60fps
        },
        
        // Actions
        setConnectionStatus: (status) =>
          set((state) => {
            state.connectionStatus = status;
            state.lastHeartbeat = status === 'connected' ? new Date() : state.lastHeartbeat;
            state.reconnectAttempts = status === 'connected' ? 0 : state.reconnectAttempts;
          }),
        
        updateConsensus: (metrics) =>
          set((state) => {
            if (state.renderOptimization.skipNextUpdate) return;
            state.consensus = state.consensus
              ? { ...state.consensus, ...metrics }
              : (metrics as ConsensusMetrics);
          }),
        
        updatePerformance: (metrics) =>
          set((state) => {
            if (state.renderOptimization.skipNextUpdate) return;
            state.performance = state.performance
              ? { ...state.performance, ...metrics }
              : (metrics as PerformanceMetrics);
          }),
        
        updateRPCLayer: (metrics) =>
          set((state) => {
            if (state.renderOptimization.skipNextUpdate) return;
            state.rpcLayer = state.rpcLayer
              ? { ...state.rpcLayer, ...metrics }
              : (metrics as RPCLayerMetrics);
          }),
        
        updateNetwork: (metrics) =>
          set((state) => {
            if (state.renderOptimization.skipNextUpdate) return;
            state.network = state.network
              ? { ...state.network, ...metrics }
              : (metrics as NetworkMetrics);
          }),
        
        updateOS: (metrics) =>
          set((state) => {
            if (state.renderOptimization.skipNextUpdate) return;
            state.os = state.os
              ? { ...state.os, ...metrics }
              : (metrics as OSMetrics);
          }),
        
        updateJito: (metrics) =>
          set((state) => {
            if (state.renderOptimization.skipNextUpdate) return;
            state.jito = state.jito
              ? { ...state.jito, ...metrics }
              : (metrics as JitoMEVMetrics);
          }),
        
        updateGeyser: (metrics) =>
          set((state) => {
            if (state.renderOptimization.skipNextUpdate) return;
            state.geyser = state.geyser
              ? { ...state.geyser, ...metrics }
              : (metrics as GeyserMetrics);
          }),
        
        updateSecurity: (metrics) =>
          set((state) => {
            if (state.renderOptimization.skipNextUpdate) return;
            state.security = state.security
              ? { ...state.security, ...metrics }
              : (metrics as SecurityMetrics);
          }),
        
        updateHealth: (health) =>
          set((state) => {
            state.health = health;
          }),
        
        addHistoricalPoint: (metric, value) =>
          set((state) => {
            const historyKey = `${metric}History` as keyof MonitoringState;
            const history = state[historyKey] as HistoricalData<any>;
            if (history && 'data' in history) {
              history.data.push({
                timestamp: new Date(),
                value,
              });
              if (history.data.length > history.maxPoints) {
                history.data.shift();
              }
            }
          }),
        
        clearHistory: (metric) =>
          set((state) => {
            if (metric) {
              const historyKey = `${metric}History` as keyof MonitoringState;
              const history = state[historyKey] as HistoricalData<any>;
              if (history && 'data' in history) {
                history.data = [];
              }
            } else {
              // Clear all history
              state.consensusHistory.data = [];
              state.performanceHistory.data = [];
              state.rpcHistory.data = [];
              state.networkHistory.data = [];
              state.osHistory.data = [];
              state.jitoHistory.data = [];
            }
          }),
        
        createSnapshot: () =>
          set((state) => {
            const snapshot: MonitoringSnapshot = {
              timestamp: new Date(),
              consensus: state.consensus!,
              performance: state.performance!,
              rpc: state.rpcLayer!,
              network: state.network!,
              os: state.os!,
              jito: state.jito!,
              geyser: state.geyser!,
              security: state.security!,
              health: state.health!,
            };
            state.snapshots.unshift(snapshot);
            if (state.snapshots.length > state.maxSnapshots) {
              state.snapshots.pop();
            }
          }),
        
        deleteSnapshot: (timestamp) =>
          set((state) => {
            state.snapshots = state.snapshots.filter(
              (s) => s.timestamp.getTime() !== timestamp.getTime()
            );
          }),
        
        exportSnapshots: async () => {
          const state = get();
          const data = JSON.stringify(state.snapshots, null, 2);
          const blob = new Blob([data], { type: 'application/json' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `solana-snapshots-${Date.now()}.json`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
        },
        
        addAlert: (alert) =>
          set((state) => {
            state.activeAlerts.unshift(alert);
            if (!alert.acknowledged) {
              state.unreadAlertCount++;
            }
            // Limit alerts
            if (state.activeAlerts.length > 1000) {
              state.activeAlerts = state.activeAlerts.slice(0, 1000);
            }
          }),
        
        acknowledgeAlert: (alertId) =>
          set((state) => {
            const alert = state.activeAlerts.find((a) => a.id === alertId);
            if (alert && !alert.acknowledged) {
              alert.acknowledged = true;
              state.unreadAlertCount = Math.max(0, state.unreadAlertCount - 1);
            }
          }),
        
        clearAlerts: () =>
          set((state) => {
            state.activeAlerts = [];
            state.unreadAlertCount = 0;
          }),
        
        updateAlertConfig: (config) =>
          set((state) => {
            state.alertConfig = { ...state.alertConfig, ...config };
          }),
        
        login: (user, token) =>
          set((state) => {
            state.currentUser = user;
            state.isAuthenticated = true;
            state.authToken = token;
            localStorage.setItem('auth_token', token);
          }),
        
        logout: () =>
          set((state) => {
            state.currentUser = null;
            state.isAuthenticated = false;
            state.authToken = null;
            localStorage.removeItem('auth_token');
          }),
        
        updateUser: (user) =>
          set((state) => {
            if (state.currentUser) {
              state.currentUser = { ...state.currentUser, ...user };
            }
          }),
        
        updateConfig: (config) =>
          set((state) => {
            state.config = { ...state.config, ...config };
            localStorage.setItem('dashboard_config', JSON.stringify(state.config));
          }),
        
        resetConfig: () =>
          set((state) => {
            state.config = DEFAULT_CONFIG;
            localStorage.removeItem('dashboard_config');
          }),
        
        queueWSMessage: (type, payload) =>
          set((state) => {
            state.wsMessageQueue.push({
              type,
              payload,
              timestamp: new Date(),
            });
            if (state.wsMessageQueue.length > MAX_QUEUE_SIZE) {
              state.wsMessageQueue.shift();
            }
          }),
        
        processWSQueue: () =>
          set((state) => {
            // Process all queued messages
            while (state.wsMessageQueue.length > 0) {
              const message = state.wsMessageQueue.shift();
              // Process message based on type
              if (message) {
                // Handle different message types
                console.log('Processing queued message:', message);
              }
            }
          }),
        
        clearWSQueue: () =>
          set((state) => {
            state.wsMessageQueue = [];
          }),
        
        enableRenderOptimization: () =>
          set((state) => {
            state.renderOptimization.skipNextUpdate = true;
          }),
        
        disableRenderOptimization: () =>
          set((state) => {
            state.renderOptimization.skipNextUpdate = false;
          }),
        
        batchUpdate: (updates) =>
          set((state) => {
            state.renderOptimization.skipNextUpdate = true;
            updates.forEach((update) => update());
            state.renderOptimization.skipNextUpdate = false;
          }),
      }))
    ),
    {
      name: 'solana-monitoring',
      trace: true,
    }
  )
);

// Performance optimization: Memoized selectors
export const selectHealthScore = (state: MonitoringState) => state.health?.overall ?? 0;
export const selectIsHealthy = (state: MonitoringState) => (state.health?.overall ?? 0) > 80;
export const selectCriticalAlerts = (state: MonitoringState) =>
  state.activeAlerts.filter((a) => a.severity === 'critical' && !a.acknowledged);
export const selectRecentSnapshots = (state: MonitoringState) =>
  state.snapshots.slice(0, 10);