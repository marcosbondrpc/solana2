/**
 * Advanced Monitoring Store for Solana MEV System
 * Handles real-time metrics, alerts, and WebSocket state management
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { devtools } from 'zustand/middleware';

// Types for monitoring system
export interface ConsensusMetrics {
  votingState: 'voting' | 'delinquent' | 'inactive';
  lastVoteSlot: number;
  rootSlot: number;
  credits: number;
  epochCredits: number;
  commission: number;
  skipRate: number;
  leaderSlots: number;
  blocksProduced: number;
  slotsSkipped?: number;
  optimisticSlot: number;
  optimisticConfirmationTime: number;
  towerHeight: number;
  towerRoot: number;
  validatorActiveStake: bigint;
  validatorDelinquentStake: bigint;
  // Optional legacy/aux fields
  slot?: number;
  blockHeight?: number;
  leaderSchedule?: string[];
  epochProgress?: number;
  voteCredits?: number;
}

export interface PerformanceMetrics {
  currentTPS: number;
  peakTPS: number;
  averageTPS: number;
  slotTime: number;
  confirmationTime: number;
  bankingStage: {
    bufferedPackets: number;
    forwardedPackets: number;
    droppedPackets: number;
    processingTime: number;
    threadsActive: number;
  };
  fetchStage: {
    packetsReceived: number;
    packetsProcessed: number;
    latency: number;
  };
  voteStage: {
    votesProcessed: number;
    voteLatency: number;
  };
  shredStage: {
    shredsReceived: number;
    shredsInserted: number;
    shredLatency: number;
  };
  replayStage: {
    slotsReplayed: number;
    forkWeight: number;
    replayLatency: number;
  };
}

export interface RPCLayerMetrics {
  endpoints: Record<string, {
    requests: number;
    errors: number;
    avgLatency: number;
    p50Latency: number;
    p95Latency: number;
    p99Latency: number;
  }>;
  totalRequests: number;
  totalErrors: number;
  cacheHitRate: number;
  wsConnections: number;
}

export interface NetworkMetrics {
  peerCount: number;
  inboundConnections: number;
  outboundConnections: number;
  bandwidth: {
    inbound: number;
    outbound: number;
  };
  gossipMessages: number;
  repairRequests: number;
  turbineLatency: number;
}

export interface OSMetrics {
  cpuUsage: number[];
  cpuTemperature: number[];
  memoryDetails: {
    used: number;
    free: number;
    cached: number;
    buffers: number;
  };
  diskIO: {
    readBytes: number;
    writeBytes: number;
    readOps: number;
    writeOps: number;
  };
  networkIO: {
    rxBytes: number;
    txBytes: number;
    rxPackets: number;
    txPackets: number;
  };
  processStats: {
    threads: number;
    openFiles: number;
    contextSwitches: number;
  };
}

export interface JitoMEVMetrics {
  bundlesReceived: number;
  bundlesForwarded: number;
  bundlesLanded: number;
  totalTips: number;
  avgTipAmount: number;
  blockEngineStatus: 'connected' | 'disconnected';
  relayStatus: 'connected' | 'disconnected';
  searchers: number;
  profitTracker: {
    daily: number;
    weekly: number;
    monthly: number;
  };
}

export interface GeyserMetrics {
  accountUpdates: number;
  slotUpdates: number;
  blockUpdates: number;
  txUpdates: number;
  queueDepth: number;
  processingLatency: number;
  droppedMessages: number;
}

export interface SecurityMetrics {
  firewallStatus: 'active' | 'inactive';
  ddosProtection: boolean;
  sshAttempts: number;
  suspiciousActivity: number;
  openPorts: number[];
  sslCertExpiry: number;
  alerts: SecurityAlert[];
}

export interface SecurityAlert {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  type: string;
  message: string;
  timestamp: Date;
  source: string;
}

export interface HealthScore {
  overall: number;
  components: {
    consensus: number;
    performance: number;
    rpc: number;
    network: number;
    system: number;
    jito: number;
    geyser: number;
    security: number;
  };
  issues: string[];
  recommendations: string[];
}

export interface Alert {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  type: string;
  message: string;
  timestamp: Date;
  acknowledged: boolean;
  source?: string;
  metadata?: Record<string, any>;
}

export interface AlertConfig {
  enabled: boolean;
  emailNotifications: boolean;
  discordWebhook?: string;
  slackWebhook?: string;
  thresholds: {
    skipRate: number;
    temperature: number;
    memoryUsage: number;
    diskUsage: number;
    rpcLatency: number;
    packetDropRate: number;
  };
}

export interface MonitoringConfig {
  updateInterval: number;
  maxHistoricalPoints: number;
  features: {
    consensus: boolean;
    performance: boolean;
    rpc: boolean;
    network: boolean;
    os: boolean;
    jito: boolean;
    geyser: boolean;
    security: boolean;
    notifications: boolean;
  };
}

interface WSMessage {
  event: string;
  data: any;
  timestamp: number;
}

interface HistoricalData<T> {
  data: Array<{ timestamp: number; value: T }>;
  maxPoints: number;
}

interface MonitoringState {
  // Connection management
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error';
  authToken?: string;
  wsQueue: WSMessage[];
  
  // Real-time metrics
  consensus: ConsensusMetrics | null;
  performance: PerformanceMetrics | null;
  rpcLayer: RPCLayerMetrics | null;
  network: NetworkMetrics | null;
  os: OSMetrics | null;
  jito: JitoMEVMetrics | null;
  geyser: GeyserMetrics | null;
  security: SecurityMetrics | null;
  health: HealthScore | null;
  
  // Historical data (for charts)
  historicalData: Map<string, any[]>;
  consensusHistory: HistoricalData<ConsensusMetrics>;
  performanceHistory: HistoricalData<PerformanceMetrics>;
  
  // Alerts & notifications
  activeAlerts: Alert[];
  alertHistory: Alert[];
  alertConfig: AlertConfig;
  
  // Configuration
  config: MonitoringConfig;
  
  // Actions
  setConnectionStatus: (status: MonitoringState['connectionStatus']) => void;
  setAuthToken: (token: string) => void;
  queueWSMessage: (event: string, data: any) => void;
  processWSQueue: () => void;
  
  // Metric updates
  updateConsensus: (metrics: ConsensusMetrics) => void;
  updatePerformance: (metrics: PerformanceMetrics) => void;
  updateRPCLayer: (metrics: RPCLayerMetrics) => void;
  updateNetwork: (metrics: NetworkMetrics) => void;
  updateOS: (metrics: OSMetrics) => void;
  updateJito: (metrics: JitoMEVMetrics) => void;
  updateGeyser: (metrics: GeyserMetrics) => void;
  updateSecurity: (metrics: SecurityMetrics) => void;
  updateHealth: (health: HealthScore) => void;
  
  // Historical data management
  addHistoricalPoint: (key: string, data: any) => void;
  clearHistoricalData: (key?: string) => void;
  
  // Alert management
  addAlert: (alert: Alert) => void;
  acknowledgeAlert: (id: string) => void;
  clearAlert: (id: string) => void;
  clearAllAlerts: () => void;
  updateAlertConfig: (config: Partial<AlertConfig>) => void;
  
  // Configuration
  updateConfig: (config: Partial<MonitoringConfig>) => void;
  
  // Batch updates for performance
  batchUpdate: (updates: (() => void)[]) => void;
}

export const useMonitoringStore = create<MonitoringState>()(
  subscribeWithSelector(
    devtools(
      immer((set, get) => ({
        // Initial state
        connectionStatus: 'disconnected',
        authToken: undefined,
        wsQueue: [],
        
        consensus: null,
        performance: null,
        rpcLayer: null,
        network: null,
        os: null,
        jito: null,
        geyser: null,
        security: null,
        health: null,
        
        historicalData: new Map(),
        
        activeAlerts: [],
        alertHistory: [],
        alertConfig: {
          enabled: true,
          emailNotifications: false,
          thresholds: {
            skipRate: 10,
            temperature: 85,
            memoryUsage: 90,
            diskUsage: 95,
            rpcLatency: 1000,
            packetDropRate: 5,
          },
        },
        
        config: {
          updateInterval: 1000,
          maxHistoricalPoints: 300,
          features: {
            consensus: true,
            performance: true,
            rpc: true,
            network: true,
            os: true,
            jito: true,
            geyser: true,
            security: true,
            notifications: true,
          },
        },
        
        // Actions
        setConnectionStatus: (status) => set((state) => {
          state.connectionStatus = status;
        }),
        
        setAuthToken: (token) => set((state) => {
          state.authToken = token;
        }),
        
        queueWSMessage: (event, data) => set((state) => {
          state.wsQueue.push({
            event,
            data,
            timestamp: Date.now(),
          });
          
          // Limit queue size
          if (state.wsQueue.length > 100) {
            state.wsQueue.shift();
          }
        }),
        
        processWSQueue: () => set((state) => {
          state.wsQueue = [];
        }),
        
        // Metric updates
        updateConsensus: (metrics) => set((state) => {
          state.consensus = metrics;
        }),
        
        updatePerformance: (metrics) => set((state) => {
          state.performance = metrics;
        }),
        
        updateRPCLayer: (metrics) => set((state) => {
          state.rpcLayer = metrics;
        }),
        
        updateNetwork: (metrics) => set((state) => {
          state.network = metrics;
        }),
        
        updateOS: (metrics) => set((state) => {
          state.os = metrics;
        }),
        
        updateJito: (metrics) => set((state) => {
          state.jito = metrics;
        }),
        
        updateGeyser: (metrics) => set((state) => {
          state.geyser = metrics;
        }),
        
        updateSecurity: (metrics) => set((state) => {
          state.security = metrics;
        }),
        
        updateHealth: (health) => set((state) => {
          state.health = health;
        }),
        
        // Historical data management
        addHistoricalPoint: (key, data) => set((state) => {
          const history = state.historicalData.get(key) || [];
          history.push({
            ...data,
            timestamp: Date.now(),
          });
          
          // Limit history size
          const maxPoints = state.config.maxHistoricalPoints;
          if (history.length > maxPoints) {
            history.splice(0, history.length - maxPoints);
          }
          
          state.historicalData.set(key, history);
        }),
        
        clearHistoricalData: (key) => set((state) => {
          if (key) {
            state.historicalData.delete(key);
          } else {
            state.historicalData.clear();
          }
        }),
        
        // Alert management
        addAlert: (alert) => set((state) => {
          // Prevent duplicate alerts
          if (!state.activeAlerts.find(a => a.id === alert.id)) {
            state.activeAlerts.push(alert);
            state.alertHistory.push(alert);
            
            // Limit alert history
            if (state.alertHistory.length > 1000) {
              state.alertHistory.shift();
            }
          }
        }),
        
        acknowledgeAlert: (id) => set((state) => {
          const alert = state.activeAlerts.find(a => a.id === id);
          if (alert) {
            alert.acknowledged = true;
          }
        }),
        
        clearAlert: (id) => set((state) => {
          state.activeAlerts = state.activeAlerts.filter(a => a.id !== id);
        }),
        
        clearAllAlerts: () => set((state) => {
          state.activeAlerts = [];
        }),
        
        updateAlertConfig: (config) => set((state) => {
          state.alertConfig = { ...state.alertConfig, ...config };
        }),
        
        // Configuration
        updateConfig: (config) => set((state) => {
          state.config = { ...state.config, ...config };
        }),
        
        // Batch updates
        batchUpdate: (updates) => set((state) => {
          updates.forEach(update => update());
        }),
      }))
    )
  )
);

// Selectors for optimized re-renders
export const selectConnectionStatus = (state: MonitoringState) => state.connectionStatus;
export const selectConsensus = (state: MonitoringState) => state.consensus;
export const selectPerformance = (state: MonitoringState) => state.performance;
export const selectRPCLayer = (state: MonitoringState) => state.rpcLayer;
export const selectNetwork = (state: MonitoringState) => state.network;
export const selectOS = (state: MonitoringState) => state.os;
export const selectJito = (state: MonitoringState) => state.jito;
export const selectGeyser = (state: MonitoringState) => state.geyser;
export const selectSecurity = (state: MonitoringState) => state.security;
export const selectHealth = (state: MonitoringState) => state.health;
export const selectActiveAlerts = (state: MonitoringState) => state.activeAlerts;
export const selectCriticalAlerts = (state: MonitoringState) => 
  state.activeAlerts.filter(a => a.severity === 'critical' && !a.acknowledged);