import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

export interface NodeMetrics {
  slot: number;
  blockHeight: number;
  epoch: number;
  slotIndex: number;
  slotsInEpoch: number;
  absoluteSlot: number;
  transactionCount: number;
  skipRate: number;
  leaderSlots: number;
  blocksProduced: number;
  health: 'healthy' | 'warning' | 'error' | 'unknown';
  version: string;
  identity: string;
  voteAccount: string;
  commission: number;
  activatedStake: number;
  networkInflation: number;
  totalSupply: number;
}

export interface SystemMetrics {
  cpuUsage: number;
  cpuCores: number;
  cpuFreq: number;
  memoryUsed: number;
  memoryTotal: number;
  memoryPercent: number;
  diskUsed: number;
  diskTotal: number;
  diskPercent: number;
  networkRx: number;
  networkTx: number;
  loadAverage: [number, number, number];
  uptime: number;
  processes: number;
}

export interface JitoMetrics {
  bundlesReceived: number;
  bundlesLanded: number;
  bundleRate: number;
  mevRewards: number;
  tipDistribution: number[];
  avgBundleLatency: number;
  blockEngineConnected: boolean;
  relayerConnected: boolean;
}

export interface RPCMetrics {
  requestsPerSecond: number;
  avgResponseTime: number;
  p99ResponseTime: number;
  p95ResponseTime: number;
  errorRate: number;
  activeConnections: number;
  wsConnections: number;
  httpConnections: number;
  methods: Record<string, {
    count: number;
    avgTime: number;
    errors: number;
  }>;
}

export interface LogEntry {
  timestamp: Date;
  level: 'info' | 'warn' | 'error' | 'debug';
  message: string;
  source: string;
}

interface NodeState {
  // Connection
  isConnected: boolean;
  network: 'mainnet-beta' | 'testnet' | 'devnet';
  
  // Metrics
  nodeMetrics: NodeMetrics | null;
  systemMetrics: SystemMetrics | null;
  jitoMetrics: JitoMetrics | null;
  rpcMetrics: RPCMetrics | null;
  
  // Historical data
  slotHistory: Array<{ time: Date; slot: number }>;
  cpuHistory: Array<{ time: Date; usage: number }>;
  memoryHistory: Array<{ time: Date; usage: number }>;
  networkHistory: Array<{ time: Date; rx: number; tx: number }>;
  tpsHistory: Array<{ time: Date; tps: number }>;
  
  // Logs
  logs: LogEntry[];
  maxLogs: number;
  
  // Actions
  setConnected: (connected: boolean) => void;
  setNetwork: (network: 'mainnet-beta' | 'testnet' | 'devnet') => void;
  updateNodeMetrics: (metrics: Partial<NodeMetrics>) => void;
  updateSystemMetrics: (metrics: Partial<SystemMetrics>) => void;
  updateJitoMetrics: (metrics: Partial<JitoMetrics>) => void;
  updateRPCMetrics: (metrics: Partial<RPCMetrics>) => void;
  addLog: (log: LogEntry) => void;
  clearLogs: () => void;
  
  // History updates
  addSlotData: (slot: number) => void;
  addCpuData: (usage: number) => void;
  addMemoryData: (usage: number) => void;
  addNetworkData: (rx: number, tx: number) => void;
  addTpsData: (tps: number) => void;
}

const MAX_HISTORY_POINTS = 100;

export const useNodeStore = create<NodeState>()(
  subscribeWithSelector((set) => ({
    // Initial state
    isConnected: false,
    network: 'mainnet-beta',
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
    
    // Actions
    setConnected: (connected) => set({ isConnected: connected }),
    
    setNetwork: (network) => set({ network }),
    
    updateNodeMetrics: (metrics) =>
      set((state) => ({
        nodeMetrics: state.nodeMetrics
          ? { ...state.nodeMetrics, ...metrics }
          : (metrics as NodeMetrics),
      })),
    
    updateSystemMetrics: (metrics) =>
      set((state) => ({
        systemMetrics: state.systemMetrics
          ? { ...state.systemMetrics, ...metrics }
          : (metrics as SystemMetrics),
      })),
    
    updateJitoMetrics: (metrics) =>
      set((state) => ({
        jitoMetrics: state.jitoMetrics
          ? { ...state.jitoMetrics, ...metrics }
          : (metrics as JitoMetrics),
      })),
    
    updateRPCMetrics: (metrics) =>
      set((state) => ({
        rpcMetrics: state.rpcMetrics
          ? { ...state.rpcMetrics, ...metrics }
          : (metrics as RPCMetrics),
      })),
    
    addLog: (log) =>
      set((state) => ({
        logs: [log, ...state.logs].slice(0, state.maxLogs),
      })),
    
    clearLogs: () => set({ logs: [] }),
    
    addSlotData: (slot) =>
      set((state) => ({
        slotHistory: [
          ...state.slotHistory,
          { time: new Date(), slot },
        ].slice(-MAX_HISTORY_POINTS),
      })),
    
    addCpuData: (usage) =>
      set((state) => ({
        cpuHistory: [
          ...state.cpuHistory,
          { time: new Date(), usage },
        ].slice(-MAX_HISTORY_POINTS),
      })),
    
    addMemoryData: (usage) =>
      set((state) => ({
        memoryHistory: [
          ...state.memoryHistory,
          { time: new Date(), usage },
        ].slice(-MAX_HISTORY_POINTS),
      })),
    
    addNetworkData: (rx, tx) =>
      set((state) => ({
        networkHistory: [
          ...state.networkHistory,
          { time: new Date(), rx, tx },
        ].slice(-MAX_HISTORY_POINTS),
      })),
    
    addTpsData: (tps) =>
      set((state) => ({
        tpsHistory: [
          ...state.tpsHistory,
          { time: new Date(), tps },
        ].slice(-MAX_HISTORY_POINTS),
      })),
  }))
);