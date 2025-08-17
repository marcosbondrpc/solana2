// Advanced Monitoring Types for Solana Node Dashboard

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
  slotsSkipped: number;
  optimisticSlot: number;
  optimisticConfirmationTime: number;
  towerHeight: number;
  towerRoot: number;
  validatorActiveStake: bigint;
  validatorDelinquentStake: bigint;
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
  endpoints: {
    [endpoint: string]: {
      requestCount: number;
      errorCount: number;
      avgLatency: number;
      p50Latency: number;
      p95Latency: number;
      p99Latency: number;
      lastError?: string;
      healthScore: number;
    };
  };
  rateLimits: {
    requestsPerSecond: number;
    burstLimit: number;
    currentUsage: number;
  };
  websocketStats: {
    activeConnections: number;
    subscriptions: number;
    messagesPerSecond: number;
    bytesPerSecond: number;
  };
  cacheStats: {
    hitRate: number;
    missRate: number;
    evictions: number;
    size: number;
  };
}

export interface NetworkMetrics {
  gossipNodes: number;
  tpuConnections: number;
  tvuConnections: number;
  repairConnections: number;
  turbineConnections: number;
  bandwidth: {
    inbound: number;
    outbound: number;
    gossipIn: number;
    gossipOut: number;
    tpuIn: number;
    tpuOut: number;
    repairIn: number;
    repairOut: number;
  };
  latencyMap: Map<string, number>;
  peerQuality: Map<string, number>;
  packetLoss: number;
  jitter: number;
}

export interface OSMetrics {
  kernelVersion: string;
  systemLoad: [number, number, number];
  cpuTemperature: number[];
  cpuFrequency: number[];
  cpuGovernor: string;
  memoryDetails: {
    used: number;
    free: number;
    cached: number;
    buffers: number;
    available: number;
    swapUsed: number;
    swapFree: number;
    hugepages: number;
  };
  diskIO: {
    readOps: number;
    writeOps: number;
    readBytes: number;
    writeBytes: number;
    avgQueueSize: number;
    utilization: number;
  };
  networkInterfaces: {
    [iface: string]: {
      rxPackets: number;
      txPackets: number;
      rxBytes: number;
      txBytes: number;
      rxErrors: number;
      txErrors: number;
      rxDropped: number;
      txDropped: number;
    };
  };
  processes: {
    total: number;
    running: number;
    sleeping: number;
    zombie: number;
  };
  fileDescriptors: {
    open: number;
    max: number;
  };
}

export interface JitoMEVMetrics {
  bundleStats: {
    received: number;
    accepted: number;
    rejected: number;
    landed: number;
    expired: number;
    successRate: number;
  };
  searchers: {
    active: number;
    total: number;
    topSearchers: Array<{
      address: string;
      bundlesSubmitted: number;
      successRate: number;
      avgTip: number;
    }>;
  };
  tips: {
    total: number;
    average: number;
    median: number;
    highest: number;
    distribution: number[];
  };
  blockEngine: {
    connected: boolean;
    latency: number;
    packetsForwarded: number;
    shredsSent: number;
    version: string;
  };
  relayer: {
    connected: boolean;
    regions: string[];
    latency: Map<string, number>;
  };
  profitability: {
    totalEarnings: number;
    dailyEarnings: number;
    weeklyEarnings: number;
    monthlyEarnings: number;
    averagePerBlock: number;
  };
}

export interface GeyserMetrics {
  plugins: Array<{
    name: string;
    version: string;
    status: 'running' | 'stopped' | 'error';
    accountsProcessed: number;
    transactionsProcessed: number;
    slotsProcessed: number;
    errorCount: number;
    avgProcessingTime: number;
    queueSize: number;
    memoryUsage: number;
  }>;
  streams: {
    active: number;
    total: number;
    bandwidth: number;
    messagesPerSecond: number;
  };
  database: {
    connected: boolean;
    latency: number;
    size: number;
    tables: number;
    connections: number;
    activeQueries: number;
  };
}

export interface SecurityMetrics {
  keyManagement: {
    validatorKeySecured: boolean;
    voteKeySecured: boolean;
    withdrawAuthority: string;
    lastRotation: Date;
    keyPermissions: string;
  };
  firewall: {
    enabled: boolean;
    rules: number;
    blockedAttempts: number;
    allowedPorts: number[];
  };
  ssl: {
    certificateValid: boolean;
    expiryDate: Date;
    tlsVersion: string;
  };
  auditLog: Array<{
    timestamp: Date;
    user: string;
    action: string;
    resource: string;
    result: 'success' | 'failure';
    ip: string;
  }>;
  alerts: Array<{
    id: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    type: string;
    message: string;
    timestamp: Date;
    acknowledged: boolean;
  }>;
}

export interface AlertConfig {
  enabled: boolean;
  thresholds: {
    cpuUsage: number;
    memoryUsage: number;
    diskUsage: number;
    skipRate: number;
    delinquency: number;
    rpcLatency: number;
    packetLoss: number;
    temperature: number;
  };
  channels: {
    email: boolean;
    slack: boolean;
    telegram: boolean;
    webhook: string;
  };
  cooldown: number;
}

export interface UserRole {
  id: string;
  name: string;
  permissions: Permission[];
  createdAt: Date;
  updatedAt: Date;
}

export interface Permission {
  resource: string;
  actions: ('read' | 'write' | 'execute' | 'delete')[];
}

export interface User {
  id: string;
  username: string;
  email: string;
  roles: UserRole[];
  lastLogin: Date;
  apiKey?: string;
  twoFactorEnabled: boolean;
}

export interface ValidatorControl {
  actions: {
    restart: boolean;
    stop: boolean;
    start: boolean;
    catchup: boolean;
    setIdentity: boolean;
    withdrawStake: boolean;
    updateCommission: boolean;
    rotateKeys: boolean;
  };
  safeguards: {
    confirmationRequired: boolean;
    twoFactorRequired: boolean;
    cooldownPeriod: number;
    maxSkipRate: number;
    minStake: number;
  };
}

export interface MonitoringSnapshot {
  timestamp: Date;
  consensus: ConsensusMetrics;
  performance: PerformanceMetrics;
  rpc: RPCLayerMetrics;
  network: NetworkMetrics;
  os: OSMetrics;
  jito: JitoMEVMetrics;
  geyser: GeyserMetrics;
  security: SecurityMetrics;
  health: HealthScore;
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

export interface DashboardConfig {
  refreshIntervals: {
    consensus: number;
    performance: number;
    rpc: number;
    network: number;
    system: number;
    jito: number;
    geyser: number;
    security: number;
  };
  retention: {
    metrics: number;
    logs: number;
    alerts: number;
    snapshots: number;
  };
  features: {
    darkMode: boolean;
    autoRefresh: boolean;
    notifications: boolean;
    advancedMode: boolean;
    exportData: boolean;
  };
}