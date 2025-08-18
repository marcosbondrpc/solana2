export interface ConsensusMetrics {
  skipRate: number;
  votingState?: 'active' | 'delinquent' | string;
}

export interface PerformanceStageBanking {
  bufferedPackets: number;
  forwardedPackets: number;
  droppedPackets: number;
  threadsActive: number;
  processingTime?: number;
}

export interface PerformanceStageFetch {
  packetsReceived: number;
  packetsProcessed: number;
  latency: number;
}

export interface PerformanceStageVote {
  votesProcessed: number;
  voteLatency: number;
}

export interface PerformanceStageShred {
  shredsReceived: number;
  shredsInserted: number;
  shredLatency: number;
}

export interface PerformanceStageReplay {
  slotsReplayed: number;
  forkWeight: number;
  replayLatency: number;
}

export interface PerformanceMetrics {
  currentTPS: number;
  averageTPS: number;
  peakTPS: number;
  slotTime: number;
  confirmationTime: number;
  bankingStage: PerformanceStageBanking;
  fetchStage: PerformanceStageFetch;
  voteStage: PerformanceStageVote;
  shredStage: PerformanceStageShred;
  replayStage: PerformanceStageReplay;
}

export interface RPCLayerMetrics {
  endpoints: Record<string, { p99Latency: number }>;
}

export interface NetworkMetrics {
  [key: string]: any;
}

export interface OSMetrics {
  cpuTemperature: number[];
  memoryDetails: { used: number; free: number };
  [key: string]: any;
}

export interface JitoMEVMetrics {
  [key: string]: any;
}

export interface GeyserMetrics {
  [key: string]: any;
}

export interface SecurityMetrics {
  alerts?: Array<{ id: string; [key: string]: any }>;
  [key: string]: any;
}

export interface HealthScore {
  overall: number;
}