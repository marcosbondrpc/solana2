/**
 * Ultimate MEV Control Plane API Service
 * High-performance service with sub-10ms response times
 * Handles billions in volume with microsecond precision
 */

import ky from 'ky';

// Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_TIMEOUT = 30000;
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000;

// Performance tracking
const perfMarks = new Map<string, number>();

// API Response Types
export interface ControlCommand {
  module: string;
  action: string;
  params: Record<string, any>;
  priority?: number;
  signature?: string;
}

export interface PolicyUpdate {
  policy_id: string;
  policy_type: string;
  thresholds: Record<string, number>;
  rules: Record<string, string>;
  enabled: boolean;
  effective_from?: number;
}

export interface ModelSwap {
  model_id: string;
  model_path: string;
  model_type: 'xgboost' | 'lightgbm' | 'tensorflow' | 'pytorch' | 'treelite';
  version: string;
  metadata: Record<string, string>;
}

export interface KillSwitch {
  target: string;
  reason: string;
  duration_ms: number;
  force?: boolean;
}

export interface MEVOpportunity {
  id: string;
  timestamp: number;
  type: 'arbitrage' | 'liquidation' | 'sandwich' | 'jit';
  dex_a: string;
  dex_b: string;
  token_pair: string;
  expected_profit: number;
  confidence: number;
  gas_estimate: number;
  slippage_tolerance: number;
  execution_window_ms: number;
  decision_dna?: string;
  bandit_arm?: string;
}

export interface JitoBundleSubmission {
  bundle_id: string;
  transactions: string[];
  tip_amount: number;
  target_slot?: number;
  relay_url: string;
  landed?: boolean;
  land_time_ms?: number;
}

export interface ThompsonSamplingMetrics {
  arms: Array<{
    id: string;
    route: string;
    alpha: number;
    beta: number;
    pulls: number;
    rewards: number;
    ucb_score: number;
    expected_value: number;
  }>;
  exploration_rate: number;
  exploitation_rate: number;
  total_pulls: number;
  convergence_metric: number;
  regret: number;
}

export interface Blake3Metrics {
  total_hashes: number;
  unique_hashes: number;
  collision_rate: number;
  dedup_savings_bytes: number;
  dedup_ratio: number;
  hash_rate_per_sec: number;
  merkle_root?: string;
}

export interface ClickHouseQuery {
  query: string;
  format?: 'JSON' | 'CSV' | 'TSV' | 'Parquet' | 'Arrow';
  database?: string;
  timeout_ms?: number;
  max_rows?: number;
  settings?: Record<string, any>;
}

export interface KafkaStreamConfig {
  topics: string[];
  consumer_group: string;
  from_beginning?: boolean;
  batch_size?: number;
  auto_commit?: boolean;
  max_poll_records?: number;
}

export interface MLModelStatus {
  model_id: string;
  status: 'loading' | 'ready' | 'training' | 'error' | 'compiling';
  accuracy: number;
  loss: number;
  predictions_per_sec: number;
  last_update: number;
  memory_usage_mb: number;
  inference_p50_us: number;
  inference_p99_us: number;
  pgo_enabled: boolean;
}

export interface SystemMetrics {
  cpu_usage: number[];
  memory_usage: number;
  disk_io: { read: number; write: number };
  network_io: { in: number; out: number };
  goroutines?: number;
  gc_pauses?: number[];
  uptime_seconds: number;
  decision_latency_p50_ms: number;
  decision_latency_p99_ms: number;
  bundle_land_rate: number;
  ingestion_rate: number;
}

export interface ServiceHealth {
  service: string;
  status: 'healthy' | 'degraded' | 'down';
  latency_ms: number;
  error_rate: number;
  throughput: number;
  last_check: number;
}

export interface Alert {
  id: string;
  type: 'info' | 'warning' | 'error' | 'critical';
  category: 'system' | 'mev' | 'model' | 'network' | 'security';
  message: string;
  details?: any;
  timestamp: number;
  acknowledged?: boolean;
}

export interface DecisionDNA {
  fingerprint: string;
  parent: string | null;
  decision_type: string;
  model_version: string;
  features_hash: string;
  outcome: string;
  timestamp: number;
  merkle_proof?: string[];
}

export interface LeaderSchedule {
  slot: number;
  leader: string;
  region: string;
  latency_ms: number;
  reliability: number;
}

class APIService {
  private client: typeof ky;
  private authToken: string | null = null;
  private requestCounter = 0;

  constructor() {
    this.client = ky.create({
      prefixUrl: API_BASE_URL,
      timeout: API_TIMEOUT,
      retry: {
        limit: MAX_RETRIES,
        delay: attemptCount => Math.min(RETRY_DELAY * Math.pow(2, attemptCount - 1), 10000),
        statusCodes: [408, 429, 500, 502, 503, 504]
      },
      hooks: {
        beforeRequest: [
          request => {
            // Add auth token
            if (this.authToken) {
              request.headers.set('Authorization', `Bearer ${this.authToken}`);
            }
            
            // Add request tracking
            const requestId = `${Date.now()}-${++this.requestCounter}`;
            request.headers.set('X-Request-ID', requestId);
            request.headers.set('X-Client-Version', '4.0.0-legendary');
            
            // Performance tracking
            perfMarks.set(requestId, performance.now());
          }
        ],
        afterResponse: [
          (_request, _options, response) => {
            // Track latency
            const requestId = response.headers.get('X-Request-ID');
            if (requestId && perfMarks.has(requestId)) {
              const latency = performance.now() - perfMarks.get(requestId)!;
              perfMarks.delete(requestId);
              
              // Log ultra-fast responses
              if (latency < 10) {
                console.debug(`⚡ Ultra-fast API response: ${latency.toFixed(2)}ms`);
              }
            }
            
            return response;
          }
        ]
      }
    });
  }

  public setAuthToken(token: string): void {
    this.authToken = token;
  }

  // Control Operations
  public async sendCommand(command: ControlCommand): Promise<any> {
    return this.client.post('api/control/command', { json: command }).json();
  }

  public async updatePolicy(policy: PolicyUpdate): Promise<any> {
    return this.client.post('api/control/policy', { json: policy }).json();
  }

  public async swapModel(swap: ModelSwap): Promise<any> {
    return this.client.post('api/control/model-swap', { json: swap }).json();
  }

  public async triggerKillSwitch(killSwitch: KillSwitch): Promise<any> {
    return this.client.post('api/control/kill-switch', { json: killSwitch }).json();
  }

  public async getControlStatus(): Promise<any> {
    return this.client.get('api/control/status').json();
  }

  public async getCommandHistory(limit = 100): Promise<any[]> {
    return this.client.get('api/control/history', { searchParams: { limit } }).json();
  }

  public async getACKChain(): Promise<any> {
    return this.client.get('api/control/ack-chain').json();
  }

  // Real-time Data
  public async getMEVOpportunities(limit = 100): Promise<MEVOpportunity[]> {
    return this.client.get('api/realtime/opportunities', { searchParams: { limit } }).json();
  }

  public async submitJitoBundle(bundle: JitoBundleSubmission): Promise<any> {
    return this.client.post('api/realtime/jito-bundle', { json: bundle }).json();
  }

  public async getThompsonSamplingMetrics(): Promise<ThompsonSamplingMetrics> {
    return this.client.get('api/realtime/thompson-sampling').json();
  }

  public async getBlake3Metrics(): Promise<Blake3Metrics> {
    return this.client.get('api/realtime/blake3-metrics').json();
  }

  public async getDecisionDNA(limit = 100): Promise<DecisionDNA[]> {
    return this.client.get('api/realtime/decision-dna', { searchParams: { limit } }).json();
  }

  public async getLeaderSchedule(): Promise<LeaderSchedule[]> {
    return this.client.get('api/realtime/leader-schedule').json();
  }

  // Data Queries
  public async executeClickHouseQuery(query: ClickHouseQuery): Promise<any> {
    return this.client.post('api/datasets/clickhouse/query', { json: query }).json();
  }

  public async getClickHouseTables(): Promise<string[]> {
    return this.client.get('api/datasets/clickhouse/tables').json();
  }

  public async getClickHouseStats(): Promise<any> {
    return this.client.get('api/datasets/clickhouse/stats').json();
  }

  public async getKafkaTopics(): Promise<string[]> {
    return this.client.get('api/datasets/kafka/topics').json();
  }

  public async subscribeToKafkaStream(config: KafkaStreamConfig): Promise<any> {
    return this.client.post('api/datasets/kafka/subscribe', { json: config }).json();
  }

  public async getKafkaConsumerGroups(): Promise<any[]> {
    return this.client.get('api/datasets/kafka/consumer-groups').json();
  }

  // ML Operations
  public async getModelStatus(modelId?: string): Promise<MLModelStatus[]> {
    const params = modelId ? { searchParams: { model_id: modelId } } : {};
    return this.client.get('api/training/models/status', params).json();
  }

  public async deployModel(modelSwap: ModelSwap): Promise<any> {
    return this.client.post('api/training/models/deploy', { json: modelSwap }).json();
  }

  public async trainModel(config: any): Promise<any> {
    return this.client.post('api/training/models/train', { json: config }).json();
  }

  public async getModelMetrics(modelId: string): Promise<any> {
    return this.client.get(`api/training/models/${modelId}/metrics`).json();
  }

  public async compileTreelite(modelId: string): Promise<any> {
    return this.client.post(`api/training/models/${modelId}/compile-treelite`).json();
  }

  public async enablePGO(modelId: string): Promise<any> {
    return this.client.post(`api/training/models/${modelId}/enable-pgo`).json();
  }

  // System Health
  public async getSystemMetrics(): Promise<SystemMetrics> {
    return this.client.get('api/health/metrics').json();
  }

  public async getServiceHealth(): Promise<ServiceHealth[]> {
    return this.client.get('api/health/services').json();
  }

  public async getAlerts(limit = 50): Promise<Alert[]> {
    return this.client.get('api/health/alerts', { searchParams: { limit } }).json();
  }

  public async acknowledgeAlert(alertId: string): Promise<any> {
    return this.client.post(`api/health/alerts/${alertId}/acknowledge`).json();
  }

  public async getSLOStatus(): Promise<any> {
    return this.client.get('api/health/slo-status').json();
  }

  // Data Export
  public async exportData(
    dataType: string,
    format: 'json' | 'csv' | 'parquet' | 'arrow',
    timeRange?: { start: number; end: number }
  ): Promise<Blob> {
    const params: any = { type: dataType, format };
    if (timeRange) {
      params.start = timeRange.start;
      params.end = timeRange.end;
    }
    
    return this.client.get('api/datasets/export', { searchParams: params }).blob();
  }

  // WebSocket Bridge
  public async getWebSocketToken(): Promise<string> {
    const response = await this.client.post('api/realtime/ws-token').json<{ token: string }>();
    return response.token;
  }

  public async getWebTransportToken(): Promise<string> {
    const response = await this.client.post('api/realtime/wt-token').json<{ token: string }>();
    return response.token;
  }

  // Advanced Operations
  public async getProtobufSchema(name: string): Promise<any> {
    return this.client.get(`api/advanced/protobuf-schema/${name}`).json();
  }

  public async getMerkleProof(fingerprint: string): Promise<any> {
    return this.client.get(`api/advanced/merkle-proof/${fingerprint}`).json();
  }

  public async getMultisigStatus(): Promise<any> {
    return this.client.get('api/advanced/multisig-status').json();
  }

  public async submitSignedCommand(command: any, signatures: string[]): Promise<any> {
    return this.client.post('api/advanced/signed-command', { 
      json: { command, signatures } 
    }).json();
  }

  // Scrapper Operations
  public async getScrapperStatus(): Promise<any> {
    return this.client.get('api/scrapper/status').json();
  }

  public async configureScrappers(config: any): Promise<any> {
    return this.client.post('api/scrapper/configure', { json: config }).json();
  }

  public async getScrapperMetrics(): Promise<any> {
    return this.client.get('api/scrapper/metrics').json();
  }

  // Node Operations
  public async getNodeStatus(): Promise<any> {
    return this.client.get('api/node/status').json();
  }

  public async getTPUMetrics(): Promise<any> {
    return this.client.get('api/node/tpu-metrics').json();
  }

  public async getTransportMetrics(): Promise<any> {
    return this.client.get('api/node/transport-metrics').json();
  }

  public async getGeyserStatus(): Promise<any> {
    return this.client.get('api/node/geyser-status').json();
  }
}

// Export singleton instance
export const apiService = new APIService();

// Export convenience functions for direct access
export const {
  sendCommand,
  updatePolicy,
  swapModel,
  triggerKillSwitch,
  getMEVOpportunities,
  submitJitoBundle,
  getThompsonSamplingMetrics,
  getBlake3Metrics,
  executeClickHouseQuery,
  getModelStatus,
  deployModel,
  trainModel,
  getSystemMetrics,
  exportData,
  getWebSocketToken,
  getWebTransportToken,
} = apiService;