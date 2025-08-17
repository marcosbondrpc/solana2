/**
 * Comprehensive API Service for MEV Control Plane
 * Integrates all backend endpoints with optimized request handling
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { useMEVStore } from '../stores/mev-store';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_TIMEOUT = 30000;
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000;

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
  model_type: 'xgboost' | 'lightgbm' | 'tensorflow' | 'pytorch';
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
}

export interface JitoBundleSubmission {
  bundle_id: string;
  transactions: string[];
  tip_amount: number;
  target_slot?: number;
  relay_url: string;
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
  }>;
  exploration_rate: number;
  exploitation_rate: number;
  total_pulls: number;
  convergence_metric: number;
}

export interface Blake3Metrics {
  total_hashes: number;
  unique_hashes: number;
  collision_rate: number;
  dedup_savings_bytes: number;
  dedup_ratio: number;
  hash_rate_per_sec: number;
}

export interface ClickHouseQuery {
  query: string;
  format?: 'JSON' | 'CSV' | 'TSV' | 'Parquet';
  database?: string;
  timeout_ms?: number;
  max_rows?: number;
}

export interface KafkaStreamConfig {
  topics: string[];
  consumer_group: string;
  from_beginning?: boolean;
  batch_size?: number;
  auto_commit?: boolean;
}

export interface MLModelStatus {
  model_id: string;
  status: 'loading' | 'ready' | 'training' | 'error';
  accuracy: number;
  loss: number;
  predictions_per_sec: number;
  last_update: number;
  memory_usage_mb: number;
}

export interface SystemMetrics {
  cpu_usage: number[];
  memory_usage: number;
  disk_io: { read: number; write: number };
  network_io: { in: number; out: number };
  goroutines?: number;
  gc_pauses?: number[];
  uptime_seconds: number;
}

class APIService {
  private api: AxiosInstance;
  private authToken: string | null = null;
  private requestInterceptor: number | null = null;
  private responseInterceptor: number | null = null;

  constructor() {
    this.api = axios.create({
      baseURL: API_BASE_URL,
      timeout: API_TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
        'X-Client-Version': '3.0.0',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor for auth and performance tracking
    this.requestInterceptor = this.api.interceptors.request.use(
      (config) => {
        // Add auth token if available
        if (this.authToken) {
          config.headers['Authorization'] = `Bearer ${this.authToken}`;
        }

        // Add request ID for tracing
        config.headers['X-Request-ID'] = `${Date.now()}-${Math.random()}`;

        // Add performance mark
        if (typeof performance !== 'undefined') {
          performance.mark(`api-request-start-${config.url}`);
        }

        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling and metrics
    this.responseInterceptor = this.api.interceptors.response.use(
      (response) => {
        // Track performance
        if (typeof performance !== 'undefined' && response.config.url) {
          performance.mark(`api-request-end-${response.config.url}`);
          performance.measure(
            `api-request-${response.config.url}`,
            `api-request-start-${response.config.url}`,
            `api-request-end-${response.config.url}`
          );
        }

        // Update latency metrics
        const latency = response.headers['x-response-time-ms'];
        if (latency) {
          useMEVStore.getState().addLatencyMetric({
            timestamp: Date.now(),
            rpcLatency: parseFloat(latency),
            quicLatency: 0,
            tcpLatency: 0,
            websocketLatency: 0,
            parsingLatency: 0,
            calculationLatency: 0,
            executionLatency: 0,
            confirmationLatency: 0,
            p50: 0,
            p95: 0,
            p99: parseFloat(latency),
            jitter: 0,
          });
        }

        return response;
      },
      async (error) => {
        // Implement retry logic for certain errors
        const config = error.config;
        if (!config || !config.retry) {
          config.retry = 0;
        }

        if (config.retry < MAX_RETRIES && this.shouldRetry(error)) {
          config.retry += 1;
          await this.delay(RETRY_DELAY * config.retry);
          return this.api(config);
        }

        // Add error to store
        useMEVStore.getState().addAlert({
          type: 'error',
          category: 'system',
          message: `API Error: ${error.message}`,
          details: error.response?.data,
        });

        return Promise.reject(error);
      }
    );
  }

  private shouldRetry(error: any): boolean {
    // Retry on network errors or 5xx server errors
    return !error.response || (error.response.status >= 500 && error.response.status < 600);
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  public setAuthToken(token: string): void {
    this.authToken = token;
  }

  // Control Operations
  public async sendCommand(command: ControlCommand): Promise<any> {
    return this.api.post('/api/control/command', command);
  }

  public async updatePolicy(policy: PolicyUpdate): Promise<any> {
    return this.api.post('/api/control/policy', policy);
  }

  public async swapModel(swap: ModelSwap): Promise<any> {
    return this.api.post('/api/control/model-swap', swap);
  }

  public async triggerKillSwitch(killSwitch: KillSwitch): Promise<any> {
    return this.api.post('/api/control/kill-switch', killSwitch);
  }

  public async getControlStatus(): Promise<any> {
    return this.api.get('/api/control/status');
  }

  // Real-time Data
  public async getMEVOpportunities(limit = 100): Promise<MEVOpportunity[]> {
    const response = await this.api.get('/api/realtime/opportunities', {
      params: { limit },
    });
    return response.data;
  }

  public async submitJitoBundle(bundle: JitoBundleSubmission): Promise<any> {
    return this.api.post('/api/realtime/jito-bundle', bundle);
  }

  public async getThompsonSamplingMetrics(): Promise<ThompsonSamplingMetrics> {
    const response = await this.api.get('/api/realtime/thompson-sampling');
    return response.data;
  }

  public async getBlake3Metrics(): Promise<Blake3Metrics> {
    const response = await this.api.get('/api/realtime/blake3-metrics');
    return response.data;
  }

  // Data Queries
  public async executeClickHouseQuery(query: ClickHouseQuery): Promise<any> {
    return this.api.post('/api/datasets/clickhouse/query', query);
  }

  public async getClickHouseTables(): Promise<string[]> {
    const response = await this.api.get('/api/datasets/clickhouse/tables');
    return response.data;
  }

  public async getKafkaTopics(): Promise<string[]> {
    const response = await this.api.get('/api/datasets/kafka/topics');
    return response.data;
  }

  public async subscribeToKafkaStream(config: KafkaStreamConfig): Promise<any> {
    return this.api.post('/api/datasets/kafka/subscribe', config);
  }

  // ML Operations
  public async getModelStatus(modelId?: string): Promise<MLModelStatus[]> {
    const response = await this.api.get('/api/training/models/status', {
      params: { model_id: modelId },
    });
    return response.data;
  }

  public async deployModel(modelSwap: ModelSwap): Promise<any> {
    return this.api.post('/api/training/models/deploy', modelSwap);
  }

  public async trainModel(config: any): Promise<any> {
    return this.api.post('/api/training/models/train', config);
  }

  public async getModelMetrics(modelId: string): Promise<any> {
    const response = await this.api.get(`/api/training/models/${modelId}/metrics`);
    return response.data;
  }

  // System Health
  public async getSystemMetrics(): Promise<SystemMetrics> {
    const response = await this.api.get('/api/health/metrics');
    return response.data;
  }

  public async getServiceHealth(): Promise<any> {
    const response = await this.api.get('/api/health/services');
    return response.data;
  }

  public async getAlerts(limit = 50): Promise<any[]> {
    const response = await this.api.get('/api/health/alerts', {
      params: { limit },
    });
    return response.data;
  }

  // Data Export
  public async exportData(
    dataType: string,
    format: 'json' | 'csv' | 'parquet',
    timeRange?: { start: number; end: number }
  ): Promise<Blob> {
    const response = await this.api.get('/api/datasets/export', {
      params: {
        type: dataType,
        format,
        start: timeRange?.start,
        end: timeRange?.end,
      },
      responseType: 'blob',
    });
    return response.data;
  }

  // WebSocket Bridge
  public async getWebSocketToken(): Promise<string> {
    const response = await this.api.post('/api/realtime/ws-token');
    return response.data.token;
  }

  // Cleanup
  public destroy(): void {
    if (this.requestInterceptor !== null) {
      this.api.interceptors.request.eject(this.requestInterceptor);
    }
    if (this.responseInterceptor !== null) {
      this.api.interceptors.response.eject(this.responseInterceptor);
    }
  }
}

// Export singleton instance
export const apiService = new APIService();

// Export convenience functions
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
  getSystemMetrics,
  exportData,
} = apiService;