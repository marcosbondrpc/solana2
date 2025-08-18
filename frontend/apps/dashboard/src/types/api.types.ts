/**
 * API Type Definitions
 * Comprehensive TypeScript types for all API requests and responses
 */

// Generic API Response wrapper
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: ApiError;
  timestamp: number;
  requestId?: string;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, any>;
  stack?: string;
}

// Node Metrics Types
export interface NodeMetrics {
  rpc: RPCMetrics;
  websocket: WebSocketMetrics;
  geyser: GeyserMetrics;
  jito: JitoMetrics;
  network: NetworkMetrics;
  system: SystemMetrics;
  timestamp: number;
}

export interface RPCMetrics {
  latency: number;
  status: ConnectionStatus;
  requests: number;
  errors: number;
  throughput: number;
  avgResponseTime: number;
  p95ResponseTime: number;
  p99ResponseTime: number;
  endpoints: RPCEndpoint[];
}

export interface RPCEndpoint {
  url: string;
  status: ConnectionStatus;
  latency: number;
  weight: number;
  failures: number;
}

export interface WebSocketMetrics {
  latency: number;
  status: ConnectionStatus;
  messages: number;
  connections: number;
  bandwidth: number;
  subscriptions: number;
  reconnects: number;
  messageRate: number;
}

export interface GeyserMetrics {
  latency: number;
  status: ConnectionStatus;
  slots: number;
  accounts: number;
  throughput: number;
  blockHeight: number;
  accountUpdates: number;
  slotUpdateRate: number;
}

export interface JitoMetrics {
  latency: number;
  status: ConnectionStatus;
  bundles: number;
  tips: number;
  successRate: number;
  avgBundleSize: number;
  totalRevenue: number;
  pendingBundles: number;
}

export interface NetworkMetrics {
  peersConnected: number;
  blockHeight: number;
  slot: number;
  tps: number;
  avgBlockTime: number;
  epochProgress: number;
  validatorCount: number;
  totalStake: number;
}

export interface SystemMetrics {
  cpuUsage: number;
  memoryUsage: number;
  memoryTotal: number;
  diskUsage: number;
  diskTotal: number;
  networkIn: number;
  networkOut: number;
  uptime: number;
  processCount: number;
}

export type ConnectionStatus = 'connected' | 'disconnected' | 'error' | 'connecting' | 'reconnecting';

// Scrapper Types
export interface Dataset {
  id: string;
  name: string;
  type: DatasetType;
  status: DatasetStatus;
  progress: number;
  totalRecords: number;
  sizeBytes: number;
  createdAt: string;
  updatedAt: string;
  startTime?: string;
  endTime?: string;
  filters: DatasetFilters;
  metadata: DatasetMetadata;
}

export type DatasetType = 'arbitrage' | 'sandwich' | 'liquidation' | 'jito' | 'custom' | 'mixed';
export type DatasetStatus = 'idle' | 'collecting' | 'processing' | 'completed' | 'error' | 'paused';

export interface DatasetFilters {
  minProfit?: number;
  maxSlippage?: number;
  tokenPairs?: string[];
  timeRange?: TimeRange;
  programs?: string[];
  minVolume?: number;
  maxLatency?: number;
  includeFailedTxs?: boolean;
}

export interface TimeRange {
  start: string;
  end: string;
}

export interface DatasetMetadata {
  source: string;
  version: string;
  schema: string;
  compression?: string;
  checksum?: string;
  tags?: string[];
}

export interface CollectRequest {
  datasetId?: string;
  type: DatasetType;
  name: string;
  filters: DatasetFilters;
  options?: CollectOptions;
}

export interface CollectOptions {
  batchSize?: number;
  parallel?: boolean;
  priority?: 'low' | 'normal' | 'high';
  notifyOnComplete?: boolean;
}

export interface MLModel {
  id: string;
  name: string;
  type: ModelType;
  architecture: string;
  status: ModelStatus;
  accuracy: number;
  loss: number;
  epoch: number;
  maxEpochs: number;
  trainingTime: number;
  createdAt: string;
  updatedAt: string;
  config: ModelConfig;
  metrics: ModelMetrics;
}

export type ModelType = 'lstm' | 'gru' | 'transformer' | 'random_forest' | 'xgboost' | 'ensemble';
export type ModelStatus = 'idle' | 'training' | 'ready' | 'failed' | 'evaluating' | 'deployed';

export interface ModelConfig {
  learningRate: number;
  batchSize: number;
  hiddenLayers: number;
  dropout: number;
  optimizer: string;
  lossFunction: string;
  earlyStoppingPatience?: number;
  validationSplit?: number;
}

export interface ModelMetrics {
  trainLoss: number[];
  valLoss: number[];
  trainAccuracy: number[];
  valAccuracy: number[];
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix?: number[][];
}

export interface TrainRequest {
  modelId?: string;
  datasetId: string;
  modelType: ModelType;
  config: ModelConfig;
  name?: string;
}

export interface ExportRequest {
  datasetId: string;
  format: ExportFormat;
  compression?: CompressionType;
  includeMetadata?: boolean;
  filters?: DatasetFilters;
}

export type ExportFormat = 'csv' | 'json' | 'parquet' | 'arrow' | 'pickle';
export type CompressionType = 'none' | 'gzip' | 'zstd' | 'lz4' | 'bzip2';

// WebSocket Message Types
export interface WSMessage<T = any> {
  type: string;
  payload: T;
  timestamp: number;
  id?: string;
}

export interface NodeMetricsUpdate extends WSMessage<NodeMetrics> {
  type: 'node-metrics';
}

export interface ScrapperProgressUpdate extends WSMessage {
  type: 'scrapper-progress';
  payload: {
    datasetId: string;
    progress: number;
    recordsProcessed: number;
    estimatedTimeRemaining: number;
    currentBatch: number;
    totalBatches: number;
    errors?: string[];
  };
}

export interface ModelTrainingUpdate extends WSMessage {
  type: 'model-training';
  payload: {
    modelId: string;
    epoch: number;
    loss: number;
    accuracy: number;
    valLoss?: number;
    valAccuracy?: number;
    timePerEpoch: number;
    estimatedTimeRemaining: number;
  };
}

// Request/Response helpers
export interface PaginationParams {
  page?: number;
  limit?: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
  hasNext: boolean;
  hasPrev: boolean;
}

// Error types
export enum ApiErrorCode {
  NETWORK_ERROR = 'NETWORK_ERROR',
  TIMEOUT = 'TIMEOUT',
  UNAUTHORIZED = 'UNAUTHORIZED',
  FORBIDDEN = 'FORBIDDEN',
  NOT_FOUND = 'NOT_FOUND',
  VALIDATION_ERROR = 'VALIDATION_ERROR',
  SERVER_ERROR = 'SERVER_ERROR',
  RATE_LIMIT = 'RATE_LIMIT',
  MAINTENANCE = 'MAINTENANCE',
}