/**
 * Node Metrics Service
 * Handles all Node infrastructure monitoring API calls and WebSocket connections
 */

import ApiBase from './api.base';
import WebSocketBase from './websocket.base';
import { API_CONFIG } from '../config/api.config';
import { 
  NodeMetrics, 
  ApiResponse,
  SystemMetrics,
  ConnectionStatus,
  NodeMetricsUpdate
} from '../types/api.types';

export interface NodeConfig {
  rpcEndpoints: string[];
  geyserEnabled: boolean;
  jitoEnabled: boolean;
  maxConnections: number;
  metricsInterval: number;
}

export interface NodeHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  uptime: number;
  lastCheck: string;
  services: {
    rpc: ConnectionStatus;
    websocket: ConnectionStatus;
    geyser: ConnectionStatus;
    jito: ConnectionStatus;
  };
  issues: string[];
}

export interface NodeStatus {
  version: string;
  network: 'mainnet' | 'testnet' | 'devnet';
  identity: string;
  voteAccount?: string;
  stake?: number;
  skipRate?: number;
  credits?: number;
}

class NodeMetricsService extends ApiBase {
  private metricsWS: NodeMetricsWebSocket | null = null;
  private pollingInterval: NodeJS.Timeout | null = null;
  private metricsCache: NodeMetrics | null = null;
  private lastFetchTime = 0;
  private cacheDuration = 5000; // 5 seconds

  /**
   * Get current node metrics
   */
  async getMetrics(useCache = true): Promise<ApiResponse<NodeMetrics>> {
    // Return cached data if fresh
    if (useCache && this.metricsCache && Date.now() - this.lastFetchTime < this.cacheDuration) {
      return {
        success: true,
        data: this.metricsCache,
        timestamp: Date.now(),
      };
    }

    const response = await this.get<NodeMetrics>(API_CONFIG.endpoints.node.metrics);
    
    if (response.success && response.data) {
      this.metricsCache = response.data;
      this.lastFetchTime = Date.now();
    }
    
    return response;
  }

  /**
   * Get node health status
   */
  async getHealth(): Promise<ApiResponse<NodeHealth>> {
    return this.get<NodeHealth>(API_CONFIG.endpoints.node.health);
  }

  /**
   * Get node status information
   */
  async getStatus(): Promise<ApiResponse<NodeStatus>> {
    return this.get<NodeStatus>(API_CONFIG.endpoints.node.status);
  }

  /**
   * Get node configuration
   */
  async getConfig(): Promise<ApiResponse<NodeConfig>> {
    return this.get<NodeConfig>(API_CONFIG.endpoints.node.config);
  }

  /**
   * Update node configuration
   */
  async updateConfig(config: Partial<NodeConfig>): Promise<ApiResponse<NodeConfig>> {
    return this.put<NodeConfig>(API_CONFIG.endpoints.node.config, config);
  }

  /**
   * Start polling for metrics
   */
  startPolling(interval = 5000, onUpdate?: (metrics: NodeMetrics) => void): void {
    this.stopPolling();
    
    const poll = async () => {
      try {
        const response = await this.getMetrics(false);
        if (response.success && response.data && onUpdate) {
          onUpdate(response.data);
        }
      } catch (error) {
        console.error('Polling error:', error);
      }
    };
    
    // Initial poll
    poll();
    
    // Set up interval
    this.pollingInterval = setInterval(poll, interval);
  }

  /**
   * Stop polling for metrics
   */
  stopPolling(): void {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
  }

  /**
   * Connect to metrics WebSocket stream
   */
  connectWebSocket(
    onMetricsUpdate?: (metrics: NodeMetrics) => void,
    onStatusChange?: (status: ConnectionStatus) => void
  ): NodeMetricsWebSocket {
    // Disconnect existing WebSocket if any
    if (this.metricsWS) {
      this.metricsWS.disconnect();
    }
    
    this.metricsWS = new NodeMetricsWebSocket();
    
    // Set up event handlers
    if (onMetricsUpdate) {
      this.metricsWS.onMetrics(onMetricsUpdate);
    }
    
    if (onStatusChange) {
      this.metricsWS.onStatusChange(onStatusChange);
    }
    
    // Auto-connect
    this.metricsWS.connect().catch(error => {
      console.error('WebSocket connection failed:', error);
    });
    
    return this.metricsWS;
  }

  /**
   * Disconnect WebSocket
   */
  disconnectWebSocket(): void {
    if (this.metricsWS) {
      this.metricsWS.disconnect();
      this.metricsWS = null;
    }
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    this.stopPolling();
    this.disconnectWebSocket();
    this.cancelAllRequests();
    this.metricsCache = null;
  }
}

/**
 * WebSocket client for real-time node metrics
 */
export class NodeMetricsWebSocket extends WebSocketBase {
  private statusChangeHandlers = new Set<(status: ConnectionStatus) => void>();
  private currentStatus: ConnectionStatus = 'disconnected';

  constructor() {
    super(API_CONFIG.ws.nodeMetrics);
    
    // Set up internal status tracking
    this.on('connected', () => this.updateStatus('connected'));
    this.on('disconnected', () => this.updateStatus('disconnected'));
    this.on('reconnecting', () => this.updateStatus('reconnecting'));
    this.onError(() => this.updateStatus('error'));
  }

  /**
   * Subscribe to metrics updates
   */
  onMetrics(handler: (metrics: NodeMetrics) => void): () => void {
    return this.on('node-metrics', (data: NodeMetricsUpdate) => {
      handler(data.payload);
    });
  }

  /**
   * Subscribe to system metrics updates
   */
  onSystemMetrics(handler: (metrics: SystemMetrics) => void): () => void {
    return this.on('system-metrics', handler);
  }

  /**
   * Subscribe to RPC metrics updates
   */
  onRPCMetrics(handler: (metrics: any) => void): () => void {
    return this.on('rpc-metrics', handler);
  }

  /**
   * Subscribe to Jito metrics updates
   */
  onJitoMetrics(handler: (metrics: any) => void): () => void {
    return this.on('jito-metrics', handler);
  }

  /**
   * Subscribe to connection status changes
   */
  onStatusChange(handler: (status: ConnectionStatus) => void): () => void {
    this.statusChangeHandlers.add(handler);
    
    // Immediately call with current status
    handler(this.currentStatus);
    
    return () => {
      this.statusChangeHandlers.delete(handler);
    };
  }

  /**
   * Request specific metrics
   */
  requestMetrics(type: string): void {
    this.send('request-metrics', { type });
  }

  /**
   * Update and emit status change
   */
  private updateStatus(status: ConnectionStatus): void {
    if (this.currentStatus !== status) {
      this.currentStatus = status;
      this.statusChangeHandlers.forEach(handler => {
        try {
          handler(status);
        } catch (error) {
          console.error('Error in status change handler:', error);
        }
      });
    }
  }

  /**
   * Get current connection status
   */
  getStatus(): ConnectionStatus {
    return this.currentStatus;
  }
}

// Export singleton instance
const nodeMetricsService = new NodeMetricsService();
export default nodeMetricsService;