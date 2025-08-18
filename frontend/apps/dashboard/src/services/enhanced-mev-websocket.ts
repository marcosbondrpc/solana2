/**
 * Enhanced MEV WebSocket Service
 * Handles real-time MEV data streaming with automatic reconnection
 */

import { EventEmitter } from 'events';

export interface MEVMetrics {
  totalProfit: number;
  successRate: number;
  activeOpportunities: number;
  gasOptimization: number;
  bundlesLanded: number;
  averageLatency: number;
  timestamp: number;
}

export interface MEVOpportunity {
  id: string;
  type: 'arbitrage' | 'liquidation' | 'jit' | 'sandwich';
  pair: string;
  dex: string;
  profitEstimate: number;
  gasEstimate: number;
  confidence: number;
  expiresAt: number;
  status: 'pending' | 'executing' | 'completed' | 'failed';
}

export interface ServiceHealth {
  service: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  latency: number;
  uptime: number;
  lastCheck: number;
}

class EnhancedMEVWebSocket extends EventEmitter {
  private ws: WebSocket | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private connectionAttempts = 0;
  private maxReconnectAttempts = 10;
  private baseReconnectDelay = 1000;
  private isIntentionalClose = false;

  constructor(private url: string) {
    super();
    this.connect();
  }

  private connect() {
    try {
      this.ws = new WebSocket(this.url);
      this.setupEventHandlers();
    } catch (error) {
      console.error('WebSocket connection error:', error);
      this.scheduleReconnect();
    }
  }

  private setupEventHandlers() {
    if (!this.ws) return;

    this.ws.onopen = () => {
      console.log('WebSocket connected successfully');
      this.connectionAttempts = 0;
      this.emit('connected');
      this.startHeartbeat();
      
      // Subscribe to MEV data streams
      this.send({
        type: 'subscribe',
        channels: ['mev-metrics', 'opportunities', 'service-health', 'alerts'],
      });
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.emit('error', error);
    };

    this.ws.onclose = () => {
      this.stopHeartbeat();
      this.emit('disconnected');
      
      if (!this.isIntentionalClose) {
        this.scheduleReconnect();
      }
    };
  }

  private handleMessage(data: any) {
    switch (data.type) {
      case 'mev-metrics':
        this.emit('metrics', data.payload as MEVMetrics);
        break;
      
      case 'opportunity':
        this.emit('opportunity', data.payload as MEVOpportunity);
        break;
      
      case 'service-health':
        this.emit('health', data.payload as ServiceHealth);
        break;
      
      case 'alert':
        this.emit('alert', data.payload);
        break;
      
      case 'pong':
        // Heartbeat response
        break;
      
      default:
        console.warn('Unknown message type:', data.type);
    }
  }

  private startHeartbeat() {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping' });
      }
    }, 30000);
  }

  private stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private scheduleReconnect() {
    if (this.connectionAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.emit('max-reconnect-attempts');
      return;
    }

    const delay = Math.min(
      this.baseReconnectDelay * Math.pow(2, this.connectionAttempts),
      30000
    );

    this.connectionAttempts++;
    console.log(`Reconnecting in ${delay}ms (attempt ${this.connectionAttempts})`);

    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  public send(data: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket not connected, queuing message');
      // Could implement a message queue here
    }
  }

  public disconnect() {
    this.isIntentionalClose = true;
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    this.stopHeartbeat();
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  public getConnectionState(): 'connecting' | 'connected' | 'disconnected' {
    if (!this.ws) return 'disconnected';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting';
      case WebSocket.OPEN:
        return 'connected';
      default:
        return 'disconnected';
    }
  }
}

// Singleton instance
let instance: EnhancedMEVWebSocket | null = null;

export function getMEVWebSocket(url?: string): EnhancedMEVWebSocket {
  if (!instance && url) {
    instance = new EnhancedMEVWebSocket(url);
  }
  
  if (!instance) {
    throw new Error('MEV WebSocket not initialized');
  }
  
  return instance;
}

export function closeMEVWebSocket() {
  if (instance) {
    instance.disconnect();
    instance = null;
  }
}