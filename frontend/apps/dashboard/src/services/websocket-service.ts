import { io, Socket } from 'socket.io-client';
import { useMonitoringStore } from '@/lib/monitoring-store';

interface WebSocketConfig {
  url: string;
  reconnectionAttempts?: number;
  reconnectionDelay?: number;
  heartbeatInterval?: number;
  messageQueueSize?: number;
  compressionThreshold?: number;
}

export class WebSocketService {
  private socket: Socket | null = null;
  private config: Required<WebSocketConfig>;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private messageBuffer: Map<string, any[]> = new Map();
  private lastMessageTimestamps: Map<string, number> = new Map();
  private performanceMetrics: Map<string, number[]> = new Map();
  private isReconnecting = false;
  private connectionStartTime: number = 0;
  private messageCount = 0;
  private bytesReceived = 0;

  constructor(config: WebSocketConfig) {
    this.config = {
      url: config.url,
      reconnectionAttempts: config.reconnectionAttempts ?? 10,
      reconnectionDelay: config.reconnectionDelay ?? 1000,
      heartbeatInterval: config.heartbeatInterval ?? 30000,
      messageQueueSize: config.messageQueueSize ?? 1000,
      compressionThreshold: config.compressionThreshold ?? 1024,
    };
  }

  public connect(): void {
    if (this.socket?.connected) return;

    const store = useMonitoringStore.getState();
    store.setConnectionStatus('connecting');
    this.connectionStartTime = Date.now();

    this.socket = io(this.config.url, {
      transports: ['websocket', 'polling'],
      reconnection: false, // We handle reconnection manually
      timeout: 20000,
      auth: {
        token: store.authToken,
      },
      query: {
        clientVersion: '2.0.0',
        features: 'monitoring,control,alerts',
      },
    });

    this.setupEventHandlers();
    this.startHeartbeat();
  }

  private setupEventHandlers(): void {
    if (!this.socket) return;

    const store = useMonitoringStore.getState();

    // Connection events
    this.socket.on('connect', () => {
      console.log('[WS] Connected to monitoring server');
      store.setConnectionStatus('connected');
      this.isReconnecting = false;
      this.messageCount = 0;
      this.bytesReceived = 0;
      
      // Request initial data
      this.socket?.emit('request:initial-state');
      
      // Process queued messages
      store.processWSQueue();
    });

    this.socket.on('disconnect', (reason) => {
      console.log('[WS] Disconnected:', reason);
      store.setConnectionStatus('disconnected');
      this.stopHeartbeat();
      
      if (reason === 'io server disconnect') {
        // Server initiated disconnect, don't auto-reconnect
        return;
      }
      
      this.scheduleReconnect();
    });

    this.socket.on('connect_error', (error) => {
      console.error('[WS] Connection error:', error.message);
      store.setConnectionStatus('error');
      this.scheduleReconnect();
    });

    // Metrics events with performance tracking
    this.socket.on('metrics:consensus', (data: any) => {
      this.trackMessagePerformance('consensus');
      store.updateConsensus(data);
      store.addHistoricalPoint('consensus', data);
      this.checkConsensusAlerts(data);
    });

    this.socket.on('metrics:performance', (data: any) => {
      this.trackMessagePerformance('performance');
      store.updatePerformance(data);
      store.addHistoricalPoint('performance', data);
      this.checkPerformanceAlerts(data);
    });

    this.socket.on('metrics:rpc', (data: any) => {
      this.trackMessagePerformance('rpc');
      store.updateRPCLayer(data);
      store.addHistoricalPoint('rpc', data);
      this.checkRPCAlerts(data);
    });

    this.socket.on('metrics:network', (data: any) => {
      this.trackMessagePerformance('network');
      store.updateNetwork(data);
      store.addHistoricalPoint('network', data);
    });

    this.socket.on('metrics:os', (data: any) => {
      this.trackMessagePerformance('os');
      store.updateOS(data);
      store.addHistoricalPoint('os', data);
      this.checkSystemAlerts(data);
    });

    this.socket.on('metrics:jito', (data: any) => {
      this.trackMessagePerformance('jito');
      store.updateJito(data);
      store.addHistoricalPoint('jito', data);
    });

    this.socket.on('metrics:geyser', (data: any) => {
      this.trackMessagePerformance('geyser');
      store.updateGeyser(data);
    });

    this.socket.on('metrics:security', (data: any) => {
      this.trackMessagePerformance('security');
      store.updateSecurity(data);
      
      // Add any new alerts
      if (data.alerts) {
        data.alerts.forEach(alert => {
          if (!store.activeAlerts.find(a => a.id === alert.id)) {
            store.addAlert(alert);
          }
        });
      }
    });

    this.socket.on('health:update', (data: any) => {
      store.updateHealth(data);
      
      // Check for critical health issues
      if (data.overall < 50) {
        store.addAlert({
          id: `health-${Date.now()}`,
          severity: 'critical',
          type: 'health',
          message: `System health critically low: ${data.overall}%`,
          timestamp: new Date(),
          acknowledged: false,
        });
      }
    });

    // Alert events
    this.socket.on('alert:new', (alert) => {
      store.addAlert(alert);
      this.showNotification(alert);
    });

    // Control events
    this.socket.on('control:result', (result) => {
      console.log('[WS] Control command result:', result);
      if (result.success) {
        this.showNotification({
          severity: 'low',
          message: `Command executed successfully: ${result.action}`,
        });
      } else {
        store.addAlert({
          id: `control-error-${Date.now()}`,
          severity: 'high',
          type: 'control',
          message: `Command failed: ${result.error}`,
          timestamp: new Date(),
          acknowledged: false,
        });
      }
    });

    // Batch update for performance
    this.socket.on('metrics:batch', (batch: any[]) => {
      store.batchUpdate(batch.map(item => () => {
        const handler = this.getBatchHandler(item.type);
        if (handler) handler(item.data);
      }));
    });

    // Heartbeat
    this.socket.on('pong', () => {
      const latency = Date.now() - this.lastPingTime;
      this.updateConnectionMetrics(latency);
    });
  }

  private trackMessagePerformance(type: string): void {
    const now = Date.now();
    const lastTimestamp = this.lastMessageTimestamps.get(type) || 0;
    const interval = now - lastTimestamp;
    
    this.lastMessageTimestamps.set(type, now);
    
    // Track performance metrics
    let metrics = this.performanceMetrics.get(type) || [];
    metrics.push(interval);
    if (metrics.length > 100) metrics.shift();
    this.performanceMetrics.set(type, metrics);
    
    this.messageCount++;
  }

  private getBatchHandler(type: string): ((data: any) => void) | null {
    const store = useMonitoringStore.getState();
    const handlers: Record<string, (data: any) => void> = {
      consensus: (data) => store.updateConsensus(data),
      performance: (data) => store.updatePerformance(data),
      rpc: (data) => store.updateRPCLayer(data),
      network: (data) => store.updateNetwork(data),
      os: (data) => store.updateOS(data),
      jito: (data) => store.updateJito(data),
      geyser: (data) => store.updateGeyser(data),
      security: (data) => store.updateSecurity(data),
    };
    return handlers[type] || null;
  }

  private checkConsensusAlerts(metrics: any): void {
    const store = useMonitoringStore.getState();
    const config = store.alertConfig;
    
    if (metrics.skipRate > config.thresholds.skipRate) {
      store.addAlert({
        id: `skip-rate-${Date.now()}`,
        severity: 'high',
        type: 'consensus',
        message: `High skip rate detected: ${metrics.skipRate.toFixed(2)}%`,
        timestamp: new Date(),
        acknowledged: false,
      });
    }
    
    if (metrics.votingState === 'delinquent') {
      store.addAlert({
        id: `delinquent-${Date.now()}`,
        severity: 'critical',
        type: 'consensus',
        message: 'Validator is delinquent!',
        timestamp: new Date(),
        acknowledged: false,
      });
    }
  }

  private checkPerformanceAlerts(metrics: any): void {
    const store = useMonitoringStore.getState();
    
    if (metrics.bankingStage.droppedPackets > 1000) {
      store.addAlert({
        id: `dropped-packets-${Date.now()}`,
        severity: 'medium',
        type: 'performance',
        message: `High packet drop rate: ${metrics.bankingStage.droppedPackets}`,
        timestamp: new Date(),
        acknowledged: false,
      });
    }
  }

  private checkRPCAlerts(metrics: any): void {
    const store = useMonitoringStore.getState();
    const config = store.alertConfig;
    
    Object.entries(metrics.endpoints).forEach(([endpoint, stats]) => {
      if (stats.p99Latency > config.thresholds.rpcLatency) {
        store.addAlert({
          id: `rpc-latency-${endpoint}-${Date.now()}`,
          severity: 'medium',
          type: 'rpc',
          message: `High RPC latency for ${endpoint}: ${stats.p99Latency}ms`,
          timestamp: new Date(),
          acknowledged: false,
        });
      }
    });
  }

  private checkSystemAlerts(metrics: any): void {
    const store = useMonitoringStore.getState();
    const config = store.alertConfig;
    
    if (metrics.cpuTemperature.some(temp => temp > config.thresholds.temperature)) {
      store.addAlert({
        id: `cpu-temp-${Date.now()}`,
        severity: 'high',
        type: 'system',
        message: `High CPU temperature: ${Math.max(...metrics.cpuTemperature)}°C`,
        timestamp: new Date(),
        acknowledged: false,
      });
    }
    
    const memoryPercent = (metrics.memoryDetails.used / 
      (metrics.memoryDetails.used + metrics.memoryDetails.free)) * 100;
    
    if (memoryPercent > config.thresholds.memoryUsage) {
      store.addAlert({
        id: `memory-${Date.now()}`,
        severity: 'medium',
        type: 'system',
        message: `High memory usage: ${memoryPercent.toFixed(1)}%`,
        timestamp: new Date(),
        acknowledged: false,
      });
    }
  }

  private showNotification(alert: any): void {
    const store = useMonitoringStore.getState();
    
    if (!store.config.features.notifications) return;
    if (alert.severity === 'low') return;
    
    // Check if browser supports notifications
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('Solana Node Alert', {
        body: alert.message,
        icon: '/icon.png',
        tag: alert.id,
        requireInteraction: alert.severity === 'critical',
      });
    }
  }

  private lastPingTime = 0;

  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatTimer = setInterval(() => {
      if (this.socket?.connected) {
        this.lastPingTime = Date.now();
        this.socket.emit('ping');
      }
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private scheduleReconnect(): void {
    if (this.isReconnecting) return;
    
    this.isReconnecting = true;
    const store = useMonitoringStore.getState();
    
    let attempts = 0;
    const maxAttempts = this.config.reconnectionAttempts;
    
    const attemptReconnect = () => {
      if (attempts >= maxAttempts) {
        console.error('[WS] Max reconnection attempts reached');
        store.setConnectionStatus('error');
        this.isReconnecting = false;
        return;
      }
      
      attempts++;
      console.log(`[WS] Reconnection attempt ${attempts}/${maxAttempts}`);
      
      this.connect();
      
      this.reconnectTimer = setTimeout(() => {
        if (!this.socket?.connected) {
          attemptReconnect();
        }
      }, this.config.reconnectionDelay * Math.pow(2, attempts - 1));
    };
    
    attemptReconnect();
  }

  private updateConnectionMetrics(latency: number): void {
    // Track connection quality metrics
    const metrics = {
      latency,
      uptime: Date.now() - this.connectionStartTime,
      messagesReceived: this.messageCount,
      bytesReceived: this.bytesReceived,
    };
    
    // Could emit these to the store or use for diagnostics
    console.log('[WS] Connection metrics:', metrics);
  }

  public disconnect(): void {
    this.stopHeartbeat();
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    
    const store = useMonitoringStore.getState();
    store.setConnectionStatus('disconnected');
  }

  public emit(event: string, data: any): void {
    if (this.socket?.connected) {
      this.socket.emit(event, data);
    } else {
      // Queue message if not connected
      const store = useMonitoringStore.getState();
      store.queueWSMessage(event, data);
    }
  }

  public on(event: string, handler: (...args: any[]) => void): void {
    this.socket?.on(event, handler);
  }

  public off(event: string, handler?: (...args: any[]) => void): void {
    this.socket?.off(event, handler);
  }

  public getPerformanceStats(): Map<string, number[]> {
    return this.performanceMetrics;
  }

  public isConnected(): boolean {
    return this.socket?.connected ?? false;
  }
}

// Singleton instance
let wsService: WebSocketService | null = null;

export function getWebSocketService(): WebSocketService {
  if (!wsService) {
    wsService = new WebSocketService({
      url: process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:42392',
    });
  }
  return wsService;
}

export function destroyWebSocketService(): void {
  if (wsService) {
    wsService.disconnect();
    wsService = null;
  }
}