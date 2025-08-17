import { encode, decode } from '@msgpack/msgpack';
import EventEmitter from 'eventemitter3';
import { useMEVStore } from '@/stores/mev-store';

export interface WebSocketMessage {
  type: 'arbitrage' | 'bundle' | 'latency' | 'flow' | 'profit' | 'system' | 'heartbeat';
  data: any;
  timestamp: number;
  sequence: number;
}

interface ConnectionOptions {
  url: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  binaryProtocol?: boolean;
  compression?: boolean;
}

class MEVWebSocketService extends EventEmitter {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectInterval: number;
  private maxReconnectAttempts: number;
  private heartbeatInterval: number;
  private reconnectAttempts: number = 0;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private messageQueue: WebSocketMessage[] = [];
  private isConnected: boolean = false;
  private lastSequence: number = 0;
  private missedSequences: Set<number> = new Set();
  private latencyProbes: Map<number, number> = new Map();
  private binaryProtocol: boolean;
  private compression: boolean;
  private sharedBuffer: SharedArrayBuffer | null = null;
  private bufferView: DataView | null = null;
  private stats = {
    messagesReceived: 0,
    bytesSent: 0,
    bytesReceived: 0,
    errors: 0,
    reconnects: 0,
    avgLatency: 0,
    latencySamples: [] as number[]
  };
  
  constructor(options: ConnectionOptions) {
    super();
    this.url = options.url;
    this.reconnectInterval = options.reconnectInterval || 5000;
    this.maxReconnectAttempts = options.maxReconnectAttempts || 10;
    this.heartbeatInterval = options.heartbeatInterval || 30000;
    this.binaryProtocol = options.binaryProtocol !== false;
    this.compression = options.compression !== false;
    
    // Initialize SharedArrayBuffer for zero-copy data sharing
    if (typeof SharedArrayBuffer !== 'undefined') {
      this.sharedBuffer = new SharedArrayBuffer(1024 * 1024); // 1MB buffer
      this.bufferView = new DataView(this.sharedBuffer);
    }
  }
  
  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }
    
    try {
      // Add compression and binary protocol headers
      const protocols = [];
      if (this.compression) protocols.push('permessage-deflate');
      
      this.ws = new WebSocket(this.url, protocols);
      this.ws.binaryType = 'arraybuffer';
      
      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onerror = this.handleError.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
      
    } catch (error) {
      console.error('WebSocket connection error:', error);
      this.scheduleReconnect();
    }
  }
  
  private handleOpen(): void {
    console.log('MEV WebSocket connected');
    this.isConnected = true;
    this.reconnectAttempts = 0;
    
    // Update store connection status
    const store = useMEVStore.getState();
    store.setConnectionStatus(true, 0);
    
    // Start heartbeat
    this.startHeartbeat();
    
    // Send queued messages
    this.flushMessageQueue();
    
    // Request missed sequences if any
    if (this.missedSequences.size > 0) {
      this.requestMissedSequences();
    }
    
    this.emit('connected');
  }
  
  private async handleMessage(event: MessageEvent): Promise<void> {
    const startTime = performance.now();
    
    try {
      let message: WebSocketMessage;
      
      if (event.data instanceof ArrayBuffer) {
        // Binary protocol with MessagePack
        const uint8Array = new Uint8Array(event.data);
        message = decode(uint8Array) as WebSocketMessage;
        this.stats.bytesReceived += event.data.byteLength;
      } else {
        // Fallback to JSON
        message = JSON.parse(event.data);
        this.stats.bytesReceived += event.data.length;
      }
      
      // Check sequence for missed messages
      if (message.sequence) {
        if (message.sequence > this.lastSequence + 1) {
          for (let seq = this.lastSequence + 1; seq < message.sequence; seq++) {
            this.missedSequences.add(seq);
          }
        }
        this.lastSequence = message.sequence;
      }
      
      // Handle latency probe responses
      if (message.type === 'heartbeat' && message.data?.probeId) {
        const sentTime = this.latencyProbes.get(message.data.probeId);
        if (sentTime) {
          const latency = performance.now() - sentTime;
          this.updateLatencyStats(latency);
          this.latencyProbes.delete(message.data.probeId);
        }
      }
      
      // Process message based on type
      await this.processMessage(message);
      
      // Update stats
      this.stats.messagesReceived++;
      
      // Emit for subscribers
      this.emit('message', message);
      
      const processingTime = performance.now() - startTime;
      if (processingTime > 10) {
        console.warn(`Slow message processing: ${processingTime.toFixed(2)}ms`, message.type);
      }
      
    } catch (error) {
      console.error('Error processing WebSocket message:', error);
      this.stats.errors++;
    }
  }
  
  private async processMessage(message: WebSocketMessage): Promise<void> {
    const store = useMEVStore.getState();
    
    switch (message.type) {
      case 'arbitrage':
        // Use Web Worker for heavy computation if available
        if (typeof Worker !== 'undefined' && message.data.opportunities?.length > 100) {
          this.processInWorker('arbitrage', message.data);
        } else {
          message.data.opportunities?.forEach((opp: any) => {
            store.addArbitrageOpportunity({
              ...opp,
              timestamp: message.timestamp
            });
          });
        }
        break;
        
      case 'bundle':
        message.data.bundles?.forEach((bundle: any) => {
          store.addJitoBundle({
            ...bundle,
            timestamp: message.timestamp
          });
        });
        break;
        
      case 'latency':
        store.addLatencyMetric({
          ...message.data,
          timestamp: message.timestamp
        });
        break;
        
      case 'flow':
        store.updateDEXFlow({
          ...message.data,
          timestamp: message.timestamp,
          flows: new Map(message.data.flows),
          uniqueTokens: new Set(message.data.uniqueTokens)
        });
        break;
        
      case 'profit':
        store.updateProfitMetrics({
          ...message.data,
          profitByDex: new Map(message.data.profitByDex || []),
          profitByToken: new Map(message.data.profitByToken || [])
        });
        break;
        
      case 'system':
        store.updateSystemPerformance({
          ...message.data,
          timestamp: message.timestamp
        });
        break;
    }
  }
  
  private processInWorker(type: string, data: any): void {
    // Worker processing would be implemented here
    // For now, fallback to main thread
    const store = useMEVStore.getState();
    
    if (type === 'arbitrage' && data.opportunities) {
      data.opportunities.forEach((opp: any) => {
        store.addArbitrageOpportunity(opp);
      });
    }
  }
  
  private handleError(error: Event): void {
    console.error('WebSocket error:', error);
    this.stats.errors++;
    this.emit('error', error);
  }
  
  private handleClose(event: CloseEvent): void {
    console.log('WebSocket closed:', event.code, event.reason);
    this.isConnected = false;
    
    // Update store
    const store = useMEVStore.getState();
    store.setConnectionStatus(false);
    
    // Stop heartbeat
    this.stopHeartbeat();
    
    // Attempt reconnection if not intentional close
    if (event.code !== 1000) {
      this.scheduleReconnect();
    }
    
    this.emit('disconnected', event);
  }
  
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.emit('reconnect_failed');
      return;
    }
    
    this.reconnectAttempts++;
    this.stats.reconnects++;
    
    const delay = Math.min(
      this.reconnectInterval * Math.pow(1.5, this.reconnectAttempts - 1),
      30000
    );
    
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }
  
  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected) {
        const probeId = Date.now();
        this.latencyProbes.set(probeId, performance.now());
        
        this.send({
          type: 'heartbeat',
          data: { probeId },
          timestamp: Date.now(),
          sequence: 0
        });
        
        // Clean old probes
        const cutoff = performance.now() - 10000;
        for (const [id, time] of this.latencyProbes) {
          if (time < cutoff) {
            this.latencyProbes.delete(id);
          }
        }
      }
    }, this.heartbeatInterval);
  }
  
  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }
  
  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.isConnected) {
      const message = this.messageQueue.shift();
      if (message) {
        this.send(message);
      }
    }
  }
  
  private requestMissedSequences(): void {
    if (this.missedSequences.size === 0) return;
    
    const sequences = Array.from(this.missedSequences);
    this.send({
      type: 'heartbeat',
      data: {
        action: 'resend',
        sequences
      },
      timestamp: Date.now(),
      sequence: 0
    });
    
    this.missedSequences.clear();
  }
  
  private updateLatencyStats(latency: number): void {
    this.stats.latencySamples.push(latency);
    
    // Keep only last 100 samples
    if (this.stats.latencySamples.length > 100) {
      this.stats.latencySamples.shift();
    }
    
    // Calculate average
    this.stats.avgLatency = this.stats.latencySamples.reduce((a, b) => a + b, 0) / 
      this.stats.latencySamples.length;
    
    // Update store
    const store = useMEVStore.getState();
    store.setConnectionStatus(true, this.stats.avgLatency);
  }
  
  send(message: WebSocketMessage): boolean {
    if (!this.isConnected || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.messageQueue.push(message);
      return false;
    }
    
    try {
      let data: ArrayBuffer | string;
      
      if (this.binaryProtocol) {
        // Use MessagePack for binary encoding
        const encoded = encode(message);
        data = encoded.buffer.slice(
          encoded.byteOffset,
          encoded.byteOffset + encoded.byteLength
        );
        this.stats.bytesSent += data.byteLength;
      } else {
        // Fallback to JSON
        data = JSON.stringify(message);
        this.stats.bytesSent += data.length;
      }
      
      this.ws.send(data);
      return true;
      
    } catch (error) {
      console.error('Error sending WebSocket message:', error);
      this.stats.errors++;
      this.messageQueue.push(message);
      return false;
    }
  }
  
  executeArbitrage(opportunityId: string): void {
    this.send({
      type: 'arbitrage',
      data: {
        action: 'execute',
        opportunityId
      },
      timestamp: Date.now(),
      sequence: 0
    });
  }
  
  submitBundle(bundle: any): void {
    this.send({
      type: 'bundle',
      data: {
        action: 'submit',
        bundle
      },
      timestamp: Date.now(),
      sequence: 0
    });
  }
  
  updateConfiguration(config: any): void {
    this.send({
      type: 'heartbeat',
      data: {
        action: 'config',
        config
      },
      timestamp: Date.now(),
      sequence: 0
    });
  }
  
  getStats() {
    return {
      ...this.stats,
      isConnected: this.isConnected,
      reconnectAttempts: this.reconnectAttempts,
      queuedMessages: this.messageQueue.length,
      missedSequences: this.missedSequences.size
    };
  }
  
  disconnect(): void {
    this.stopHeartbeat();
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    
    this.isConnected = false;
    this.messageQueue = [];
    this.latencyProbes.clear();
    this.missedSequences.clear();
    
    // Update store
    const store = useMEVStore.getState();
    store.setConnectionStatus(false);
  }
  
  destroy(): void {
    this.disconnect();
    this.removeAllListeners();
    this.sharedBuffer = null;
    this.bufferView = null;
  }
}

// Singleton instance
let instance: MEVWebSocketService | null = null;

export const getMEVWebSocket = (options?: ConnectionOptions): MEVWebSocketService => {
  if (!instance && options) {
    instance = new MEVWebSocketService(options);
  }
  
  if (!instance) {
    throw new Error('MEVWebSocketService not initialized. Call with options first.');
  }
  
  return instance;
};

export const initializeMEVWebSocket = (url: string = 'ws://localhost:8080/mev'): MEVWebSocketService => {
  const service = getMEVWebSocket({
    url,
    reconnectInterval: 3000,
    maxReconnectAttempts: 20,
    heartbeatInterval: 25000,
    binaryProtocol: true,
    compression: true
  });
  
  service.connect();
  return service;
};

export default MEVWebSocketService;