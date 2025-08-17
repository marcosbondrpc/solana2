import { io, Socket } from 'socket.io-client';
import * as protobuf from 'protobufjs';

interface WebSocketConfig {
  url: string;
  reconnectionAttempts?: number;
  reconnectionDelay?: number;
  batchingWindow?: number;
}

export interface DetectionEvent {
  id: string;
  timestamp: number;
  type: 'SANDWICH' | 'FRONTRUN' | 'BACKRUN' | 'ARBITRAGE' | 'LIQUIDATION';
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  actors: {
    attacker: string;
    victim?: string;
  };
  metrics: {
    profitEstimate: number;
    gasUsed: number;
    latency: number;
    confidence: number;
  };
  signatures: {
    ed25519: string;
    merkleRoot: string;
  };
  venue: string;
  txHash: string;
  blockHeight: number;
}

export interface EntityProfile {
  address: string;
  style: 'SURGICAL' | 'SHOTGUN' | 'HYBRID';
  riskAppetite: number; // 0-1
  feePosture: 'AGGRESSIVE' | 'CONSERVATIVE' | 'ADAPTIVE';
  uptime: {
    hours: number;
    pattern: 'CONSISTENT' | 'SPORADIC' | 'SCHEDULED';
  };
  attackVolume: number;
  successRate: number;
  avgProfit: number;
  preferredVenues: string[];
  knownAssociates: string[];
  behavioral_embedding: number[]; // UMAP/t-SNE coordinates
}

export interface ModelMetrics {
  layerId: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  rocAuc: number;
  latencyP50: number;
  latencyP95: number;
  latencyP99: number;
  confusionMatrix: {
    tp: number;
    fp: number;
    tn: number;
    fn: number;
  };
}

class WebSocketManager {
  private socket: Socket | null = null;
  private config: WebSocketConfig;
  private messageBuffer: any[] = [];
  private batchTimer: NodeJS.Timeout | null = null;
  private listeners: Map<string, Set<Function>> = new Map();
  private protoRoot: protobuf.Root | null = null;

  constructor(config: WebSocketConfig) {
    this.config = {
      reconnectionAttempts: 10,
      reconnectionDelay: 1000,
      batchingWindow: 15,
      ...config,
    };
  }

  async initialize(): Promise<void> {
    // Load protobuf schemas for binary message handling
    try {
      this.protoRoot = await protobuf.load('/proto/detection.proto');
    } catch (error) {
      console.warn('Protobuf schema not found, falling back to JSON');
    }

    this.socket = io(this.config.url, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: this.config.reconnectionAttempts,
      reconnectionDelay: this.config.reconnectionDelay,
      reconnectionDelayMax: this.config.reconnectionDelay! * 10,
    });

    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.emit('connection', { status: 'connected' });
    });

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.emit('connection', { status: 'disconnected', reason });
    });

    this.socket.on('detection', (data: ArrayBuffer | any) => {
      if (data instanceof ArrayBuffer) {
        this.handleBinaryMessage(data);
      } else {
        this.handleJsonMessage(data);
      }
    });

    this.socket.on('entity_update', (data: EntityProfile) => {
      this.emit('entity', data);
    });

    this.socket.on('model_metrics', (data: ModelMetrics) => {
      this.emit('metrics', data);
    });

    this.socket.on('error', (error) => {
      console.error('WebSocket error:', error);
      this.emit('error', error);
    });
  }

  private handleBinaryMessage(buffer: ArrayBuffer): void {
    if (!this.protoRoot) {
      console.warn('Cannot decode binary message without protobuf schema');
      return;
    }

    try {
      const DetectionMessage = this.protoRoot.lookupType('DetectionEvent');
      const message = DetectionMessage.decode(new Uint8Array(buffer));
      const event = DetectionMessage.toObject(message) as DetectionEvent;
      
      this.bufferMessage(event);
    } catch (error) {
      console.error('Failed to decode protobuf message:', error);
    }
  }

  private handleJsonMessage(data: any): void {
    this.bufferMessage(data as DetectionEvent);
  }

  private bufferMessage(event: DetectionEvent): void {
    this.messageBuffer.push(event);
    
    if (!this.batchTimer) {
      this.batchTimer = setTimeout(() => {
        this.flushBuffer();
      }, this.config.batchingWindow);
    }
  }

  private flushBuffer(): void {
    if (this.messageBuffer.length > 0) {
      this.emit('detectionBatch', this.messageBuffer);
      this.messageBuffer = [];
    }
    this.batchTimer = null;
  }

  on(event: string, callback: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  off(event: string, callback: Function): void {
    this.listeners.get(event)?.delete(callback);
  }

  private emit(event: string, data: any): void {
    this.listeners.get(event)?.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error(`Error in event handler for ${event}:`, error);
      }
    });
  }

  subscribe(channel: string): void {
    this.socket?.emit('subscribe', { channel });
  }

  unsubscribe(channel: string): void {
    this.socket?.emit('unsubscribe', { channel });
  }

  disconnect(): void {
    if (this.batchTimer) {
      clearTimeout(this.batchTimer);
      this.flushBuffer();
    }
    this.socket?.disconnect();
    this.socket = null;
  }

  getConnectionStatus(): boolean {
    return this.socket?.connected || false;
  }
}

// Singleton instance
let wsManager: WebSocketManager | null = null;

export const getWebSocketManager = (): WebSocketManager => {
  if (!wsManager) {
    wsManager = new WebSocketManager({
      url: 'ws://localhost:4000',
      batchingWindow: 15,
    });
  }
  return wsManager;
};

export default WebSocketManager;