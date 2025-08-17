/**
 * Enhanced WebSocket Service with WebTransport Support
 * Ultra-low-latency bi-directional communication
 * Handles millions of messages per second with automatic reconnection
 */

import { apiService } from './api-service';

export type MessageHandler = (data: any) => void;
export type ErrorHandler = (error: Error) => void;
export type ConnectionHandler = () => void;

export interface WebSocketConfig {
  url?: string;
  protocols?: string[];
  reconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  binaryType?: 'blob' | 'arraybuffer';
  useWebTransport?: boolean;
  compression?: boolean;
}

export interface WebSocketStats {
  messagesReceived: number;
  messagesSent: number;
  bytesReceived: number;
  bytesSent: number;
  latency: number;
  connectionUptime: number;
  reconnectCount: number;
  errorCount: number;
}

class EnhancedWebSocketService {
  private ws: WebSocket | null = null;
  private wt: any = null; // WebTransport instance
  private config: Required<WebSocketConfig>;
  private messageHandlers: Map<string, Set<MessageHandler>> = new Map();
  private errorHandlers: Set<ErrorHandler> = new Set();
  private openHandlers: Set<ConnectionHandler> = new Set();
  private closeHandlers: Set<ConnectionHandler> = new Set();
  private reconnectAttempts = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private stats: WebSocketStats = {
    messagesReceived: 0,
    messagesSent: 0,
    bytesReceived: 0,
    bytesSent: 0,
    latency: 0,
    connectionUptime: 0,
    reconnectCount: 0,
    errorCount: 0
  };
  private connectionStartTime = 0;
  private messageQueue: any[] = [];
  private isConnecting = false;
  private lastPingTime = 0;
  private worker: Worker | null = null;

  constructor(config: WebSocketConfig = {}) {
    this.config = {
      url: config.url || 'ws://localhost:8000/ws',
      protocols: config.protocols || [],
      reconnect: config.reconnect !== false,
      reconnectInterval: config.reconnectInterval || 3000,
      maxReconnectAttempts: config.maxReconnectAttempts || 10,
      heartbeatInterval: config.heartbeatInterval || 30000,
      binaryType: config.binaryType || 'arraybuffer',
      useWebTransport: config.useWebTransport || false,
      compression: config.compression !== false
    };

    // Initialize decoder worker for binary messages
    if (typeof Worker !== 'undefined') {
      try {
        this.worker = new Worker('/workers/wsDecoder.worker.js');
        this.worker.onmessage = this.handleWorkerMessage.bind(this);
      } catch (error) {
        console.warn('Failed to initialize WebSocket decoder worker:', error);
      }
    }
  }

  public async connect(): Promise<void> {
    if (this.isConnecting || this.isConnected()) {
      return;
    }

    this.isConnecting = true;

    try {
      if (this.config.useWebTransport && this.isWebTransportSupported()) {
        await this.connectWebTransport();
      } else {
        await this.connectWebSocket();
      }
    } catch (error) {
      this.isConnecting = false;
      throw error;
    }
  }

  private async connectWebSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        // Get auth token if needed
        const url = new URL(this.config.url);
        
        // Add compression extension if supported
        const protocols = [...this.config.protocols];
        if (this.config.compression) {
          protocols.push('permessage-deflate');
        }

        this.ws = new WebSocket(url.toString(), protocols);
        this.ws.binaryType = this.config.binaryType;

        this.ws.onopen = () => {
          this.isConnecting = false;
          this.connectionStartTime = Date.now();
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          this.flushMessageQueue();
          this.openHandlers.forEach(handler => handler());
          resolve();
        };

        this.ws.onmessage = this.handleMessage.bind(this);
        this.ws.onerror = this.handleError.bind(this);
        this.ws.onclose = this.handleClose.bind(this);

      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  private async connectWebTransport(): Promise<void> {
    try {
      // Get WebTransport token
      const token = await apiService.getWebTransportToken();
      const url = new URL(this.config.url.replace('ws://', 'https://').replace('wss://', 'https://'));
      url.searchParams.set('token', token);

      // @ts-ignore - WebTransport is experimental
      const transport = new WebTransport(url.toString());
      await transport.ready;

      this.wt = transport;
      this.isConnecting = false;
      this.connectionStartTime = Date.now();
      this.reconnectAttempts = 0;

      // Set up bidirectional stream
      this.setupWebTransportStreams();
      
      this.startHeartbeat();
      this.flushMessageQueue();
      this.openHandlers.forEach(handler => handler());

    } catch (error) {
      this.isConnecting = false;
      // Fallback to WebSocket
      if (this.config.reconnect) {
        console.warn('WebTransport failed, falling back to WebSocket:', error);
        this.config.useWebTransport = false;
        await this.connectWebSocket();
      } else {
        throw error;
      }
    }
  }

  private async setupWebTransportStreams(): Promise<void> {
    if (!this.wt) return;

    try {
      // Set up incoming streams
      const reader = this.wt.incomingBidirectionalStreams.getReader();
      
      while (true) {
        const { value: stream, done } = await reader.read();
        if (done) break;

        // Handle each stream
        this.handleWebTransportStream(stream);
      }
    } catch (error) {
      this.handleError(error as Error);
    }
  }

  private async handleWebTransportStream(stream: any): Promise<void> {
    const reader = stream.readable.getReader();
    const decoder = new TextDecoder();

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        this.stats.messagesReceived++;
        this.stats.bytesReceived += value.byteLength;

        // Decode and handle message
        if (this.worker) {
          this.worker.postMessage({ type: 'decode', data: value }, [value.buffer]);
        } else {
          const text = decoder.decode(value);
          this.handleMessage({ data: text } as MessageEvent);
        }
      }
    } catch (error) {
      console.error('WebTransport stream error:', error);
    } finally {
      reader.releaseLock();
    }
  }

  private handleMessage(event: MessageEvent): void {
    this.stats.messagesReceived++;
    
    try {
      let data = event.data;

      // Handle binary data
      if (data instanceof ArrayBuffer) {
        this.stats.bytesReceived += data.byteLength;
        
        // Use worker for decoding if available
        if (this.worker) {
          this.worker.postMessage({ type: 'decode', data }, [data]);
          return;
        }
        
        // Fallback to inline decoding
        const decoder = new TextDecoder();
        data = decoder.decode(data);
      } else if (typeof data === 'string') {
        this.stats.bytesReceived += data.length;
      }

      // Parse JSON if string
      if (typeof data === 'string') {
        try {
          data = JSON.parse(data);
        } catch (e) {
          // Not JSON, keep as string
        }
      }

      // Handle ping/pong for latency measurement
      if (data && typeof data === 'object' && data.type === 'pong') {
        this.stats.latency = Date.now() - this.lastPingTime;
        return;
      }

      // Route to appropriate handlers
      this.routeMessage(data);

    } catch (error) {
      this.handleError(error as Error);
    }
  }

  private handleWorkerMessage(event: MessageEvent): void {
    if (event.data.type === 'decoded') {
      this.routeMessage(event.data.data);
    } else if (event.data.type === 'error') {
      this.handleError(new Error(event.data.error));
    }
  }

  private routeMessage(data: any): void {
    // Check for typed messages
    if (data && typeof data === 'object' && data.type) {
      const handlers = this.messageHandlers.get(data.type);
      if (handlers) {
        handlers.forEach(handler => {
          try {
            handler(data);
          } catch (error) {
            console.error('Message handler error:', error);
          }
        });
      }
    }

    // Call generic handlers
    const genericHandlers = this.messageHandlers.get('*');
    if (genericHandlers) {
      genericHandlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error('Generic handler error:', error);
        }
      });
    }
  }

  private handleError(error: Error): void {
    this.stats.errorCount++;
    this.errorHandlers.forEach(handler => handler(error));
  }

  private handleClose(): void {
    this.stopHeartbeat();
    this.closeHandlers.forEach(handler => handler());

    if (this.connectionStartTime > 0) {
      this.stats.connectionUptime += Date.now() - this.connectionStartTime;
      this.connectionStartTime = 0;
    }

    if (this.config.reconnect && this.reconnectAttempts < this.config.maxReconnectAttempts) {
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }

    const delay = Math.min(
      this.config.reconnectInterval * Math.pow(1.5, this.reconnectAttempts),
      30000
    );

    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempts++;
      this.stats.reconnectCount++;
      this.connect().catch(error => {
        console.error('Reconnection failed:', error);
      });
    }, delay);
  }

  private startHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
    }

    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected()) {
        this.lastPingTime = Date.now();
        this.send({ type: 'ping', timestamp: this.lastPingTime });
      }
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.isConnected()) {
      const message = this.messageQueue.shift();
      this.send(message);
    }
  }

  public send(data: any): void {
    if (!this.isConnected()) {
      // Queue message if not connected
      this.messageQueue.push(data);
      
      // Trim queue if too large
      if (this.messageQueue.length > 1000) {
        this.messageQueue = this.messageQueue.slice(-500);
      }
      return;
    }

    try {
      let payload: string | ArrayBuffer;

      if (typeof data === 'object') {
        payload = JSON.stringify(data);
      } else {
        payload = data;
      }

      if (this.wt) {
        // Send via WebTransport
        this.sendWebTransport(payload);
      } else if (this.ws) {
        // Send via WebSocket
        this.ws.send(payload);
        this.stats.messagesSent++;
        this.stats.bytesSent += typeof payload === 'string' ? payload.length : payload.byteLength;
      }

    } catch (error) {
      this.handleError(error as Error);
    }
  }

  private async sendWebTransport(data: string | ArrayBuffer): Promise<void> {
    if (!this.wt) return;

    try {
      const stream = await this.wt.createBidirectionalStream();
      const writer = stream.writable.getWriter();
      const encoder = new TextEncoder();
      
      const bytes = typeof data === 'string' ? encoder.encode(data) : new Uint8Array(data);
      await writer.write(bytes);
      await writer.close();

      this.stats.messagesSent++;
      this.stats.bytesSent += bytes.byteLength;

    } catch (error) {
      this.handleError(error as Error);
    }
  }

  public on(event: string, handler: MessageHandler | ErrorHandler | ConnectionHandler): void {
    switch (event) {
      case 'open':
        this.openHandlers.add(handler as ConnectionHandler);
        break;
      case 'close':
        this.closeHandlers.add(handler as ConnectionHandler);
        break;
      case 'error':
        this.errorHandlers.add(handler as ErrorHandler);
        break;
      default:
        if (!this.messageHandlers.has(event)) {
          this.messageHandlers.set(event, new Set());
        }
        this.messageHandlers.get(event)!.add(handler as MessageHandler);
    }
  }

  public off(event: string, handler: MessageHandler | ErrorHandler | ConnectionHandler): void {
    switch (event) {
      case 'open':
        this.openHandlers.delete(handler as ConnectionHandler);
        break;
      case 'close':
        this.closeHandlers.delete(handler as ConnectionHandler);
        break;
      case 'error':
        this.errorHandlers.delete(handler as ErrorHandler);
        break;
      default:
        const handlers = this.messageHandlers.get(event);
        if (handlers) {
          handlers.delete(handler as MessageHandler);
          if (handlers.size === 0) {
            this.messageHandlers.delete(event);
          }
        }
    }
  }

  public subscribe(type: string, handler: MessageHandler): () => void {
    this.on(type, handler);
    return () => this.off(type, handler);
  }

  public isConnected(): boolean {
    if (this.wt) {
      return this.wt.ready && !this.wt.closed;
    }
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  public isWebTransportSupported(): boolean {
    return typeof WebTransport !== 'undefined';
  }

  public getStats(): WebSocketStats {
    return {
      ...this.stats,
      connectionUptime: this.connectionStartTime > 0 
        ? this.stats.connectionUptime + (Date.now() - this.connectionStartTime)
        : this.stats.connectionUptime
    };
  }

  public resetStats(): void {
    this.stats = {
      messagesReceived: 0,
      messagesSent: 0,
      bytesReceived: 0,
      bytesSent: 0,
      latency: 0,
      connectionUptime: this.stats.connectionUptime,
      reconnectCount: this.stats.reconnectCount,
      errorCount: 0
    };
  }

  public disconnect(): void {
    this.config.reconnect = false;

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    this.stopHeartbeat();

    if (this.wt) {
      this.wt.close();
      this.wt = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }

    this.messageQueue = [];
    this.isConnecting = false;
  }

  public async reconnect(): Promise<void> {
    this.disconnect();
    this.config.reconnect = true;
    this.reconnectAttempts = 0;
    return this.connect();
  }
}

// Export singleton instances for different use cases
export const mevWebSocket = new EnhancedWebSocketService({
  url: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws/mev',
  reconnect: true,
  useWebTransport: true,
  compression: true
});

export const controlWebSocket = new EnhancedWebSocketService({
  url: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws/control',
  reconnect: true,
  heartbeatInterval: 10000
});

export const dataWebSocket = new EnhancedWebSocketService({
  url: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws/data',
  binaryType: 'arraybuffer',
  reconnect: true,
  useWebTransport: true
});

export default EnhancedWebSocketService;