/**
 * WebSocket Base Service
 * Advanced WebSocket client with automatic reconnection, heartbeat, and message queuing
 */

import { API_CONFIG } from '../config/api.config';
import { WSMessage } from '../types/api.types';

export type WSEventHandler = (data: any) => void;
export type WSErrorHandler = (error: Error) => void;

export interface WSOptions {
  reconnect?: boolean;
  reconnectAttempts?: number;
  reconnectDelay?: number;
  reconnectDelayMax?: number;
  heartbeatInterval?: number;
  messageTimeout?: number;
  protocols?: string[];
}

export enum WSReadyState {
  CONNECTING = 0,
  OPEN = 1,
  CLOSING = 2,
  CLOSED = 3,
}

class WebSocketBase {
  protected ws: WebSocket | null = null;
  protected url: string;
  protected options: WSOptions;
  protected reconnectAttempt = 0;
  protected reconnectTimer: NodeJS.Timeout | null = null;
  protected heartbeatTimer: NodeJS.Timeout | null = null;
  protected messageHandlers = new Map<string, Set<WSEventHandler>>();
  protected errorHandlers = new Set<WSErrorHandler>();
  protected connectionPromise: Promise<void> | null = null;
  protected messageQueue: WSMessage[] = [];
  protected isIntentionallyClosed = false;
  protected lastPingTime = 0;
  protected latency = 0;

  constructor(endpoint: string, options: WSOptions = {}) {
    this.url = `${API_CONFIG.WS_BASE_URL}${endpoint}`;
    this.options = {
      reconnect: options.reconnect ?? API_CONFIG.websocket.reconnect,
      reconnectAttempts: options.reconnectAttempts ?? API_CONFIG.websocket.reconnectAttempts,
      reconnectDelay: options.reconnectDelay ?? API_CONFIG.websocket.reconnectDelay,
      reconnectDelayMax: options.reconnectDelayMax ?? API_CONFIG.websocket.reconnectDelayMax,
      heartbeatInterval: options.heartbeatInterval ?? API_CONFIG.websocket.heartbeatInterval,
      messageTimeout: options.messageTimeout ?? API_CONFIG.websocket.messageTimeout,
      protocols: options.protocols,
    };
  }

  /**
   * Connect to WebSocket server
   */
  async connect(): Promise<void> {
    if (this.ws?.readyState === WSReadyState.OPEN) {
      return Promise.resolve();
    }

    if (this.connectionPromise) {
      return this.connectionPromise;
    }

    this.isIntentionallyClosed = false;
    
    this.connectionPromise = new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url, this.options.protocols);
        
        const timeout = setTimeout(() => {
          reject(new Error('WebSocket connection timeout'));
          this.ws?.close();
        }, this.options.messageTimeout!);

        this.ws.onopen = () => {
          clearTimeout(timeout);
          this.reconnectAttempt = 0;
          this.connectionPromise = null;
          this.startHeartbeat();
          this.flushMessageQueue();
          this.emit('connected', { timestamp: Date.now() });
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event);
        };

        this.ws.onerror = (event) => {
          clearTimeout(timeout);
          const error = new Error('WebSocket error');
          this.handleError(error);
          
          if (this.connectionPromise) {
            reject(error);
            this.connectionPromise = null;
          }
        };

        this.ws.onclose = (event) => {
          clearTimeout(timeout);
          this.stopHeartbeat();
          this.emit('disconnected', { code: event.code, reason: event.reason });
          
          if (this.connectionPromise) {
            reject(new Error(`WebSocket closed: ${event.reason || 'Unknown reason'}`));
            this.connectionPromise = null;
          }

          if (!this.isIntentionallyClosed && this.options.reconnect) {
            this.scheduleReconnect();
          }
        };
      } catch (error) {
        this.connectionPromise = null;
        reject(error);
      }
    });

    return this.connectionPromise;
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    this.isIntentionallyClosed = true;
    this.stopHeartbeat();
    this.cancelReconnect();
    
    if (this.ws) {
      if (this.ws.readyState === WSReadyState.OPEN) {
        this.ws.close(1000, 'Client disconnect');
      }
      this.ws = null;
    }
    
    this.messageQueue = [];
    this.connectionPromise = null;
  }

  /**
   * Send a message through WebSocket
   */
  send<T = any>(type: string, payload: T): void {
    const message: WSMessage<T> = {
      type,
      payload,
      timestamp: Date.now(),
      id: this.generateMessageId(),
    };

    if (this.ws?.readyState === WSReadyState.OPEN) {
      try {
        this.ws.send(JSON.stringify(message));
      } catch (error) {
        this.handleError(error as Error);
        this.queueMessage(message);
      }
    } else {
      this.queueMessage(message);
      
      // Try to reconnect if not already attempting
      if (this.options.reconnect && !this.reconnectTimer) {
        this.connect().catch(() => {
          // Reconnection will be handled by scheduleReconnect
        });
      }
    }
  }

  /**
   * Subscribe to message events
   */
  on(event: string, handler: WSEventHandler): () => void {
    if (!this.messageHandlers.has(event)) {
      this.messageHandlers.set(event, new Set());
    }
    
    this.messageHandlers.get(event)!.add(handler);
    
    // Return unsubscribe function
    return () => {
      this.off(event, handler);
    };
  }

  /**
   * Unsubscribe from message events
   */
  off(event: string, handler: WSEventHandler): void {
    const handlers = this.messageHandlers.get(event);
    if (handlers) {
      handlers.delete(handler);
      if (handlers.size === 0) {
        this.messageHandlers.delete(event);
      }
    }
  }

  /**
   * Subscribe to error events
   */
  onError(handler: WSErrorHandler): () => void {
    this.errorHandlers.add(handler);
    return () => {
      this.errorHandlers.delete(handler);
    };
  }

  /**
   * Get current connection state
   */
  get readyState(): WSReadyState {
    return this.ws?.readyState ?? WSReadyState.CLOSED;
  }

  /**
   * Get connection status
   */
  get isConnected(): boolean {
    return this.ws?.readyState === WSReadyState.OPEN;
  }

  /**
   * Get current latency
   */
  getLatency(): number {
    return this.latency;
  }

  /**
   * Handle incoming messages
   */
  protected handleMessage(event: MessageEvent): void {
    try {
      const data = JSON.parse(event.data);
      
      // Handle ping/pong for latency measurement
      if (data.type === 'pong') {
        this.latency = Date.now() - this.lastPingTime;
        this.emit('latency', { latency: this.latency });
        return;
      }
      
      // Emit to specific handlers
      this.emit(data.type, data.payload || data);
      
      // Emit to global handlers
      this.emit('message', data);
    } catch (error) {
      this.handleError(new Error(`Failed to parse message: ${error}`));
    }
  }

  /**
   * Handle errors
   */
  protected handleError(error: Error): void {
    console.error('[WebSocket Error]', error);
    this.errorHandlers.forEach(handler => {
      try {
        handler(error);
      } catch (e) {
        console.error('Error in error handler:', e);
      }
    });
  }

  /**
   * Emit event to handlers
   */
  protected emit(event: string, data: any): void {
    const handlers = this.messageHandlers.get(event);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in handler for event '${event}':`, error);
        }
      });
    }
  }

  /**
   * Start heartbeat mechanism
   */
  protected startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WSReadyState.OPEN) {
        this.lastPingTime = Date.now();
        this.send('ping', { timestamp: this.lastPingTime });
      }
    }, this.options.heartbeatInterval!);
  }

  /**
   * Stop heartbeat mechanism
   */
  protected stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * Schedule reconnection attempt
   */
  protected scheduleReconnect(): void {
    if (this.reconnectAttempt >= this.options.reconnectAttempts!) {
      this.emit('reconnect-failed', { attempts: this.reconnectAttempt });
      return;
    }

    const delay = Math.min(
      this.options.reconnectDelay! * Math.pow(2, this.reconnectAttempt),
      this.options.reconnectDelayMax!
    );
    
    this.reconnectAttempt++;
    
    this.emit('reconnecting', { 
      attempt: this.reconnectAttempt, 
      maxAttempts: this.options.reconnectAttempts,
      delay 
    });
    
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect().catch(() => {
        // Will trigger another reconnect attempt
      });
    }, delay);
  }

  /**
   * Cancel scheduled reconnection
   */
  protected cancelReconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.reconnectAttempt = 0;
  }

  /**
   * Queue message for later sending
   */
  protected queueMessage(message: WSMessage): void {
    this.messageQueue.push(message);
    
    // Limit queue size to prevent memory issues
    if (this.messageQueue.length > 100) {
      this.messageQueue.shift();
    }
  }

  /**
   * Send all queued messages
   */
  protected flushMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.ws?.readyState === WSReadyState.OPEN) {
      const message = this.messageQueue.shift()!;
      try {
        this.ws.send(JSON.stringify(message));
      } catch (error) {
        this.handleError(error as Error);
        this.messageQueue.unshift(message);
        break;
      }
    }
  }

  /**
   * Generate unique message ID
   */
  protected generateMessageId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}

export default WebSocketBase;