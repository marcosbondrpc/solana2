import { EventEmitter } from 'eventemitter3';
import ReconnectingWebSocket from 'reconnecting-websocket';
import PQueue from 'p-queue';
import pRetry from 'p-retry';
import { compress, decompress } from '../utils/compression';
import { MessageParser } from '../utils/messageParser';
import { ExponentialBackoff } from '../utils/backoff';
import { HeartbeatManager } from '../utils/heartbeat';
import type {
  WebSocketConfig,
  WebSocketMessage,
  WebSocketState,
  SubscriptionHandler,
  ConnectionEvents,
} from '../types';

export class WebSocketClient extends EventEmitter<ConnectionEvents> {
  private ws: ReconnectingWebSocket | null = null;
  private config: WebSocketConfig;
  private state: WebSocketState = 'disconnected';
  private subscriptions = new Map<string, Set<SubscriptionHandler>>();
  private messageQueue: PQueue;
  private messageParser: MessageParser;
  private backoff: ExponentialBackoff;
  private heartbeat: HeartbeatManager;
  private decoder: Worker | null = null;
  private encoder: Worker | null = null;
  private messageBuffer: ArrayBuffer[] = [];
  private lastMessageTime = 0;
  private reconnectAttempts = 0;
  private metrics = {
    messagesReceived: 0,
    messagesSent: 0,
    bytesReceived: 0,
    bytesSent: 0,
    latency: 0,
    reconnects: 0,
  };

  constructor(config: WebSocketConfig) {
    super();
    this.config = {
      reconnectInterval: 1000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      messageQueueSize: 1000,
      enableCompression: true,
      enableBinary: true,
      enableWorkers: true,
      ...config,
    };

    this.messageQueue = new PQueue({
      concurrency: 1,
      interval: 10,
      intervalCap: 100,
    });

    this.messageParser = new MessageParser(this.config);
    this.backoff = new ExponentialBackoff({
      min: this.config.reconnectInterval,
      max: 30000,
      factor: 1.5,
    });

    this.heartbeat = new HeartbeatManager(
      this.config.heartbeatInterval,
      () => this.send({ type: 'ping' }),
      () => this.reconnect()
    );

    if (this.config.enableWorkers && typeof Worker !== 'undefined') {
      this.initializeWorkers();
    }
  }

  private initializeWorkers() {
    try {
      this.decoder = new Worker(
        new URL('../workers/decoder.worker.ts', import.meta.url),
        { type: 'module' }
      );
      
      this.encoder = new Worker(
        new URL('../workers/encoder.worker.ts', import.meta.url),
        { type: 'module' }
      );

      this.decoder.onmessage = (event) => {
        const { id, data, error } = event.data;
        if (error) {
          console.error('Decoder error:', error);
          return;
        }
        this.handleParsedMessage(data);
      };

      this.encoder.onmessage = (event) => {
        const { id, data, error } = event.data;
        if (error) {
          console.error('Encoder error:', error);
          return;
        }
        this.sendRaw(data);
      };
    } catch (error) {
      console.warn('Failed to initialize workers:', error);
      this.config.enableWorkers = false;
    }
  }

  async connect(): Promise<void> {
    if (this.state === 'connected' || this.state === 'connecting') {
      return;
    }

    this.setState('connecting');

    try {
      await pRetry(
        async () => {
          this.ws = new ReconnectingWebSocket(
            this.config.url,
            this.config.protocols,
            {
              maxReconnectionDelay: 10000,
              minReconnectionDelay: this.config.reconnectInterval,
              reconnectionDelayGrowFactor: 1.3,
              maxRetries: this.config.maxReconnectAttempts,
              binaryType: 'arraybuffer',
            }
          );

          this.setupEventHandlers();

          return new Promise<void>((resolve, reject) => {
            const timeout = setTimeout(() => {
              reject(new Error('Connection timeout'));
            }, 10000);

            this.ws!.addEventListener('open', () => {
              clearTimeout(timeout);
              resolve();
            });

            this.ws!.addEventListener('error', (error) => {
              clearTimeout(timeout);
              reject(error);
            });
          });
        },
        {
          retries: 3,
          onFailedAttempt: (error) => {
            console.warn(`Connection attempt ${error.attemptNumber} failed:`, error.message);
            this.reconnectAttempts++;
          },
        }
      );

      this.setState('connected');
      this.heartbeat.start();
      this.emit('connected');
      this.metrics.reconnects = this.reconnectAttempts;
      this.reconnectAttempts = 0;

      // Process buffered messages
      while (this.messageBuffer.length > 0) {
        const buffer = this.messageBuffer.shift()!;
        await this.handleMessage(buffer);
      }
    } catch (error) {
      this.setState('error');
      this.emit('error', error as Error);
      throw error;
    }
  }

  private setupEventHandlers() {
    if (!this.ws) return;

    this.ws.addEventListener('message', async (event) => {
      this.lastMessageTime = Date.now();
      this.metrics.messagesReceived++;

      if (event.data instanceof ArrayBuffer) {
        this.metrics.bytesReceived += event.data.byteLength;
        
        if (this.state !== 'connected') {
          this.messageBuffer.push(event.data);
          return;
        }

        await this.handleMessage(event.data);
      } else {
        const data = JSON.parse(event.data as string);
        this.handleParsedMessage(data);
      }
    });

    this.ws.addEventListener('close', (event) => {
      this.setState('disconnected');
      this.heartbeat.stop();
      this.emit('disconnected', { code: event.code, reason: event.reason });
    });

    this.ws.addEventListener('error', (event) => {
      console.error('WebSocket error:', event);
      this.emit('error', new Error('WebSocket error'));
    });
  }

  private async handleMessage(buffer: ArrayBuffer) {
    const startTime = performance.now();

    try {
      if (this.config.enableWorkers && this.decoder) {
        this.decoder.postMessage({ 
          id: Date.now(), 
          buffer,
          compressed: this.config.enableCompression 
        });
      } else {
        let data = buffer;
        
        if (this.config.enableCompression) {
          data = await decompress(data);
        }

        const message = this.messageParser.parse(data);
        this.handleParsedMessage(message);
      }
    } catch (error) {
      console.error('Failed to handle message:', error);
      this.emit('error', error as Error);
    }

    this.metrics.latency = performance.now() - startTime;
  }

  private handleParsedMessage(message: WebSocketMessage) {
    // Handle system messages
    if (message.type === 'pong') {
      this.heartbeat.handlePong();
      return;
    }

    // Emit raw message event
    this.emit('message', message);

    // Handle subscriptions
    if (message.channel) {
      const handlers = this.subscriptions.get(message.channel);
      if (handlers) {
        handlers.forEach((handler) => {
          try {
            handler(message);
          } catch (error) {
            console.error('Subscription handler error:', error);
          }
        });
      }
    }
  }

  async send(message: WebSocketMessage): Promise<void> {
    if (this.state !== 'connected') {
      throw new Error('WebSocket is not connected');
    }

    return this.messageQueue.add(async () => {
      const startTime = performance.now();

      try {
        if (this.config.enableWorkers && this.encoder) {
          this.encoder.postMessage({
            id: Date.now(),
            message,
            compress: this.config.enableCompression,
          });
        } else {
          let data = this.messageParser.serialize(message);
          
          if (this.config.enableCompression) {
            data = await compress(data);
          }

          this.sendRaw(data);
        }

        this.metrics.messagesSent++;
        this.metrics.latency = performance.now() - startTime;
      } catch (error) {
        console.error('Failed to send message:', error);
        throw error;
      }
    });
  }

  private sendRaw(data: ArrayBuffer | string) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket is not open');
    }

    this.ws.send(data);
    
    if (data instanceof ArrayBuffer) {
      this.metrics.bytesSent += data.byteLength;
    } else {
      this.metrics.bytesSent += new Blob([data]).size;
    }
  }

  subscribe(channel: string, handler: SubscriptionHandler): () => void {
    if (!this.subscriptions.has(channel)) {
      this.subscriptions.set(channel, new Set());
      
      // Send subscription message
      this.send({
        type: 'subscribe',
        channel,
      }).catch(console.error);
    }

    this.subscriptions.get(channel)!.add(handler);

    // Return unsubscribe function
    return () => {
      const handlers = this.subscriptions.get(channel);
      if (handlers) {
        handlers.delete(handler);
        
        if (handlers.size === 0) {
          this.subscriptions.delete(channel);
          
          // Send unsubscribe message
          this.send({
            type: 'unsubscribe',
            channel,
          }).catch(console.error);
        }
      }
    };
  }

  async reconnect(): Promise<void> {
    this.disconnect();
    await this.connect();
  }

  disconnect() {
    this.setState('disconnected');
    this.heartbeat.stop();
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    if (this.decoder) {
      this.decoder.terminate();
      this.decoder = null;
    }

    if (this.encoder) {
      this.encoder.terminate();
      this.encoder = null;
    }

    this.subscriptions.clear();
    this.messageBuffer = [];
    this.emit('disconnected', { code: 1000, reason: 'Manual disconnect' });
  }

  private setState(state: WebSocketState) {
    this.state = state;
    this.emit('stateChange', state);
  }

  getState(): WebSocketState {
    return this.state;
  }

  getMetrics() {
    return { ...this.metrics };
  }

  isConnected(): boolean {
    return this.state === 'connected';
  }

  destroy() {
    this.disconnect();
    this.removeAllListeners();
    this.messageQueue.clear();
  }
}