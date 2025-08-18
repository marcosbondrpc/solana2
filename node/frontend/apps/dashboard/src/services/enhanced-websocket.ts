/**
 * Enhanced WebSocket Service with Multi-Protocol Support
 * Handles WebSocket, WebTransport, and Server-Sent Events
 */

import { io, Socket } from 'socket.io-client';
import { useMEVStore } from '../stores/mev-store';
import { useControlStore } from '../stores/control-store';
import { useBanditStore } from '../stores/bandit-store';

// Protocol Buffer support
interface ProtobufDecoder {
  decode(data: ArrayBuffer): any;
}


export class EnhancedWebSocketService {
  private sockets: Map<string, Socket> = new Map();
  private eventSources: Map<string, EventSource> = new Map();
  private webTransports: Map<string, any> = new Map();
  private reconnectTimers: Map<string, NodeJS.Timeout> = new Map();
  private messageHandlers: Map<string, Set<Function>> = new Map();
  private protobufDecoders: Map<string, ProtobufDecoder> = new Map();
  
  // Performance tracking
  private metrics = {
    messagesReceived: 0,
    bytesReceived: 0,
    latencies: [] as number[],
    errors: 0,
  };

  // Service endpoints
  private readonly services = {
    mev: {
      url: import.meta.env.VITE_MEV_WS_URL || 'ws://localhost:8001',
      protocol: 'websocket' as const,
      topics: ['opportunities', 'bundles', 'metrics'],
    },
    control: {
      url: import.meta.env.VITE_CONTROL_WS_URL || 'ws://localhost:8002',
      protocol: 'websocket' as const,
      topics: ['commands', 'status', 'alerts'],
    },
    kafka: {
      url: import.meta.env.VITE_KAFKA_WS_URL || 'ws://localhost:8003',
      protocol: 'websocket' as const,
      topics: ['trades', 'blocks', 'transactions'],
    },
    clickhouse: {
      url: import.meta.env.VITE_CLICKHOUSE_SSE_URL || 'http://localhost:8004/events',
      protocol: 'sse' as const,
      topics: ['queries', 'metrics'],
    },
    ml: {
      url: import.meta.env.VITE_ML_WS_URL || 'ws://localhost:8005',
      protocol: 'websocket' as const,
      topics: ['predictions', 'training', 'models'],
    },
    webtransport: {
      url: import.meta.env.VITE_WT_URL || 'https://localhost:8443',
      protocol: 'webtransport' as const,
      topics: ['realtime', 'protobuf'],
    },
  };

  constructor() {
    this.initializeProtobufDecoders();
  }

  private initializeProtobufDecoders(): void {
    // Initialize protobuf decoders for binary message handling
    // These would be loaded from generated protobuf files
    this.protobufDecoders.set('mev', {
      decode: (data: ArrayBuffer) => {
        // Decode MEV protobuf messages
        const view = new DataView(data);
        // Simplified decoding logic
        return {
          type: view.getUint8(0),
          timestamp: view.getBigUint64(1, true),
          // ... more fields
        };
      },
    });
  }

  public async connectAll(): Promise<void> {
    const connections = Object.entries(this.services).map(([name, config]) => 
      this.connectService(name, config)
    );
    
    await Promise.allSettled(connections);
  }

  private async connectService(
    name: string,
    config: typeof this.services[keyof typeof this.services]
  ): Promise<void> {
    try {
      switch (config.protocol) {
        case 'websocket':
          await this.connectWebSocket(name, config);
          break;
        case 'sse':
          await this.connectSSE(name, config);
          break;
        case 'webtransport':
          await this.connectWebTransport(name, config);
          break;
      }
    } catch (error) {
      console.error(`Failed to connect to ${name}:`, error);
      this.scheduleReconnect(name, config);
    }
  }

  private async connectWebSocket(
    name: string,
    config: typeof this.services[keyof typeof this.services]
  ): Promise<void> {
    const socket = io(config.url, {
      transports: ['websocket'],
      reconnection: false,
      timeout: 10000,
      query: {
        service: name,
        topics: config.topics.join(','),
      },
    });

    socket.on('connect', () => {
      console.log(`[WS] Connected to ${name}`);
      this.handleServiceConnected(name);
    });

    socket.on('disconnect', (reason) => {
      console.log(`[WS] Disconnected from ${name}:`, reason);
      this.handleServiceDisconnected(name);
      this.scheduleReconnect(name, config);
    });

    // Handle different message types
    socket.on('mev:opportunity', (data) => {
      this.handleMEVOpportunity(data);
    });

    socket.on('mev:bundle', (data) => {
      this.handleJitoBundle(data);
    });

    socket.on('control:command', (data) => {
      this.handleControlCommand(data);
    });

    socket.on('kafka:message', (data) => {
      this.handleKafkaMessage(data);
    });

    socket.on('ml:prediction', (data) => {
      this.handleMLPrediction(data);
    });

    socket.on('metrics', (data) => {
      this.handleMetrics(name, data);
    });

    socket.on('error', (error) => {
      console.error(`[WS] Error from ${name}:`, error);
      this.metrics.errors++;
    });

    // Binary message handling
    socket.on('binary', (data: ArrayBuffer) => {
      this.handleBinaryMessage(name, data);
    });

    this.sockets.set(name, socket);
  }

  private async connectSSE(
    name: string,
    config: typeof this.services[keyof typeof this.services]
  ): Promise<void> {
    const eventSource = new EventSource(config.url);

    eventSource.onopen = () => {
      console.log(`[SSE] Connected to ${name}`);
      this.handleServiceConnected(name);
    };

    eventSource.onerror = (error) => {
      console.error(`[SSE] Error from ${name}:`, error);
      this.handleServiceDisconnected(name);
      this.scheduleReconnect(name, config);
    };

    // Handle different event types
    eventSource.addEventListener('query', (event) => {
      this.handleClickHouseQuery(JSON.parse(event.data));
    });

    eventSource.addEventListener('metric', (event) => {
      this.handleMetrics(name, JSON.parse(event.data));
    });

    this.eventSources.set(name, eventSource);
  }

  private async connectWebTransport(
    name: string,
    config: typeof this.services[keyof typeof this.services]
  ): Promise<void> {
    // Check if WebTransport is available
    if (!('WebTransport' in window)) {
      console.warn('WebTransport not supported, falling back to WebSocket');
      return this.connectWebSocket(name, { ...config, protocol: 'websocket' });
    }

    try {
      const transport = new (window as any).WebTransport(config.url);
      await transport.ready;

      console.log(`[WT] Connected to ${name}`);
      this.handleServiceConnected(name);

      // Handle bidirectional streams
      const reader = transport.incomingBidirectionalStreams.getReader();
      this.processWebTransportStreams(name, reader);

      // Handle unidirectional streams
      const uniReader = transport.incomingUnidirectionalStreams.getReader();
      this.processWebTransportUniStreams(name, uniReader);

      // Handle datagrams for ultra-low latency
      if (transport.datagrams) {
        this.processWebTransportDatagrams(name, transport.datagrams);
      }

      this.webTransports.set(name, transport);
    } catch (error) {
      console.error(`[WT] Failed to connect to ${name}:`, error);
      this.scheduleReconnect(name, config);
    }
  }

  private async processWebTransportStreams(name: string, reader: any): Promise<void> {
    try {
      while (true) {
        const { value: stream, done } = await reader.read();
        if (done) break;

        const streamReader = stream.readable.getReader();
        const streamWriter = stream.writable.getWriter();

        // Process incoming messages
        this.processStreamMessages(name, streamReader);

        // Store writer for sending messages
        this.storeStreamWriter(name, streamWriter);
      }
    } catch (error) {
      console.error(`[WT] Stream error for ${name}:`, error);
    }
  }

  private async processWebTransportUniStreams(name: string, reader: any): Promise<void> {
    try {
      while (true) {
        const { value: stream, done } = await reader.read();
        if (done) break;

        const streamReader = stream.getReader();
        this.processStreamMessages(name, streamReader);
      }
    } catch (error) {
      console.error(`[WT] Uni-stream error for ${name}:`, error);
    }
  }

  private async processWebTransportDatagrams(name: string, datagrams: any): Promise<void> {
    const reader = datagrams.readable.getReader();
    
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        // Process ultra-low latency datagram
        this.handleDatagram(name, value);
      }
    } catch (error) {
      console.error(`[WT] Datagram error for ${name}:`, error);
    }
  }

  private async processStreamMessages(name: string, reader: any): Promise<void> {
    const decoder = new TextDecoder();
    
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        // Check if binary or text
        if (value instanceof Uint8Array) {
          // Try to decode as text first
          try {
            const text = decoder.decode(value);
            const message = JSON.parse(text);
            this.handleMessage(name, message);
          } catch {
            // Handle as binary protobuf
            this.handleBinaryMessage(name, value.buffer);
          }
        }
      }
    } catch (error) {
      console.error(`[WT] Message processing error for ${name}:`, error);
    }
  }

  private storeStreamWriter(name: string, writer: any): void {
    // Store writer for sending messages back
    // Implementation depends on your needs
  }

  // Message Handlers
  private handleMEVOpportunity(data: any): void {
    const store = useMEVStore.getState();
    
    store.addArbitrageOpportunity({
      id: data.id,
      timestamp: data.timestamp,
      dexA: data.dex_a,
      dexB: data.dex_b,
      tokenIn: data.token_in,
      tokenOut: data.token_out,
      amountIn: data.amount_in,
      expectedProfit: data.expected_profit,
      slippage: data.slippage,
      path: data.path,
      confidence: data.confidence,
      status: 'pending',
    });

    this.updateMetrics(data);
  }

  private handleJitoBundle(data: any): void {
    const store = useMEVStore.getState();
    
    store.addJitoBundle({
      id: data.id,
      timestamp: data.timestamp,
      bundleId: data.bundle_id,
      transactions: data.transactions,
      tip: data.tip,
      status: data.status,
      relayUrl: data.relay_url,
      submitLatency: data.submit_latency,
      mevType: data.mev_type,
    });
  }

  private handleControlCommand(data: any): void {
    const store = useControlStore.getState();
    store.processCommand(data);
  }

  private handleKafkaMessage(data: any): void {
    // Process Kafka stream messages
    const topic = data.topic;
    const handlers = this.messageHandlers.get(topic);
    
    if (handlers) {
      handlers.forEach(handler => handler(data));
    }
  }

  private handleMLPrediction(data: any): void {
    // Update ML predictions in stores
    const store = useBanditStore.getState();
    store.updatePrediction(data);
  }

  private handleClickHouseQuery(data: any): void {
    // Handle ClickHouse query results
    console.log('ClickHouse query result:', data);
  }

  private handleMetrics(service: string, data: any): void {
    const store = useMEVStore.getState();
    
    if (data.latency) {
      store.addLatencyMetric(data.latency);
    }
    
    if (data.system) {
      store.updateSystemPerformance(data.system);
    }
    
    if (data.profit) {
      store.updateProfitMetrics(data.profit);
    }
  }

  private handleBinaryMessage(service: string, data: ArrayBufferLike): void {
    const decoder = this.protobufDecoders.get(service);
    
    if (decoder) {
      try {
        const buf = data instanceof ArrayBuffer ? data : new Uint8Array(data as any).slice().buffer;
        const message = decoder.decode(buf);
        this.handleMessage(service, message);
      } catch (error) {
        console.error(`Failed to decode protobuf from ${service}:`, error);
      }
    }
  }

  private handleDatagram(service: string, data: ArrayBuffer): void {
    // Handle ultra-low latency datagrams
    const view = new DataView(data);
    const messageType = view.getUint8(0);
    
    switch (messageType) {
      case 0x01: // Heartbeat
        this.handleHeartbeat(service, view.getBigUint64(1, true));
        break;
      case 0x02: // Quick metric
        this.handleQuickMetric(service, view);
        break;
      default:
        console.warn(`Unknown datagram type ${messageType} from ${service}`);
    }
  }

  private handleMessage(service: string, message: any): void {
    // Generic message handler
    this.metrics.messagesReceived++;
    
    // Route to specific handlers based on message type
    if (message.type) {
      const handlers = this.messageHandlers.get(message.type);
      if (handlers) {
        handlers.forEach(handler => handler(message));
      }
    }
  }

  private handleHeartbeat(service: string, timestamp: bigint): void {
    const latency = Date.now() - Number(timestamp);
    this.metrics.latencies.push(latency);
    
    // Keep only last 100 latencies
    if (this.metrics.latencies.length > 100) {
      this.metrics.latencies.shift();
    }
  }

  private handleQuickMetric(service: string, view: DataView): void {
    // Handle quick metrics sent via datagram
    const metricType = view.getUint8(9);
    const value = view.getFloat64(10, true);
    
    // Update appropriate store
    console.log(`Quick metric from ${service}: type=${metricType}, value=${value}`);
  }

  private handleServiceConnected(service: string): void {
    // Clear reconnect timer if exists
    const timer = this.reconnectTimers.get(service);
    if (timer) {
      clearTimeout(timer);
      this.reconnectTimers.delete(service);
    }

    // Update connection status
    useMEVStore.getState().setConnectionStatus(true);
  }

  private handleServiceDisconnected(service: string): void {
    // Update connection status if all services disconnected
    const allDisconnected = 
      this.sockets.size === 0 && 
      this.eventSources.size === 0 && 
      this.webTransports.size === 0;
    
    if (allDisconnected) {
      useMEVStore.getState().setConnectionStatus(false);
    }
  }

  private scheduleReconnect(
    name: string,
    config: typeof this.services[keyof typeof this.services]
  ): void {
    // Clear existing timer
    const existingTimer = this.reconnectTimers.get(name);
    if (existingTimer) {
      clearTimeout(existingTimer);
    }

    // Schedule reconnection
    const timer = setTimeout(() => {
      console.log(`Attempting to reconnect to ${name}...`);
      this.connectService(name, config);
    }, 5000);

    this.reconnectTimers.set(name, timer);
  }

  private updateMetrics(data: any): void {
    if (data.bytes) {
      this.metrics.bytesReceived += data.bytes;
    }
  }

  // Public methods for subscribing to messages
  public subscribe(topic: string, handler: Function): () => void {
    if (!this.messageHandlers.has(topic)) {
      this.messageHandlers.set(topic, new Set());
    }
    
    this.messageHandlers.get(topic)!.add(handler);
    
    // Return unsubscribe function
    return () => {
      const handlers = this.messageHandlers.get(topic);
      if (handlers) {
        handlers.delete(handler);
      }
    };
  }

  // Send message to specific service
  public send(service: string, message: any): void {
    const socket = this.sockets.get(service);
    if (socket?.connected) {
      socket.emit('message', message);
    }

    const transport = this.webTransports.get(service);
    if (transport) {
      // Send via WebTransport
      // Implementation depends on stream setup
    }
  }

  // Get metrics
  public getMetrics(): typeof this.metrics {
    return { ...this.metrics };
  }

  // Disconnect all services
  public disconnectAll(): void {
    // Close WebSockets
    this.sockets.forEach((socket, name) => {
      socket.disconnect();
      this.sockets.delete(name);
    });

    // Close EventSources
    this.eventSources.forEach((source, name) => {
      source.close();
      this.eventSources.delete(name);
    });

    // Close WebTransports
    this.webTransports.forEach((transport, name) => {
      transport.close();
      this.webTransports.delete(name);
    });

    // Clear reconnect timers
    this.reconnectTimers.forEach(timer => clearTimeout(timer));
    this.reconnectTimers.clear();

    // Clear handlers
    this.messageHandlers.clear();
  }
}

// Export singleton instance
export const enhancedWebSocket = new EnhancedWebSocketService();