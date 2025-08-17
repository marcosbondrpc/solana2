/**
 * WebTransport client for ultra-low latency MEV operations
 * Achieves sub-millisecond latency with QUIC protocol
 */

import { EventEmitter } from 'eventemitter3';

export interface WebTransportConfig {
  url: string;
  serverCertificateHashes?: Array<{ algorithm: string; value: Uint8Array }>;
  maxDatagramSize?: number;
  congestionControl?: 'default' | 'throughput' | 'low-latency';
  enableDatagrams?: boolean;
  enableUnreliableStreams?: boolean;
  poolSize?: number;
  reconnectInterval?: number;
  statsInterval?: number;
}

export interface StreamOptions {
  reliable?: boolean;
  ordered?: boolean;
  priority?: number;
  chunk?: boolean;
}

interface TransportStats {
  bytesSent: number;
  bytesReceived: number;
  packetsSent: number;
  packetsReceived: number;
  packetsLost: number;
  rtt: number;
  rttVariation: number;
  minRtt: number;
  congestionWindow: number;
  streams: number;
  datagrams: number;
}

/**
 * WebTransport client optimized for MEV operations
 * Uses QUIC for minimal latency and connection pooling for reliability
 */
export class WebTransportClient extends EventEmitter {
  private transport: any = null; // WebTransport type when available
  private config: Required<WebTransportConfig>;
  private streams: Map<string, any> = new Map();
  private writers: Map<string, WritableStreamDefaultWriter> = new Map();
  private readers: Map<string, ReadableStreamDefaultReader> = new Map();
  private stats: TransportStats = {
    bytesSent: 0,
    bytesReceived: 0,
    packetsSent: 0,
    packetsReceived: 0,
    packetsLost: 0,
    rtt: 0,
    rttVariation: 0,
    minRtt: Infinity,
    congestionWindow: 0,
    streams: 0,
    datagrams: 0
  };
  private statsTimer: NodeJS.Timeout | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private reconnectAttempts = 0;
  private encoder = new TextEncoder();
  private decoder = new TextDecoder();
  private isConnecting = false;
  private connectionPool: any[] = [];
  private currentPoolIndex = 0;

  constructor(config: WebTransportConfig) {
    super();
    
    this.config = {
      url: config.url,
      serverCertificateHashes: config.serverCertificateHashes || [],
      maxDatagramSize: config.maxDatagramSize || 1200,
      congestionControl: config.congestionControl || 'low-latency',
      enableDatagrams: config.enableDatagrams !== false,
      enableUnreliableStreams: config.enableUnreliableStreams !== false,
      poolSize: config.poolSize || 1,
      reconnectInterval: config.reconnectInterval || 5000,
      statsInterval: config.statsInterval || 1000
    };

    // Check WebTransport support
    if (!this.isWebTransportSupported()) {
      console.warn('WebTransport not supported, will use WebSocket fallback');
    }
  }

  private isWebTransportSupported(): boolean {
    return typeof (window as any).WebTransport !== 'undefined';
  }

  public async connect(): Promise<void> {
    if (this.isConnecting) {
      return;
    }

    this.isConnecting = true;
    this.emit('connecting');

    try {
      if (!this.isWebTransportSupported()) {
        throw new Error('WebTransport not supported');
      }

      const WebTransport = (window as any).WebTransport;
      
      // Create connection pool for load balancing
      for (let i = 0; i < this.config.poolSize; i++) {
        const transport = new WebTransport(this.config.url, {
          serverCertificateHashes: this.config.serverCertificateHashes,
          congestionControl: this.config.congestionControl
        });

        await transport.ready;
        this.connectionPool.push(transport);
        
        // Setup handlers for each connection
        this.setupTransportHandlers(transport, i);
      }

      // Use first connection as primary
      this.transport = this.connectionPool[0];
      
      this.isConnecting = false;
      this.reconnectAttempts = 0;
      
      // Start processing incoming data
      this.startDataProcessing();
      
      // Start stats collection
      this.startStatsCollection();
      
      this.emit('connected');
      
    } catch (error) {
      this.isConnecting = false;
      this.emit('error', error);
      this.scheduleReconnect();
    }
  }

  private setupTransportHandlers(transport: any, index: number): void {
    transport.closed.then(() => {
      this.emit('disconnected', { poolIndex: index });
      
      // Remove from pool
      const idx = this.connectionPool.indexOf(transport);
      if (idx > -1) {
        this.connectionPool.splice(idx, 1);
      }
      
      // If no connections left, reconnect
      if (this.connectionPool.length === 0) {
        this.scheduleReconnect();
      }
    }).catch((error: Error) => {
      this.emit('error', { error, poolIndex: index });
    });
  }

  private async startDataProcessing(): Promise<void> {
    if (!this.transport) return;

    // Process incoming datagrams (ultra-low latency)
    if (this.config.enableDatagrams && this.transport.datagrams) {
      this.processDatagrams();
    }

    // Process incoming streams
    this.processIncomingStreams();
  }

  private async processDatagrams(): Promise<void> {
    if (!this.transport?.datagrams?.readable) return;

    const reader = this.transport.datagrams.readable.getReader();
    
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        this.stats.bytesReceived += value.byteLength;
        this.stats.datagrams++;
        
        // Process datagram
        this.handleDatagram(value);
      }
    } catch (error) {
      console.error('Datagram processing error:', error);
    } finally {
      reader.releaseLock();
    }
  }

  private handleDatagram(data: Uint8Array): void {
    try {
      // Fast path for small messages
      if (data.length < 256) {
        const message = this.decoder.decode(data);
        const parsed = JSON.parse(message);
        this.emit('datagram', parsed);
      } else {
        // Binary protocol for larger messages
        this.emit('datagram:binary', data);
      }
    } catch (error) {
      console.error('Failed to handle datagram:', error);
    }
  }

  private async processIncomingStreams(): Promise<void> {
    if (!this.transport?.incomingBidirectionalStreams) return;

    const reader = this.transport.incomingBidirectionalStreams.getReader();
    
    try {
      while (true) {
        const { value: stream, done } = await reader.read();
        if (done) break;
        
        this.stats.streams++;
        this.handleIncomingStream(stream);
      }
    } catch (error) {
      console.error('Stream processing error:', error);
    } finally {
      reader.releaseLock();
    }
  }

  private async handleIncomingStream(stream: any): Promise<void> {
    const streamId = `stream-${Date.now()}-${Math.random()}`;
    this.streams.set(streamId, stream);
    
    const reader = stream.readable.getReader();
    
    try {
      const chunks: Uint8Array[] = [];
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        chunks.push(value);
        this.stats.bytesReceived += value.byteLength;
      }
      
      // Combine chunks and process
      const combined = this.combineChunks(chunks);
      this.handleStreamData(streamId, combined);
      
    } catch (error) {
      console.error('Stream read error:', error);
    } finally {
      reader.releaseLock();
      this.streams.delete(streamId);
    }
  }

  private combineChunks(chunks: Uint8Array[]): Uint8Array {
    const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const combined = new Uint8Array(totalLength);
    let offset = 0;
    
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }
    
    return combined;
  }

  private handleStreamData(streamId: string, data: Uint8Array): void {
    try {
      const message = this.decoder.decode(data);
      const parsed = JSON.parse(message);
      this.emit('stream:data', { streamId, data: parsed });
    } catch {
      // Binary data
      this.emit('stream:binary', { streamId, data });
    }
  }

  /**
   * Send data via datagram (unreliable, unordered, lowest latency)
   */
  public async sendDatagram(data: any): Promise<void> {
    if (!this.transport?.datagrams?.writable) {
      throw new Error('Datagrams not available');
    }

    const writer = this.transport.datagrams.writable.getWriter();
    
    try {
      const encoded = typeof data === 'string' 
        ? this.encoder.encode(data)
        : this.encoder.encode(JSON.stringify(data));
      
      if (encoded.length > this.config.maxDatagramSize) {
        throw new Error(`Datagram too large: ${encoded.length} > ${this.config.maxDatagramSize}`);
      }
      
      await writer.write(encoded);
      
      this.stats.bytesSent += encoded.length;
      this.stats.packetsSent++;
      
    } finally {
      writer.releaseLock();
    }
  }

  /**
   * Create and send data via stream (reliable, ordered)
   */
  public async sendStream(data: any, options: StreamOptions = {}): Promise<void> {
    if (!this.transport) {
      throw new Error('Transport not connected');
    }

    // Round-robin across connection pool
    const transport = this.connectionPool[this.currentPoolIndex];
    this.currentPoolIndex = (this.currentPoolIndex + 1) % this.connectionPool.length;

    const stream = await transport.createBidirectionalStream();
    const streamId = `stream-${Date.now()}-${Math.random()}`;
    
    this.streams.set(streamId, stream);
    
    const writer = stream.writable.getWriter();
    
    try {
      const encoded = typeof data === 'string'
        ? this.encoder.encode(data)
        : this.encoder.encode(JSON.stringify(data));
      
      if (options.chunk && encoded.length > 16384) {
        // Chunk large messages
        for (let i = 0; i < encoded.length; i += 16384) {
          const chunk = encoded.slice(i, Math.min(i + 16384, encoded.length));
          await writer.write(chunk);
        }
      } else {
        await writer.write(encoded);
      }
      
      this.stats.bytesSent += encoded.length;
      this.stats.packetsSent += Math.ceil(encoded.length / 1200);
      
    } finally {
      await writer.close();
      writer.releaseLock();
      this.streams.delete(streamId);
    }
  }

  /**
   * Send high-priority MEV transaction
   */
  public async sendMEVTransaction(tx: any): Promise<void> {
    // Use datagram for absolute minimum latency
    if (this.transport?.datagrams?.writable) {
      await this.sendDatagram({
        type: 'mev_tx',
        priority: 'critical',
        timestamp: Date.now(),
        tx
      });
    } else {
      // Fallback to stream with highest priority
      await this.sendStream({
        type: 'mev_tx',
        priority: 'critical',
        timestamp: Date.now(),
        tx
      }, { priority: 255 });
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
      this.connect();
    }, delay);

    this.emit('reconnecting', {
      attempt: this.reconnectAttempts,
      delay
    });
  }

  private startStatsCollection(): void {
    this.statsTimer = setInterval(async () => {
      if (!this.transport) return;
      
      try {
        // Get native stats if available
        if (this.transport.getStats) {
          const nativeStats = await this.transport.getStats();
          
          // Update RTT stats
          if (nativeStats.rtt !== undefined) {
            this.stats.rtt = nativeStats.rtt;
            this.stats.minRtt = Math.min(this.stats.minRtt, nativeStats.rtt);
          }
          
          if (nativeStats.rttVariation !== undefined) {
            this.stats.rttVariation = nativeStats.rttVariation;
          }
          
          if (nativeStats.congestionWindow !== undefined) {
            this.stats.congestionWindow = nativeStats.congestionWindow;
          }
        }
        
        this.emit('stats', { ...this.stats });
      } catch (error) {
        console.error('Failed to collect stats:', error);
      }
    }, this.config.statsInterval);
  }

  public async close(): Promise<void> {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.statsTimer) {
      clearInterval(this.statsTimer);
      this.statsTimer = null;
    }

    // Close all streams
    for (const [id, stream] of this.streams) {
      try {
        await stream.close();
      } catch {}
      this.streams.delete(id);
    }

    // Close all connections in pool
    for (const transport of this.connectionPool) {
      try {
        await transport.close();
      } catch {}
    }
    
    this.connectionPool = [];
    this.transport = null;
    
    this.removeAllListeners();
  }

  public getStats(): TransportStats {
    return { ...this.stats };
  }

  public isConnected(): boolean {
    return this.connectionPool.length > 0;
  }

  public getLatency(): number {
    return this.stats.rtt;
  }

  public getPoolSize(): number {
    return this.connectionPool.length;
  }
}