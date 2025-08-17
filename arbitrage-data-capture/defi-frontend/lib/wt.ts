/**
 * WebTransport Client - HTTP/3 QUIC-based transport
 * Ultra-low latency, unreliable datagram support for lossy networks
 * Fallback to WebSocket when WebTransport unavailable
 */

import { decodeBinaryFrame, encodeBinaryFrame } from './ws-proto';
import { RealtimeClient } from './ws';

// Check for WebTransport support
declare global {
  interface Window {
    WebTransport?: any;
  }
}

export interface WebTransportConfig {
  url: string;
  jwt?: string;
  enableDatagrams?: boolean;
  enableStreams?: boolean;
  congestionControl?: 'default' | 'throughput' | 'low-latency';
  maxDatagramSize?: number;
  reconnectDelay?: number;
  maxReconnectAttempts?: number;
}

export interface WTStats {
  packetsReceived: number;
  packetsDropped: number;
  bytesReceived: number;
  bytesSent: number;
  rtt: number;
  congestionWindow: number;
  datagramsReceived: number;
  datagramsDropped: number;
  streamsOpened: number;
  streamsClosed: number;
}

/**
 * WebTransport client with automatic WebSocket fallback
 */
export class WTClient {
  private transport: any = null;
  private config: Required<WebTransportConfig>;
  private stats: WTStats;
  private callbacks: Map<string, Set<(data: any) => void>> = new Map();
  private reconnectAttempts = 0;
  private isConnecting = false;
  private fallbackClient: RealtimeClient | null = null;
  private datagramReader: ReadableStreamDefaultReader | null = null;
  private datagramWriter: WritableStreamDefaultWriter | null = null;
  private bidirectionalStreams: Map<string, any> = new Map();
  private statsTimer: number | null = null;
  
  // Ring buffer for datagrams
  private datagramBuffer: ArrayBuffer[] = [];
  private datagramBufferSize = 4096;
  private datagramHead = 0;
  private datagramTail = 0;
  
  constructor(config: WebTransportConfig) {
    this.config = {
      enableDatagrams: config.enableDatagrams ?? true,
      enableStreams: config.enableStreams ?? true,
      congestionControl: config.congestionControl || 'low-latency',
      maxDatagramSize: config.maxDatagramSize || 1200, // MTU safe
      reconnectDelay: config.reconnectDelay || 1000,
      maxReconnectAttempts: config.maxReconnectAttempts || 10,
      ...config
    };
    
    this.stats = {
      packetsReceived: 0,
      packetsDropped: 0,
      bytesReceived: 0,
      bytesSent: 0,
      rtt: 0,
      congestionWindow: 0,
      datagramsReceived: 0,
      datagramsDropped: 0,
      streamsOpened: 0,
      streamsClosed: 0
    };
    
    // Check WebTransport support
    if (!this.isWebTransportSupported()) {
      console.warn('[WT] WebTransport not supported, will use WebSocket fallback');
    }
    
    this.startStatsCollection();
  }
  
  /**
   * Check if WebTransport is supported
   */
  private isWebTransportSupported(): boolean {
    return typeof window !== 'undefined' && 
           'WebTransport' in window &&
           typeof window.WebTransport === 'function';
  }
  
  /**
   * Connect to WebTransport server or fallback to WebSocket
   */
  async connect(): Promise<void> {
    if (this.transport?.ready || this.isConnecting) {
      return;
    }
    
    this.isConnecting = true;
    
    // Try WebTransport first
    if (this.isWebTransportSupported()) {
      try {
        await this.connectWebTransport();
        return;
      } catch (error) {
        console.warn('[WT] WebTransport connection failed, falling back to WebSocket:', error);
      }
    }
    
    // Fallback to WebSocket
    await this.connectWebSocketFallback();
  }
  
  /**
   * Connect via WebTransport
   */
  private async connectWebTransport(): Promise<void> {
    const url = new URL(this.config.url);
    
    // Add authentication token if provided
    if (this.config.jwt) {
      url.searchParams.set('token', this.config.jwt);
    }
    
    // Add transport hints
    url.searchParams.set('datagrams', this.config.enableDatagrams ? 'true' : 'false');
    url.searchParams.set('congestion', this.config.congestionControl);
    
    // Create WebTransport instance
    this.transport = new window.WebTransport!(url.toString());
    
    // Wait for connection
    await this.transport.ready;
    
    console.log('[WT] Connected via WebTransport');
    this.isConnecting = false;
    this.reconnectAttempts = 0;
    
    // Setup datagram handling
    if (this.config.enableDatagrams) {
      this.setupDatagrams();
    }
    
    // Setup bidirectional streams
    if (this.config.enableStreams) {
      this.setupStreams();
    }
    
    // Monitor connection
    this.transport.closed.then(() => {
      this.handleClose();
    }).catch((error: any) => {
      this.handleError(error);
    });
    
    this.emit('connected', { transport: 'webtransport' });
  }
  
  /**
   * Setup datagram handling (unreliable, unordered)
   */
  private async setupDatagrams(): Promise<void> {
    if (!this.transport.datagrams) {
      console.warn('[WT] Datagrams not supported');
      return;
    }
    
    // Get readable stream for incoming datagrams
    this.datagramReader = this.transport.datagrams.readable.getReader();
    
    // Get writable stream for outgoing datagrams
    this.datagramWriter = this.transport.datagrams.writable.getWriter();
    
    // Start reading datagrams
    this.readDatagrams();
  }
  
  /**
   * Read incoming datagrams
   */
  private async readDatagrams(): Promise<void> {
    if (!this.datagramReader) return;
    
    try {
      while (true) {
        const { value, done } = await this.datagramReader.read();
        
        if (done) {
          break;
        }
        
        this.stats.datagramsReceived++;
        this.stats.bytesReceived += value.byteLength;
        
        // Process datagram
        await this.processDatagrams(value);
      }
    } catch (error) {
      console.error('[WT] Datagram read error:', error);
    }
  }
  
  /**
   * Process received datagram
   */
  private async processDatagrams(data: ArrayBuffer): Promise<void> {
    try {
      // Decode protobuf message
      const messages = await decodeBinaryFrame(data, true);
      
      // Emit messages
      messages.forEach(msg => {
        this.emit(msg.type || 'message', msg);
      });
      
    } catch (error) {
      console.error('[WT] Datagram decode error:', error);
      this.stats.datagramsDropped++;
    }
  }
  
  /**
   * Setup bidirectional streams (reliable, ordered)
   */
  private async setupStreams(): Promise<void> {
    if (!this.transport.incomingBidirectionalStreams) {
      console.warn('[WT] Streams not supported');
      return;
    }
    
    // Accept incoming streams
    const reader = this.transport.incomingBidirectionalStreams.getReader();
    
    try {
      while (true) {
        const { value: stream, done } = await reader.read();
        
        if (done) {
          break;
        }
        
        this.stats.streamsOpened++;
        this.handleIncomingStream(stream);
      }
    } catch (error) {
      console.error('[WT] Stream accept error:', error);
    }
  }
  
  /**
   * Handle incoming bidirectional stream
   */
  private async handleIncomingStream(stream: any): Promise<void> {
    const reader = stream.readable.getReader();
    const streamId = `stream-${Date.now()}`;
    
    this.bidirectionalStreams.set(streamId, stream);
    
    try {
      while (true) {
        const { value, done } = await reader.read();
        
        if (done) {
          break;
        }
        
        // Process stream data
        await this.processStreamData(value);
      }
    } catch (error) {
      console.error('[WT] Stream read error:', error);
    } finally {
      this.bidirectionalStreams.delete(streamId);
      this.stats.streamsClosed++;
    }
  }
  
  /**
   * Process stream data (similar to datagram but reliable)
   */
  private async processStreamData(data: ArrayBuffer): Promise<void> {
    try {
      const messages = await decodeBinaryFrame(data, true);
      
      messages.forEach(msg => {
        // Mark as reliable delivery
        msg._reliable = true;
        this.emit(msg.type || 'message', msg);
      });
      
    } catch (error) {
      console.error('[WT] Stream decode error:', error);
    }
  }
  
  /**
   * Send data via datagram (unreliable)
   */
  async sendDatagram(data: any): Promise<void> {
    if (!this.datagramWriter) {
      console.warn('[WT] Datagram writer not available');
      return;
    }
    
    try {
      const encoded = encodeBinaryFrame(data, true);
      
      // Check size limit
      if (encoded.byteLength > this.config.maxDatagramSize) {
        console.warn('[WT] Datagram too large, consider using stream');
        return;
      }
      
      await this.datagramWriter.write(encoded);
      this.stats.bytesSent += encoded.byteLength;
      
    } catch (error) {
      console.error('[WT] Datagram send error:', error);
    }
  }
  
  /**
   * Send data via stream (reliable)
   */
  async sendStream(data: any): Promise<void> {
    try {
      // Create new bidirectional stream
      const stream = await this.transport.createBidirectionalStream();
      const writer = stream.writable.getWriter();
      
      // Encode and send
      const encoded = encodeBinaryFrame(data, true);
      await writer.write(encoded);
      await writer.close();
      
      this.stats.bytesSent += encoded.byteLength;
      
    } catch (error) {
      console.error('[WT] Stream send error:', error);
    }
  }
  
  /**
   * Send data (auto-select transport method)
   */
  async send(data: any, reliable: boolean = false): Promise<void> {
    if (this.fallbackClient) {
      // Use WebSocket fallback
      this.fallbackClient.send(data);
      return;
    }
    
    if (!this.transport?.ready) {
      console.warn('[WT] Not connected');
      return;
    }
    
    // Choose transport method
    if (reliable || !this.config.enableDatagrams) {
      await this.sendStream(data);
    } else {
      await this.sendDatagram(data);
    }
  }
  
  /**
   * Connect via WebSocket fallback
   */
  private async connectWebSocketFallback(): Promise<void> {
    // Convert URL to WebSocket
    const wsUrl = this.config.url.replace('https://', 'wss://');
    
    this.fallbackClient = new RealtimeClient({
      url: wsUrl,
      mode: 'proto',
      jwt: this.config.jwt,
      enableCompression: true,
      compressionLevel: 3
    });
    
    // Forward events
    this.fallbackClient.on('*', ({ type, data }) => {
      this.emit(type, data);
    });
    
    await this.fallbackClient.connect();
    
    this.isConnecting = false;
    console.log('[WT] Connected via WebSocket fallback');
    this.emit('connected', { transport: 'websocket' });
  }
  
  /**
   * Subscribe to events
   */
  on(type: string, callback: (data: any) => void): () => void {
    if (!this.callbacks.has(type)) {
      this.callbacks.set(type, new Set());
    }
    
    this.callbacks.get(type)!.add(callback);
    
    return () => {
      this.callbacks.get(type)?.delete(callback);
    };
  }
  
  /**
   * Emit event to subscribers
   */
  private emit(type: string, data: any): void {
    this.callbacks.get(type)?.forEach(cb => {
      try {
        cb(data);
      } catch (error) {
        console.error('[WT] Callback error:', error);
      }
    });
    
    // Wildcard subscribers
    this.callbacks.get('*')?.forEach(cb => {
      try {
        cb({ type, data });
      } catch (error) {
        console.error('[WT] Wildcard callback error:', error);
      }
    });
  }
  
  /**
   * Handle connection error
   */
  private handleError(error: any): void {
    console.error('[WT] Connection error:', error);
    this.emit('error', error);
    this.handleClose();
  }
  
  /**
   * Handle connection close
   */
  private handleClose(): void {
    console.log('[WT] Connection closed');
    this.isConnecting = false;
    
    // Cleanup
    this.datagramReader = null;
    this.datagramWriter = null;
    this.bidirectionalStreams.clear();
    
    this.emit('disconnected', { stats: this.getStats() });
    
    // Attempt reconnection
    if (this.reconnectAttempts < this.config.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(
        this.config.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
        30000
      );
      
      console.log(`[WT] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
      setTimeout(() => this.connect(), delay);
    }
  }
  
  /**
   * Start collecting statistics
   */
  private startStatsCollection(): void {
    this.statsTimer = window.setInterval(async () => {
      if (this.transport?.ready) {
        try {
          // Get transport stats if available
          const stats = await this.transport.getStats?.();
          if (stats) {
            this.stats.rtt = stats.rtt || 0;
            this.stats.congestionWindow = stats.congestionWindow || 0;
            this.stats.packetsReceived = stats.packetsReceived || 0;
            this.stats.packetsDropped = stats.packetsLost || 0;
          }
        } catch (error) {
          // Stats API might not be available
        }
      }
      
      this.emit('stats', this.stats);
    }, 1000);
  }
  
  /**
   * Get current statistics
   */
  getStats(): WTStats {
    return { ...this.stats };
  }
  
  /**
   * Disconnect and cleanup
   */
  async disconnect(): Promise<void> {
    if (this.statsTimer) {
      clearInterval(this.statsTimer);
      this.statsTimer = null;
    }
    
    if (this.fallbackClient) {
      this.fallbackClient.disconnect();
      this.fallbackClient = null;
    }
    
    if (this.transport) {
      await this.transport.close();
      this.transport = null;
    }
    
    this.datagramReader = null;
    this.datagramWriter = null;
    this.bidirectionalStreams.clear();
  }
}

// Export singleton for convenience
export const wtClient = new WTClient({
  url: import.meta.env.VITE_WT_URL || 'https://localhost:4433/wt',
  enableDatagrams: true,
  congestionControl: 'low-latency'
});