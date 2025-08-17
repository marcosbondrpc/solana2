import { mevActions } from '../stores/mevStore';
import { MEVTransaction } from './clickhouse';

export class MEVWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;
  private url: string;
  private token: string;
  private decoder?: Worker;
  private coalescor?: Worker;
  
  constructor(url: string, token: string) {
    this.url = url;
    this.token = token;
    this.initializeWorkers();
  }
  
  private initializeWorkers() {
    // Initialize protobuf decoder worker
    this.decoder = new Worker(
      new URL('../workers/protoDecoder.worker.ts', import.meta.url),
      { type: 'module' }
    );
    
    // Initialize coalescing worker for batching
    this.coalescor = new Worker(
      new URL('../workers/protoCoalesce.worker.ts', import.meta.url),
      { type: 'module' }
    );
    
    // Handle decoded messages
    this.decoder.onmessage = (event) => {
      if (event.data?.type === 'env') {
        this.coalescor?.postMessage({ type: 'env', env: event.data.env });
      } else if (event.data?.type === 'decoded') {
        this.handleDecodedMessage(event.data.payload);
      }
    };
    
    // Handle batched messages
    this.coalescor.onmessage = (event) => {
      if (event.data?.type === 'batch') {
        this.handleBatchedMessages(event.data.items);
      }
    };
  }
  
  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;
    
    try {
      mevActions.setConnectionStatus('ws', 'connecting');
      
      // Construct WebSocket URL with authentication token
      const wsUrl = `${this.url}?token=${encodeURIComponent(this.token)}`;
      
      this.ws = new WebSocket(wsUrl);
      this.ws.binaryType = 'arraybuffer';
      
      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onerror = this.handleError.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
    } catch (error) {
      console.error('[WebSocket] Connection failed:', error);
      mevActions.setConnectionStatus('ws', 'error');
      this.scheduleReconnect();
    }
  }
  
  private handleOpen() {
    console.log('[WebSocket] Connected');
    mevActions.setConnectionStatus('ws', 'connected');
    this.reconnectAttempts = 0;
    this.startHeartbeat();
  }
  
  private handleMessage(event: MessageEvent) {
    try {
      if (event.data instanceof ArrayBuffer) {
        // Send to decoder worker for protobuf decoding
        this.decoder?.postMessage(
          { type: 'frame', buf: event.data },
          [event.data]
        );
      } else if (typeof event.data === 'string') {
        // Handle JSON messages
        try {
          const data = JSON.parse(event.data);
          this.handleJSONMessage(data);
        } catch (err) {
          console.error('[WebSocket] Failed to parse JSON:', err);
        }
      }
    } catch (error) {
      console.error('[WebSocket] Message handling error:', error);
    }
  }
  
  private handleDecodedMessage(payload: any) {
    // Process decoded protobuf message
    if (payload.type === 'transaction') {
      const tx = this.transformTransaction(payload);
      mevActions.addTransaction(tx);
    } else if (payload.type === 'metrics') {
      mevActions.updateSystemMetrics(payload.metrics);
    } else if (payload.type === 'bundle_stats') {
      mevActions.updateBundleStats(payload.stats);
    }
  }
  
  private handleBatchedMessages(items: any[]) {
    // Process batched messages for efficient updates
    const transactions = items
      .filter(item => item.type === 'transaction')
      .map(item => this.transformTransaction(item));
    
    transactions.forEach(tx => mevActions.addTransaction(tx));
    
    // Process other message types
    const metrics = items.filter(item => item.type === 'metrics');
    const stats = items.filter(item => item.type === 'bundle_stats');
    
    if (metrics.length > 0) {
      mevActions.updateSystemMetrics(metrics.flatMap(m => m.metrics));
    }
    
    if (stats.length > 0) {
      mevActions.updateBundleStats(stats.flatMap(s => s.stats));
    }
  }
  
  private handleJSONMessage(data: any) {
    // Handle JSON format messages (fallback)
    if (data.type === 'ping') {
      this.send({ type: 'pong', timestamp: Date.now() });
    } else if (data.type === 'transaction') {
      const tx = this.transformTransaction(data.payload);
      mevActions.addTransaction(tx);
    } else if (data.type === 'system_metrics') {
      mevActions.updateSystemMetrics(data.metrics);
    } else if (data.type === 'bundle_stats') {
      mevActions.updateBundleStats(data.stats);
    }
  }
  
  private transformTransaction(data: any): MEVTransaction {
    // Transform raw data to MEVTransaction format
    return {
      timestamp: new Date(data.timestamp || Date.now()),
      block_slot: data.block_slot || 0,
      transaction_signature: data.transaction_signature || data.signature || '',
      transaction_type: data.transaction_type || 'arbitrage',
      program_id: data.program_id || '',
      input_amount: parseFloat(data.input_amount || '0'),
      output_amount: parseFloat(data.output_amount || '0'),
      profit_amount: parseFloat(data.profit_amount || '0'),
      profit_percentage: parseFloat(data.profit_percentage || '0'),
      gas_used: parseFloat(data.gas_used || '0'),
      latency_ms: parseFloat(data.latency_ms || '0'),
      bundle_landed: Boolean(data.bundle_landed),
      decision_dna: data.decision_dna || '',
      route_path: Array.isArray(data.route_path) ? data.route_path : [],
      slippage_bps: parseFloat(data.slippage_bps || '0'),
      priority_fee: parseFloat(data.priority_fee || '0'),
    };
  }
  
  private handleError(event: Event) {
    console.error('[WebSocket] Error:', event);
    mevActions.setConnectionStatus('ws', 'error');
  }
  
  private handleClose(event: CloseEvent) {
    console.log('[WebSocket] Disconnected:', event.code, event.reason);
    mevActions.setConnectionStatus('ws', 'disconnected');
    this.stopHeartbeat();
    this.scheduleReconnect();
  }
  
  private scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WebSocket] Max reconnection attempts reached');
      return;
    }
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }
    
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts),
      30000
    );
    
    console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts + 1})`);
    
    this.reconnectTimeout = setTimeout(() => {
      this.reconnectAttempts++;
      this.connect();
    }, delay);
  }
  
  private startHeartbeat() {
    this.stopHeartbeat();
    
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping', timestamp: Date.now() });
      }
    }, 30000); // Send heartbeat every 30 seconds
  }
  
  private stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }
  
  send(data: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      try {
        if (typeof data === 'object') {
          this.ws.send(JSON.stringify(data));
        } else {
          this.ws.send(data);
        }
      } catch (error) {
        console.error('[WebSocket] Send error:', error);
      }
    } else {
      console.warn('[WebSocket] Cannot send - not connected');
    }
  }
  
  disconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
    this.stopHeartbeat();
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    
    if (this.decoder) {
      this.decoder.terminate();
      this.decoder = undefined;
    }
    
    if (this.coalescor) {
      this.coalescor.terminate();
      this.coalescor = undefined;
    }
    
    mevActions.setConnectionStatus('ws', 'disconnected');
  }
  
  getReadyState(): number {
    return this.ws?.readyState || WebSocket.CLOSED;
  }
  
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// Singleton instance
let wsInstance: MEVWebSocket | null = null;

export function connectMEVWebSocket(url?: string, token?: string): MEVWebSocket {
  const wsUrl = url || process.env.NEXT_PUBLIC_WS_URL || 'ws://45.157.234.184:8001/ws';
  const authToken = token || process.env.NEXT_PUBLIC_WS_TOKEN || '';
  
  if (!wsInstance) {
    wsInstance = new MEVWebSocket(wsUrl, authToken);
  }
  
  wsInstance.connect();
  return wsInstance;
}

export function disconnectMEVWebSocket() {
  if (wsInstance) {
    wsInstance.disconnect();
    wsInstance = null;
  }
}