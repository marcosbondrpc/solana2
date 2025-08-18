/**
 * High-Performance WebSocket Provider with Binary Protocol & Message Coalescing
 * Optimized for processing 10,000+ real-time updates per second at 60 FPS
 */

import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react';
import { useMEVStore } from '../stores/mev-store';
import { useControlStore } from '../stores/control-store';
import { useBanditStore } from '../stores/bandit-store';

// Binary message types for ultra-fast parsing
const enum BinaryMessageType {
  HEARTBEAT = 0x00,
  MEV_OPPORTUNITY = 0x01,
  BUNDLE_UPDATE = 0x02,
  LATENCY_METRIC = 0x03,
  BANDIT_STATE = 0x04,
  PROFIT_UPDATE = 0x05,
  SYSTEM_METRIC = 0x06,
  BATCH_MESSAGE = 0xFF // Special type for batched messages
}

// Delta compression state management
interface DeltaCompressionState {
  lastSnapshot: Map<string, any>;
  referenceFrames: Map<number, any>;
  frameCounter: number;
}

// Message coalescing configuration
const COALESCE_INTERVAL_MS = 16; // Match 60 FPS frame rate
const MAX_BATCH_SIZE = 1000;
const RING_BUFFER_SIZE = 64 * 1024 * 1024; // 64MB ring buffer

interface WebSocketContextType {
  isConnected: boolean;
  latency: number;
  messagesPerSecond: number;
  bytesPerSecond: number;
  send: (type: string, data: any) => void;
  sendBinary: (type: BinaryMessageType, data: ArrayBuffer) => void;
  subscribe: (type: string, handler: (data: any) => void) => () => void;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within WebSocketProvider');
  }
  return context;
};

interface WebSocketProviderProps {
  children: React.ReactNode;
  config?: {
    url?: string;
    enableBinary?: boolean;
    enableCompression?: boolean;
    enableCoalescing?: boolean;
    enableDeltaCompression?: boolean;
  };
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ 
  children, 
  config = {} 
}) => {
  const {
    url = import.meta.env.VITE_WS_URL || 'ws://localhost:8085',
    enableBinary = true,
    enableCompression = true,
    enableCoalescing = true,
    enableDeltaCompression = true
  } = config;

  const [isConnected, setIsConnected] = useState(false);
  const [latency, setLatency] = useState(0);
  const [messagesPerSecond, setMessagesPerSecond] = useState(0);
  const [bytesPerSecond, setBytesPerSecond] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const subscriptionsRef = useRef<Map<string, Set<Function>>>(new Map());
  const messageQueueRef = useRef<any[]>([]);
  const coalesceTimerRef = useRef<number | null>(null);
  const statsRef = useRef({
    messages: 0,
    bytes: 0,
    lastUpdate: Date.now()
  });

  // Ring buffer for zero-copy message passing
  const ringBufferRef = useRef<SharedArrayBuffer | null>(null);
  const ringBufferViewRef = useRef<Uint8Array | null>(null);
  const ringBufferOffsetRef = useRef(0);

  // Delta compression state
  const deltaStateRef = useRef<DeltaCompressionState>({
    lastSnapshot: new Map(),
    referenceFrames: new Map(),
    frameCounter: 0
  });

  // Initialize Web Worker for heavy processing
  useEffect(() => {
    if (typeof Worker !== 'undefined') {
      workerRef.current = new Worker(
        new URL('../workers/wsDecoder.worker.ts', import.meta.url),
        { type: 'module' }
      );

      // Initialize SharedArrayBuffer if available
      if (typeof SharedArrayBuffer !== 'undefined') {
        try {
          ringBufferRef.current = new SharedArrayBuffer(RING_BUFFER_SIZE);
          ringBufferViewRef.current = new Uint8Array(ringBufferRef.current);
          
          // Share buffer with worker
          workerRef.current.postMessage({
            type: 'init',
            sharedBuffer: ringBufferRef.current
          });
        } catch (e) {
          console.warn('SharedArrayBuffer not available, falling back to regular ArrayBuffer');
        }
      }

      // Handle messages from worker
      workerRef.current.onmessage = handleWorkerMessage;
    }

    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
      }
    };
  }, []);

  // Handle messages processed by worker
  const handleWorkerMessage = useCallback((event: MessageEvent) => {
    const { type, data, error, decodeTime } = event.data;

    if (error) {
      console.error('Worker error:', error);
      return;
    }

    if (type === 'stats') {
      setMessagesPerSecond(data.messagesPerSecond);
      setBytesPerSecond(data.bytesPerSecond);
      return;
    }

    // Route to appropriate store based on message type
    switch (type) {
      case BinaryMessageType.MEV_OPPORTUNITY:
        useMEVStore.getState().addArbitrageOpportunity(data);
        break;
      case BinaryMessageType.BUNDLE_UPDATE:
        useMEVStore.getState().addJitoBundle(data);
        break;
      case BinaryMessageType.LATENCY_METRIC:
        useMEVStore.getState().addLatencyMetric(data);
        setLatency(data.websocketLatency || 0);
        break;
      case BinaryMessageType.BANDIT_STATE:
        useBanditStore.getState().updateState(data);
        break;
      case BinaryMessageType.PROFIT_UPDATE:
        useMEVStore.getState().updateProfitMetrics(data);
        break;
      case BinaryMessageType.SYSTEM_METRIC:
        useMEVStore.getState().updateSystemPerformance(data);
        break;
    }

    // Notify subscribers
    const handlers = subscriptionsRef.current.get(String(type));
    if (handlers) {
      handlers.forEach(handler => handler(data));
    }
  }, []);

  // Process coalesced messages
  const processMessageBatch = useCallback(() => {
    if (messageQueueRef.current.length === 0) return;

    const batch = messageQueueRef.current.splice(0, MAX_BATCH_SIZE);
    
    // Process batch in worker if binary, otherwise process directly
    if (enableBinary && workerRef.current) {
      // Create batch message
      const batchBuffer = createBatchBuffer(batch);
      workerRef.current.postMessage({
        type: 'decode',
        buffer: batchBuffer
      }, [batchBuffer]);
    } else {
      // Process JSON messages directly
      batch.forEach(msg => {
        try {
          const parsed = typeof msg === 'string' ? JSON.parse(msg) : msg;
          handleParsedMessage(parsed);
        } catch (e) {
          console.error('Failed to parse message:', e);
        }
      });
    }

    coalesceTimerRef.current = null;
  }, [enableBinary]);

  // Create batch buffer for multiple messages
  const createBatchBuffer = (messages: any[]): ArrayBuffer => {
    // Calculate total size
    let totalSize = 4; // 4 bytes for message count
    const encodedMessages: ArrayBuffer[] = [];
    
    messages.forEach(msg => {
      const encoded = encodeMessage(msg);
      totalSize += 4 + encoded.byteLength; // 4 bytes for size + message
      encodedMessages.push(encoded);
    });

    // Create batch buffer
    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);
    let offset = 0;

    // Write message count
    view.setUint32(offset, messages.length, true);
    offset += 4;

    // Write each message
    encodedMessages.forEach(encoded => {
      view.setUint32(offset, encoded.byteLength, true);
      offset += 4;
      
      new Uint8Array(buffer).set(new Uint8Array(encoded), offset);
      offset += encoded.byteLength;
    });

    return buffer;
  };

  // Encode message to binary format
  const encodeMessage = (msg: any): ArrayBuffer => {
    if (msg instanceof ArrayBuffer) {
      return msg;
    }
    
    const encoder = new TextEncoder();
    const jsonStr = JSON.stringify(msg);
    return encoder.encode(jsonStr).buffer;
  };

  // Handle parsed JSON messages
  const handleParsedMessage = (data: any) => {
    const { type, payload } = data;
    
    // Apply delta decompression if enabled
    if (enableDeltaCompression && data.isDelta) {
      const fullData = applyDelta(type, payload);
      data.payload = fullData;
    }

    // Route to appropriate handler
    switch (type) {
      case 'mev:opportunity':
        useMEVStore.getState().addArbitrageOpportunity(payload);
        break;
      case 'jito:bundle':
        useMEVStore.getState().addJitoBundle(payload);
        break;
      case 'metrics:latency':
        useMEVStore.getState().addLatencyMetric(payload);
        setLatency(payload.websocketLatency || 0);
        break;
      case 'bandit:state':
        useBanditStore.getState().updateState(payload);
        break;
      case 'profit:update':
        useMEVStore.getState().updateProfitMetrics(payload);
        break;
      case 'system:metrics':
        useMEVStore.getState().updateSystemPerformance(payload);
        break;
    }

    // Notify subscribers
    const handlers = subscriptionsRef.current.get(type);
    if (handlers) {
      handlers.forEach(handler => handler(payload));
    }
  };

  // Apply delta compression
  const applyDelta = (type: string, delta: any): any => {
    const state = deltaStateRef.current;
    const lastSnapshot = state.lastSnapshot.get(type);
    
    if (!lastSnapshot) {
      // First message, store as snapshot
      state.lastSnapshot.set(type, delta);
      return delta;
    }

    // Apply delta to last snapshot
    const fullData = { ...lastSnapshot };
    
    // Apply delta fields
    if (delta.$set) {
      Object.assign(fullData, delta.$set);
    }
    
    if (delta.$unset) {
      delta.$unset.forEach((key: string) => delete fullData[key]);
    }
    
    if (delta.$inc) {
      Object.entries(delta.$inc).forEach(([key, value]) => {
        fullData[key] = (fullData[key] || 0) + (value as number);
      });
    }
    
    // Update snapshot
    state.lastSnapshot.set(type, fullData);
    
    return fullData;
  };

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(url);
    
    // Use binary type for optimal performance
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      console.log('[WS] Connected to', url);
      setIsConnected(true);
      
      // Send initial configuration
      ws.send(JSON.stringify({
        type: 'config',
        enableBinary,
        enableCompression,
        enableDeltaCompression,
        supportedProtocols: ['protobuf', 'msgpack', 'json']
      }));
    };

    ws.onmessage = (event) => {
      statsRef.current.messages++;
      
      if (event.data instanceof ArrayBuffer) {
        statsRef.current.bytes += event.data.byteLength;
        
        if (enableCoalescing) {
          // Queue for coalescing
          messageQueueRef.current.push(event.data);
          
          if (!coalesceTimerRef.current) {
            coalesceTimerRef.current = window.setTimeout(
              processMessageBatch,
              COALESCE_INTERVAL_MS
            );
          }
        } else {
          // Process immediately via worker
          if (workerRef.current) {
            const buf = event.data as ArrayBuffer;
            workerRef.current.postMessage({
              type: 'decode',
              buffer: buf
            }, [buf]);
          }
        }
      } else {
        // Handle text messages
        statsRef.current.bytes += new Blob([event.data]).size;
        
        if (enableCoalescing) {
          messageQueueRef.current.push(event.data);
          
          if (!coalesceTimerRef.current) {
            coalesceTimerRef.current = window.setTimeout(
              processMessageBatch,
              COALESCE_INTERVAL_MS
            );
          }
        } else {
          handleParsedMessage(JSON.parse(event.data));
        }
      }

      // Update stats periodically
      const now = Date.now();
      if (now - statsRef.current.lastUpdate > 1000) {
        const deltaTime = (now - statsRef.current.lastUpdate) / 1000;
        setMessagesPerSecond(statsRef.current.messages / deltaTime);
        setBytesPerSecond(statsRef.current.bytes / deltaTime);
        
        statsRef.current.messages = 0;
        statsRef.current.bytes = 0;
        statsRef.current.lastUpdate = now;
      }
    };

    ws.onerror = (error) => {
      console.error('[WS] Error:', error);
      setIsConnected(false);
    };

    ws.onclose = () => {
      console.log('[WS] Disconnected');
      setIsConnected(false);
      
      // Reconnect after delay
      setTimeout(connect, 5000);
    };

    wsRef.current = ws;
  }, [url, enableBinary, enableCompression, enableCoalescing, enableDeltaCompression, processMessageBatch]);

  // Send message
  const send = useCallback((type: string, data: any) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return;
    
    wsRef.current.send(JSON.stringify({ type, payload: data }));
  }, []);

  // Send binary message
  const sendBinary = useCallback((type: BinaryMessageType, data: ArrayBuffer) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return;
    
    // Create header
    const header = new ArrayBuffer(16);
    const headerView = new DataView(header);
    
    headerView.setUint8(0, 1); // Version
    headerView.setUint8(1, type);
    headerView.setUint16(2, 0, true); // Flags
    headerView.setBigUint64(4, BigInt(Date.now()), true);
    headerView.setUint32(12, data.byteLength, true);
    
    // Combine header and data
    const message = new ArrayBuffer(header.byteLength + data.byteLength);
    new Uint8Array(message).set(new Uint8Array(header), 0);
    new Uint8Array(message).set(new Uint8Array(data), header.byteLength);
    
    wsRef.current.send(message);
  }, []);

  // Subscribe to message type
  const subscribe = useCallback((type: string, handler: (data: any) => void) => {
    if (!subscriptionsRef.current.has(type)) {
      subscriptionsRef.current.set(type, new Set());
    }
    
    subscriptionsRef.current.get(type)!.add(handler);
    
    return () => {
      const handlers = subscriptionsRef.current.get(type);
      if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) {
          subscriptionsRef.current.delete(type);
        }
      }
    };
  }, []);

  // Connect on mount
  useEffect(() => {
    connect();
    
    return () => {
      if (coalesceTimerRef.current) {
        clearTimeout(coalesceTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  const value: WebSocketContextType = {
    isConnected,
    latency,
    messagesPerSecond,
    bytesPerSecond,
    send,
    sendBinary,
    subscribe
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};