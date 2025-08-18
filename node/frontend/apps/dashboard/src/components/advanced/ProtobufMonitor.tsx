import { useEffect, useRef, useState, useCallback } from 'react';
import { proxy, useSnapshot } from 'valtio';

const protobufState = proxy({
  messages: [] as Array<{
    id: string;
    type: string;
    size: number;
    timestamp: number;
    decoded: any;
    raw: string;
    latency: number;
  }>,
  
  stats: {
    totalMessages: 0,
    totalBytes: 0,
    messagesPerSecond: 0,
    bytesPerSecond: 0,
    avgSize: 0,
    avgLatency: 0,
    typeDistribution: new Map<string, number>()
  },
  
  filters: {
    types: [] as string[],
    minSize: 0,
    maxSize: Infinity,
    searchTerm: ''
  },
  
  performance: {
    decodeTime: 0,
    renderTime: 0,
    bufferUtilization: 0
  }
});

// High-performance message buffer
class MessageBuffer {
  private buffer: any[];
  private capacity: number;
  private writePos = 0;
  
  constructor(capacity: number = 100000) {
    this.capacity = capacity;
    this.buffer = new Array(capacity);
  }
  
  push(message: any): void {
    this.buffer[this.writePos % this.capacity] = message;
    this.writePos++;
  }
  
  getRecent(count: number): any[] {
    const start = Math.max(0, this.writePos - count);
    const messages: any[] = [];
    for (let i = start; i < this.writePos && messages.length < count; i++) {
      const msg = this.buffer[i % this.capacity];
      if (msg) messages.push(msg);
    }
    return messages;
  }
  
  getUtilization(): number {
    return Math.min(this.writePos / this.capacity, 1);
  }
}

const messageBuffer = new MessageBuffer();

export default function ProtobufMonitor() {
  const state = useSnapshot(protobufState);
  const wsRef = useRef<WebSocket | null>(null);
  const statsIntervalRef = useRef<number>();
  const [paused, setPaused] = useState(false);
  const [selectedMessage, setSelectedMessage] = useState<any>(null);
  
  // Connect to protobuf stream
  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(process.env.NEXT_PUBLIC_PROTOBUF_WS || 'ws://localhost:8080/protobuf');
      wsRef.current = ws;
      
      ws.binaryType = 'arraybuffer';
      
      ws.onmessage = async (event) => {
        if (paused) return;
        
        const startTime = performance.now();
        
        try {
          // Decode protobuf message
          const buffer = new Uint8Array(event.data);
          const decoded = await decodeProtobuf(buffer);
          
          const decodeTime = performance.now() - startTime;
          
          const message = {
            id: `msg_${Date.now()}_${Math.random()}`,
            type: decoded.type || 'unknown',
            size: buffer.byteLength,
            timestamp: Date.now(),
            decoded: decoded.payload,
            raw: bufferToHex(buffer),
            latency: decoded.latency || 0
          };
          
          // Add to buffer
          messageBuffer.push(message);
          
          // Update state (throttled)
          if (protobufState.messages.length < 1000) {
            protobufState.messages.unshift(message);
            if (protobufState.messages.length > 100) {
              protobufState.messages.pop();
            }
          }
          
          // Update stats
          protobufState.stats.totalMessages++;
          protobufState.stats.totalBytes += message.size;
          
          const typeCount = protobufState.stats.typeDistribution.get(message.type) || 0;
          protobufState.stats.typeDistribution.set(message.type, typeCount + 1);
          
          protobufState.performance.decodeTime = decodeTime;
          protobufState.performance.bufferUtilization = messageBuffer.getUtilization();
          
        } catch (error) {
          console.error('Failed to decode protobuf:', error);
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
      
      ws.onclose = () => {
        // Reconnect after 2 seconds
        setTimeout(connect, 2000);
      };
    };
    
    connect();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [paused]);
  
  // Calculate stats
  useEffect(() => {
    let lastMessages = 0;
    let lastBytes = 0;
    let lastTime = Date.now();
    
    const updateStats = () => {
      const now = Date.now();
      const deltaTime = (now - lastTime) / 1000;
      
      const deltaMessages = protobufState.stats.totalMessages - lastMessages;
      const deltaBytes = protobufState.stats.totalBytes - lastBytes;
      
      protobufState.stats.messagesPerSecond = deltaMessages / deltaTime;
      protobufState.stats.bytesPerSecond = deltaBytes / deltaTime;
      protobufState.stats.avgSize = protobufState.stats.totalMessages > 0 ?
        protobufState.stats.totalBytes / protobufState.stats.totalMessages : 0;
      
      // Calculate average latency from recent messages
      const recentMessages = messageBuffer.getRecent(100);
      if (recentMessages.length > 0) {
        const totalLatency = recentMessages.reduce((sum, msg) => sum + msg.latency, 0);
        protobufState.stats.avgLatency = totalLatency / recentMessages.length;
      }
      
      lastMessages = protobufState.stats.totalMessages;
      lastBytes = protobufState.stats.totalBytes;
      lastTime = now;
    };
    
    statsIntervalRef.current = window.setInterval(updateStats, 1000);
    
    return () => {
      if (statsIntervalRef.current) {
        clearInterval(statsIntervalRef.current);
      }
    };
  }, []);
  
  // Decode protobuf (simplified - use actual protobuf library in production)
  async function decodeProtobuf(buffer: Uint8Array): Promise<any> {
    // This is a simplified decoder - use protobufjs in production
    const view = new DataView(buffer.buffer);
    
    // Read message type (first byte)
    const type = view.getUint8(0);
    const typeMap = ['MEV', 'ARB', 'OUTCOME', 'METRICS', 'TRAINING', 'DATASET', 'BANDIT', 'DNA'];
    
    return {
      type: typeMap[type] || 'UNKNOWN',
      payload: {
        // Simplified payload extraction
        timestamp: Date.now(),
        size: buffer.length,
        data: Array.from(buffer.slice(1, Math.min(100, buffer.length)))
      },
      latency: Math.random() * 10 // Simulated latency
    };
  }
  
  function bufferToHex(buffer: Uint8Array): string {
    return Array.from(buffer.slice(0, 32))
      .map(b => b.toString(16).padStart(2, '0'))
      .join(' ');
  }
  
  const exportMessages = useCallback(() => {
    const data = {
      timestamp: Date.now(),
      messages: messageBuffer.getRecent(10000),
      stats: {
        totalMessages: protobufState.stats.totalMessages,
        totalBytes: protobufState.stats.totalBytes,
        avgSize: protobufState.stats.avgSize,
        avgLatency: protobufState.stats.avgLatency,
        typeDistribution: Object.fromEntries(protobufState.stats.typeDistribution)
      }
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `protobuf_capture_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, []);
  
  const clearMessages = useCallback(() => {
    protobufState.messages = [];
    protobufState.stats.totalMessages = 0;
    protobufState.stats.totalBytes = 0;
    protobufState.stats.typeDistribution.clear();
  }, []);
  
  const formatBytes = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
    return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
  };
  
  return (
    <div style={{
      padding: '24px',
      backgroundColor: '#0a0a0a',
      color: '#00ff00',
      fontFamily: 'Monaco, monospace',
      minHeight: '100vh'
    }}>
      <h1 style={{ fontSize: '2.5rem', marginBottom: '30px', textShadow: '0 0 15px #00ff00' }}>
        Protobuf Ingestion Monitor
      </h1>
      
      {/* Stats Dashboard */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
        gap: '15px',
        marginBottom: '30px'
      }}>
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h4 style={{ fontSize: '12px', marginBottom: '10px' }}>Messages/sec</h4>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
            {state.stats.messagesPerSecond.toFixed(0)}
          </div>
        </div>
        
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h4 style={{ fontSize: '12px', marginBottom: '10px' }}>Throughput</h4>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
            {formatBytes(state.stats.bytesPerSecond)}/s
          </div>
        </div>
        
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h4 style={{ fontSize: '12px', marginBottom: '10px' }}>Total Messages</h4>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
            {state.stats.totalMessages.toLocaleString()}
          </div>
        </div>
        
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h4 style={{ fontSize: '12px', marginBottom: '10px' }}>Total Data</h4>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
            {formatBytes(state.stats.totalBytes)}
          </div>
        </div>
        
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h4 style={{ fontSize: '12px', marginBottom: '10px' }}>Avg Size</h4>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
            {formatBytes(state.stats.avgSize)}
          </div>
        </div>
        
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h4 style={{ fontSize: '12px', marginBottom: '10px' }}>Avg Latency</h4>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
            {state.stats.avgLatency.toFixed(2)}ms
          </div>
        </div>
        
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h4 style={{ fontSize: '12px', marginBottom: '10px' }}>Decode Time</h4>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
            {state.performance.decodeTime.toFixed(2)}ms
          </div>
        </div>
        
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h4 style={{ fontSize: '12px', marginBottom: '10px' }}>Buffer Usage</h4>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
            {(state.performance.bufferUtilization * 100).toFixed(0)}%
          </div>
        </div>
      </div>
      
      {/* Type Distribution */}
      <div style={{
        marginBottom: '30px',
        padding: '20px',
        border: '1px solid #00ff00',
        borderRadius: '8px',
        background: 'rgba(0, 255, 0, 0.05)'
      }}>
        <h3>Message Type Distribution</h3>
        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', marginTop: '15px' }}>
          {Array.from(state.stats.typeDistribution.entries()).map(([type, count]) => (
            <div key={type} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{
                width: '12px',
                height: '12px',
                borderRadius: '2px',
                background: type === 'MEV' ? '#00ff00' :
                           type === 'ARB' ? '#00aaff' :
                           type === 'BANDIT' ? '#ff9900' :
                           type === 'DNA' ? '#ff00ff' : '#666'
              }} />
              <span>{type}: {count}</span>
              <span style={{ opacity: 0.5 }}>
                ({((count / state.stats.totalMessages) * 100).toFixed(1)}%)
              </span>
            </div>
          ))}
        </div>
      </div>
      
      {/* Controls */}
      <div style={{
        marginBottom: '30px',
        padding: '20px',
        border: '1px solid #00ff00',
        borderRadius: '8px',
        background: 'rgba(0, 255, 0, 0.05)',
        display: 'flex',
        gap: '20px',
        alignItems: 'center'
      }}>
        <button
          onClick={() => setPaused(!paused)}
          style={{
            padding: '10px 20px',
            background: paused ? '#ff9900' : '#00ff00',
            color: paused ? '#fff' : '#000',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
        >
          {paused ? 'Resume' : 'Pause'}
        </button>
        
        <button
          onClick={clearMessages}
          style={{
            padding: '10px 20px',
            background: '#ff3333',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
        >
          Clear
        </button>
        
        <button
          onClick={exportMessages}
          style={{
            padding: '10px 20px',
            background: '#00aaff',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
        >
          Export
        </button>
        
        <input
          type="text"
          placeholder="Filter messages..."
          value={state.filters.searchTerm}
          onChange={(e) => protobufState.filters.searchTerm = e.target.value}
          style={{
            flex: 1,
            padding: '10px',
            background: '#000',
            color: '#00ff00',
            border: '1px solid #00ff00',
            borderRadius: '4px'
          }}
        />
      </div>
      
      {/* Message Stream */}
      <div style={{
        padding: '20px',
        border: '1px solid #00ff00',
        borderRadius: '8px',
        background: 'rgba(0, 255, 0, 0.05)'
      }}>
        <h3>Live Message Stream</h3>
        
        <div style={{
          maxHeight: '400px',
          overflowY: 'auto',
          marginTop: '15px',
          fontFamily: 'monospace',
          fontSize: '12px'
        }}>
          {state.messages
            .filter(msg => !state.filters.searchTerm || 
                          msg.type.includes(state.filters.searchTerm) ||
                          msg.raw.includes(state.filters.searchTerm))
            .map(msg => (
              <div
                key={msg.id}
                onClick={() => setSelectedMessage(msg)}
                style={{
                  padding: '10px',
                  marginBottom: '5px',
                  background: 'rgba(0, 255, 0, 0.02)',
                  border: '1px solid #00ff0033',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(0, 255, 0, 0.1)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(0, 255, 0, 0.02)';
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                  <span style={{
                    color: msg.type === 'MEV' ? '#00ff00' :
                           msg.type === 'ARB' ? '#00aaff' :
                           msg.type === 'BANDIT' ? '#ff9900' :
                           msg.type === 'DNA' ? '#ff00ff' : '#666',
                    fontWeight: 'bold'
                  }}>
                    {msg.type}
                  </span>
                  <span>{formatBytes(msg.size)}</span>
                  <span>{msg.latency.toFixed(2)}ms</span>
                  <span>{new Date(msg.timestamp).toLocaleTimeString()}</span>
                </div>
                <div style={{ opacity: 0.7 }}>
                  {msg.raw}...
                </div>
              </div>
            ))}
        </div>
      </div>
      
      {/* Selected Message Details */}
      {selectedMessage && (
        <div style={{
          marginTop: '30px',
          padding: '20px',
          border: '2px solid #ffff00',
          borderRadius: '8px',
          background: 'rgba(255, 255, 0, 0.05)'
        }}>
          <h3>Message Details</h3>
          <pre style={{
            marginTop: '15px',
            padding: '15px',
            background: '#000',
            borderRadius: '4px',
            overflow: 'auto'
          }}>
            {JSON.stringify(selectedMessage, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}