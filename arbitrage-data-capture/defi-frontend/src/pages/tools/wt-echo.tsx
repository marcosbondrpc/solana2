import { useEffect, useRef, useState, useCallback } from 'react';
import { proxy, useSnapshot } from 'valtio';

// WebTransport state using Valtio for lock-free updates
const wtState = proxy({
  connected: false,
  rtt: 0,
  minRtt: Infinity,
  maxRtt: 0,
  avgRtt: 0,
  p50Rtt: 0,
  p99Rtt: 0,
  packetsSent: 0,
  packetsReceived: 0,
  bytesTransferred: 0,
  connectionState: 'disconnected' as 'disconnected' | 'connecting' | 'connected' | 'error',
  lastError: null as string | null,
  datagramQueue: [] as number[],
  jitter: 0,
  throughput: 0
});

// Ring buffer for RTT measurements (lock-free)
class RingBuffer {
  private buffer: Float32Array;
  private writePos = 0;
  private size: number;

  constructor(size: number) {
    this.size = size;
    this.buffer = new Float32Array(size);
  }

  push(value: number): void {
    this.buffer[this.writePos % this.size] = value;
    this.writePos++;
  }

  getStats(): { p50: number; p99: number; avg: number; min: number; max: number } {
    const filled = Math.min(this.writePos, this.size);
    if (filled === 0) return { p50: 0, p99: 0, avg: 0, min: 0, max: 0 };
    
    const values = Array.from(this.buffer.slice(0, filled)).sort((a, b) => a - b);
    const p50Index = Math.floor(filled * 0.5);
    const p99Index = Math.floor(filled * 0.99);
    const sum = values.reduce((a, b) => a + b, 0);
    
    return {
      p50: values[p50Index],
      p99: values[p99Index],
      avg: sum / filled,
      min: values[0],
      max: values[filled - 1]
    };
  }
}

const rttBuffer = new RingBuffer(10000); // Store last 10k measurements

export default function WTEcho() {
  const state = useSnapshot(wtState);
  const transportRef = useRef<any>(null);
  const readerRef = useRef<any>(null);
  const writerRef = useRef<any>(null);
  const rafRef = useRef<number>(0);
  const startTimeRef = useRef<number>(0);
  const lastUpdateRef = useRef<number>(0);
  
  // High-precision timestamp using performance.now()
  const getHighResTimestamp = useCallback(() => {
    return performance.now() * 1000000; // Convert to nanoseconds
  }, []);

  // Ultra-fast RAF loop for UI updates
  const updateLoop = useCallback(() => {
    const now = performance.now();
    if (now - lastUpdateRef.current > 16) { // Cap at 60fps for UI
      const stats = rttBuffer.getStats();
      wtState.p50Rtt = stats.p50;
      wtState.p99Rtt = stats.p99;
      wtState.avgRtt = stats.avg;
      wtState.minRtt = stats.min;
      wtState.maxRtt = stats.max;
      
      // Calculate jitter (variation in RTT)
      if (wtState.datagramQueue.length > 1) {
        const diffs = [];
        for (let i = 1; i < wtState.datagramQueue.length; i++) {
          diffs.push(Math.abs(wtState.datagramQueue[i] - wtState.datagramQueue[i-1]));
        }
        wtState.jitter = diffs.reduce((a, b) => a + b, 0) / diffs.length;
      }
      
      // Calculate throughput
      const elapsed = (now - startTimeRef.current) / 1000; // seconds
      if (elapsed > 0) {
        wtState.throughput = wtState.bytesTransferred / elapsed / 1024 / 1024; // MB/s
      }
      
      lastUpdateRef.current = now;
    }
    rafRef.current = requestAnimationFrame(updateLoop);
  }, []);

  const connect = useCallback(async () => {
    try {
      wtState.connectionState = 'connecting';
      wtState.lastError = null;
      
      const url = process.env.NEXT_PUBLIC_WT_URL || 'https://localhost:4433/echo';
      
      // @ts-ignore - WebTransport API
      const transport = new WebTransport(url);
      transportRef.current = transport;
      
      transport.closed.then(() => {
        wtState.connected = false;
        wtState.connectionState = 'disconnected';
      }).catch((error: any) => {
        wtState.lastError = error.message;
        wtState.connectionState = 'error';
      });
      
      await transport.ready;
      
      wtState.connected = true;
      wtState.connectionState = 'connected';
      startTimeRef.current = performance.now();
      
      // Start the datagram reader
      const reader = transport.datagrams.readable.getReader();
      readerRef.current = reader;
      
      // High-performance async reader loop
      (async () => {
        try {
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            if (!value) continue;
            
            // Decode timestamp from server
            const view = new DataView(value.buffer, value.byteOffset, value.byteLength);
            const serverTs = Number(view.getBigUint64(0, false));
            const now = BigInt(Date.now()) * 1000000n;
            const rttNs = now - BigInt(serverTs);
            const rttMs = Number(rttNs) / 1e6;
            
            // Update state atomically
            wtState.rtt = rttMs;
            wtState.packetsReceived++;
            wtState.bytesTransferred += value.byteLength;
            
            // Store in ring buffer
            rttBuffer.push(rttMs);
            
            // Keep last 100 RTT values for jitter calculation
            wtState.datagramQueue.push(rttMs);
            if (wtState.datagramQueue.length > 100) {
              wtState.datagramQueue.shift();
            }
          }
        } catch (error) {
          console.error('Reader error:', error);
        }
      })();
      
      // Start update loop
      updateLoop();
      
    } catch (error: any) {
      wtState.lastError = error.message;
      wtState.connectionState = 'error';
      console.error('Connection error:', error);
    }
  }, [updateLoop]);

  const disconnect = useCallback(() => {
    if (transportRef.current) {
      transportRef.current.close();
      transportRef.current = null;
    }
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
    }
    wtState.connected = false;
    wtState.connectionState = 'disconnected';
  }, []);

  const sendDatagram = useCallback(async () => {
    if (!transportRef.current || !wtState.connected) return;
    
    try {
      const writer = transportRef.current.datagrams.writable.getWriter();
      
      // Create a test payload with timestamp
      const buffer = new ArrayBuffer(64);
      const view = new DataView(buffer);
      
      // Add current timestamp for RTT measurement
      const ts = BigInt(Date.now()) * 1000000n;
      view.setBigUint64(0, ts, false);
      
      // Add some test data
      for (let i = 8; i < 64; i += 4) {
        view.setFloat32(i, Math.random(), false);
      }
      
      await writer.write(new Uint8Array(buffer));
      writer.releaseLock();
      
      wtState.packetsSent++;
      wtState.bytesTransferred += buffer.byteLength;
      
    } catch (error) {
      console.error('Send error:', error);
    }
  }, []);

  const sendBurst = useCallback(async () => {
    // Send a burst of 100 datagrams for stress testing
    const promises = [];
    for (let i = 0; i < 100; i++) {
      promises.push(sendDatagram());
      await new Promise(resolve => setTimeout(resolve, 1)); // 1ms spacing
    }
    await Promise.all(promises);
  }, [sendDatagram]);

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  // Format number with precision
  const formatNumber = (num: number, decimals: number = 2): string => {
    if (!isFinite(num)) return '0';
    return num.toFixed(decimals);
  };

  return (
    <div className="wt-echo-container" style={{
      padding: '24px',
      fontFamily: 'Monaco, monospace',
      backgroundColor: '#0a0a0a',
      color: '#00ff00',
      minHeight: '100vh'
    }}>
      <h1 style={{ fontSize: '2rem', marginBottom: '20px', textShadow: '0 0 10px #00ff00' }}>
        WebTransport Echo Client - Ultra Low Latency
      </h1>
      
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
        gap: '20px',
        marginBottom: '30px'
      }}>
        {/* Connection Status */}
        <div style={{
          padding: '20px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h3>Connection Status</h3>
          <div>State: <span style={{
            color: state.connectionState === 'connected' ? '#00ff00' : 
                   state.connectionState === 'error' ? '#ff0000' : '#ffff00'
          }}>{state.connectionState}</span></div>
          {state.lastError && (
            <div style={{ color: '#ff0000' }}>Error: {state.lastError}</div>
          )}
        </div>

        {/* RTT Statistics */}
        <div style={{
          padding: '20px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h3>RTT Statistics (ms)</h3>
          <div>Current: <span style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
            {formatNumber(state.rtt, 3)}
          </span></div>
          <div>Min: {formatNumber(state.minRtt, 3)}</div>
          <div>Max: {formatNumber(state.maxRtt, 3)}</div>
          <div>Avg: {formatNumber(state.avgRtt, 3)}</div>
          <div>P50: {formatNumber(state.p50Rtt, 3)}</div>
          <div>P99: {formatNumber(state.p99Rtt, 3)}</div>
          <div>Jitter: {formatNumber(state.jitter, 3)}</div>
        </div>

        {/* Traffic Statistics */}
        <div style={{
          padding: '20px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h3>Traffic Statistics</h3>
          <div>Packets Sent: {state.packetsSent}</div>
          <div>Packets Received: {state.packetsReceived}</div>
          <div>Bytes Transferred: {(state.bytesTransferred / 1024).toFixed(2)} KB</div>
          <div>Throughput: {formatNumber(state.throughput, 3)} MB/s</div>
        </div>
      </div>

      {/* RTT Visualization */}
      <div style={{
        marginBottom: '30px',
        padding: '20px',
        border: '1px solid #00ff00',
        borderRadius: '8px',
        background: 'rgba(0, 255, 0, 0.05)'
      }}>
        <h3>RTT Visualization</h3>
        <div style={{
          height: '100px',
          background: '#000',
          borderRadius: '4px',
          padding: '10px',
          display: 'flex',
          alignItems: 'flex-end',
          gap: '2px'
        }}>
          {state.datagramQueue.slice(-50).map((rtt, i) => (
            <div
              key={i}
              style={{
                flex: 1,
                background: `hsl(${120 - Math.min(rtt * 10, 120)}, 100%, 50%)`,
                height: `${Math.min((rtt / 10) * 100, 100)}%`,
                transition: 'height 0.1s ease-out'
              }}
            />
          ))}
        </div>
      </div>

      {/* Control Buttons */}
      <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
        {!state.connected ? (
          <button
            onClick={connect}
            disabled={state.connectionState === 'connecting'}
            style={{
              padding: '12px 24px',
              fontSize: '16px',
              background: '#00ff00',
              color: '#000',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}
          >
            {state.connectionState === 'connecting' ? 'Connecting...' : 'Connect'}
          </button>
        ) : (
          <>
            <button
              onClick={disconnect}
              style={{
                padding: '12px 24px',
                fontSize: '16px',
                background: '#ff3333',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontWeight: 'bold'
              }}
            >
              Disconnect
            </button>
            <button
              onClick={sendDatagram}
              style={{
                padding: '12px 24px',
                fontSize: '16px',
                background: '#00aaff',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontWeight: 'bold'
              }}
            >
              Send Datagram
            </button>
            <button
              onClick={sendBurst}
              style={{
                padding: '12px 24px',
                fontSize: '16px',
                background: '#ff9900',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontWeight: 'bold'
              }}
            >
              Send Burst (100)
            </button>
          </>
        )}
      </div>
    </div>
  );
}