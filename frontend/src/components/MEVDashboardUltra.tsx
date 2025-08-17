import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { useSnapshot } from 'valtio';
import { mevStore } from '../stores/mevStore';
import { ProtobufWebTransport } from './ProtobufWebTransport';
import { GPUAcceleratedChart } from './GPUAcceleratedChart';
import { VirtualizedDataGrid } from './VirtualizedDataGrid';
import { SharedMemoryWorker } from '../lib/SharedMemoryWorker';
import { NetworkTopology3D } from './NetworkTopology3D';
import { ThompsonBanditVisualizer } from './ThompsonBanditVisualizer';
import { LatencyHeatmap } from './LatencyHeatmap';
import { ExecutionMonitor } from './ExecutionMonitor';
import { BundlePredictionPanel } from './BundlePredictionPanel';

// Performance metrics
const RENDER_BUDGET_MS = 16.67; // 60fps target
const UPDATE_BATCH_SIZE = 1000;
const CHART_UPDATE_INTERVAL = 33; // 30fps for charts

export const MEVDashboardUltra: React.FC = () => {
  const snap = useSnapshot(mevStore);
  const frameRef = useRef<number>();
  const lastUpdateRef = useRef<number>(0);
  const sharedWorkerRef = useRef<SharedMemoryWorker>();
  const transportRef = useRef<ProtobufWebTransport>();
  
  const [metrics, setMetrics] = useState({
    fps: 60,
    latency: 0,
    throughput: 0,
    bundleRate: 0,
    profitability: 0,
    opportunities: 0,
    memoryUsage: 0,
    cpuUsage: 0
  });

  // Initialize SharedArrayBuffer worker
  useEffect(() => {
    if ((window as any).sharedMemoryEnabled) {
      sharedWorkerRef.current = new SharedMemoryWorker();
      sharedWorkerRef.current.initialize().then(() => {
        console.log('SharedMemoryWorker initialized');
      });
    }
  }, []);

  // Initialize WebTransport connection
  useEffect(() => {
    transportRef.current = new ProtobufWebTransport({
      url: 'https://45.157.234.184:4433/mev-stream',
      onMessage: handleWebTransportMessage,
      onError: console.error
    });
    
    transportRef.current.connect();
    
    return () => {
      transportRef.current?.disconnect();
    };
  }, []);

  // High-performance message handler
  const handleWebTransportMessage = useCallback(async (data: ArrayBuffer) => {
    const now = performance.now();
    
    // Process in SharedArrayBuffer if available
    if (sharedWorkerRef.current) {
      const processed = await sharedWorkerRef.current.processData(data);
      mevStore.addOpportunities(processed.opportunities);
      mevStore.updateMetrics(processed.metrics);
    } else {
      // Fallback to main thread processing
      const decoder = new TextDecoder();
      const message = JSON.parse(decoder.decode(data));
      mevStore.addOpportunities([message]);
    }
    
    // Update latency metric
    const processingTime = performance.now() - now;
    setMetrics(prev => ({
      ...prev,
      latency: processingTime
    }));
  }, []);

  // High-performance render loop
  const renderLoop = useCallback(() => {
    const now = performance.now();
    const deltaTime = now - lastUpdateRef.current;
    
    if (deltaTime >= CHART_UPDATE_INTERVAL) {
      // Update FPS counter
      const fps = Math.round(1000 / deltaTime);
      setMetrics(prev => ({ ...prev, fps }));
      
      // Update memory usage
      if ('memory' in performance) {
        const memInfo = (performance as any).memory;
        const memoryUsage = Math.round(memInfo.usedJSHeapSize / 1048576);
        setMetrics(prev => ({ ...prev, memoryUsage }));
      }
      
      lastUpdateRef.current = now;
    }
    
    frameRef.current = requestAnimationFrame(renderLoop);
  }, []);

  // Start render loop
  useEffect(() => {
    frameRef.current = requestAnimationFrame(renderLoop);
    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
    };
  }, [renderLoop]);

  // Memoized grid columns
  const gridColumns = useMemo(() => [
    { key: 'id', label: 'ID', width: 80 },
    { key: 'timestamp', label: 'Time', width: 120 },
    { key: 'type', label: 'Type', width: 100 },
    { key: 'tokenA', label: 'Token A', width: 100 },
    { key: 'tokenB', label: 'Token B', width: 100 },
    { key: 'profit', label: 'Profit', width: 120 },
    { key: 'confidence', label: 'ML Confidence', width: 110 },
    { key: 'latency', label: 'Latency (μs)', width: 100 },
    { key: 'status', label: 'Status', width: 100 },
    { key: 'dna', label: 'Decision DNA', width: 150 }
  ], []);

  return (
    <div className="mev-dashboard-ultra">
      {/* Performance Metrics Bar */}
      <div className="metrics-bar">
        <div className="metric">
          <span className="metric-label">FPS</span>
          <span className="metric-value">{metrics.fps}</span>
        </div>
        <div className="metric">
          <span className="metric-label">Latency</span>
          <span className="metric-value">{metrics.latency.toFixed(2)}ms</span>
        </div>
        <div className="metric">
          <span className="metric-label">Throughput</span>
          <span className="metric-value">{snap.metrics.throughput}/s</span>
        </div>
        <div className="metric">
          <span className="metric-label">Bundle Rate</span>
          <span className="metric-value">{snap.metrics.bundleSuccessRate}%</span>
        </div>
        <div className="metric">
          <span className="metric-label">Memory</span>
          <span className="metric-value">{metrics.memoryUsage}MB</span>
        </div>
        <div className="metric">
          <span className="metric-label">Opportunities</span>
          <span className="metric-value">{snap.opportunities.length}</span>
        </div>
      </div>

      {/* Main Dashboard Grid */}
      <div className="dashboard-grid">
        {/* Left Panel - Real-time Feed */}
        <div className="panel panel-feed">
          <h2 className="panel-title">MEV Opportunities (Real-time)</h2>
          <VirtualizedDataGrid
            data={snap.opportunities}
            columns={gridColumns}
            rowHeight={32}
            height={400}
            onRowClick={(row) => console.log('Selected:', row)}
          />
        </div>

        {/* Center Top - GPU Accelerated Chart */}
        <div className="panel panel-chart">
          <h2 className="panel-title">Profit Analytics (GPU Accelerated)</h2>
          <GPUAcceleratedChart
            data={snap.profitHistory}
            width={600}
            height={300}
            updateInterval={CHART_UPDATE_INTERVAL}
          />
        </div>

        {/* Center Bottom - 3D Network Topology */}
        <div className="panel panel-3d">
          <h2 className="panel-title">DEX Network Topology</h2>
          <NetworkTopology3D
            nodes={snap.dexNodes}
            edges={snap.dexEdges}
            width={600}
            height={300}
          />
        </div>

        {/* Right Top - Thompson Sampling Visualizer */}
        <div className="panel panel-bandit">
          <h2 className="panel-title">Thompson Sampling Multi-Armed Bandit</h2>
          <ThompsonBanditVisualizer
            bandits={snap.bandits}
            width={400}
            height={250}
          />
        </div>

        {/* Right Middle - Latency Heatmap */}
        <div className="panel panel-heatmap">
          <h2 className="panel-title">Validator Latency Heatmap</h2>
          <LatencyHeatmap
            data={snap.latencyData}
            width={400}
            height={200}
          />
        </div>

        {/* Right Bottom - Bundle Prediction */}
        <div className="panel panel-prediction">
          <h2 className="panel-title">Bundle Success Prediction</h2>
          <BundlePredictionPanel
            predictions={snap.bundlePredictions}
            accuracy={snap.predictionAccuracy}
          />
        </div>

        {/* Bottom - Execution Monitor */}
        <div className="panel panel-execution">
          <h2 className="panel-title">Execution Monitor (Nanosecond Precision)</h2>
          <ExecutionMonitor
            executions={snap.executions}
            targetLatency={8}
          />
        </div>
      </div>

      {/* WebTransport Status */}
      <div className="transport-status">
        <span className={`status-indicator ${transportRef.current?.isConnected ? 'connected' : 'disconnected'}`} />
        <span>WebTransport: {transportRef.current?.isConnected ? 'Connected' : 'Disconnected'}</span>
        <span className="latency-display">RTT: {transportRef.current?.rtt || 0}μs</span>
      </div>
    </div>
  );
};