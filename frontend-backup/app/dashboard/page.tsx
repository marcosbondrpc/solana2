/**
 * Ultimate MEV Dashboard - Main control interface
 * Real-time monitoring and control for billions in volume
 */

'use client';

import React, { useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { motion } from 'framer-motion';
import { toast, Toaster } from 'sonner';
import { useControlStore } from '../../stores/controlStore';
import { useBanditStore } from '../../stores/banditStore';
import { useMEVStore } from '../../stores/mevStore';
import { mevWebSocket, controlWebSocket, dataWebSocket } from '../../lib/enhanced-websocket';
import { apiService } from '../../lib/api-service';

// Dynamically import heavy components
const MEVControlCenter = dynamic(
  () => import('../../components/advanced/MEVControlCenter').then(mod => ({ default: mod.MEVControlCenter })),
  { 
    loading: () => <div className="animate-pulse bg-gray-900/50 rounded-lg h-96" />,
    ssr: false 
  }
);

const MEVOpportunitiesPanel = dynamic(
  () => import('../../components/mev/MEVOpportunitiesPanel').then(mod => ({ default: mod.MEVOpportunitiesPanel })),
  { 
    loading: () => <div className="animate-pulse bg-gray-900/50 rounded-lg h-96" />,
    ssr: false 
  }
);

const BanditDashboard = dynamic(
  () => import('../../components/advanced/BanditDashboard').then(mod => ({ default: mod.BanditDashboard })),
  { 
    loading: () => <div className="animate-pulse bg-gray-900/50 rounded-lg h-96" />,
    ssr: false 
  }
);

const DecisionDNA = dynamic(
  () => import('../../components/advanced/DecisionDNA').then(mod => ({ default: mod.DecisionDNA })),
  { 
    loading: () => <div className="animate-pulse bg-gray-900/50 rounded-lg h-96" />,
    ssr: false 
  }
);

const LatencyHeatmap = dynamic(
  () => import('../../components/mev/LatencyHeatmap').then(mod => ({ default: mod.LatencyHeatmap })),
  { 
    loading: () => <div className="animate-pulse bg-gray-900/50 rounded-lg h-96" />,
    ssr: false 
  }
);

interface SystemHealth {
  status: 'healthy' | 'degraded' | 'critical';
  metrics: {
    decisionLatencyP50: number;
    decisionLatencyP99: number;
    bundleLandRate: number;
    ingestionRate: number;
    modelInference: number;
    activeOpportunities: number;
    profitRealized: number;
    gasEfficiency: number;
  };
  alerts: Array<{
    id: string;
    type: 'info' | 'warning' | 'error' | 'critical';
    message: string;
    timestamp: number;
  }>;
}

export default function DashboardPage() {
  const [activeView, setActiveView] = useState<'overview' | 'mev' | 'control' | 'analytics'>('overview');
  const [systemHealth, setSystemHealth] = useState<SystemHealth>({
    status: 'healthy',
    metrics: {
      decisionLatencyP50: 0,
      decisionLatencyP99: 0,
      bundleLandRate: 0,
      ingestionRate: 0,
      modelInference: 0,
      activeOpportunities: 0,
      profitRealized: 0,
      gasEfficiency: 0
    },
    alerts: []
  });
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [darkMode] = useState(true); // Always dark for trading

  const { systemStatus, loadCommandHistory, verifyACKChain, updateSystemStatus } = useControlStore();
  const { loadMetrics: loadBanditMetrics } = useBanditStore();

  // Initialize dashboard
  useEffect(() => {
    const initDashboard = async () => {
      try {
        // Load initial data
        await Promise.all([
          loadCommandHistory(),
          verifyACKChain(),
          updateSystemStatus(),
          loadBanditMetrics(),
          loadSystemHealth()
        ]);

        // Connect WebSockets
        connectWebSockets();

        // Start performance monitoring
        startPerformanceMonitoring();

        toast.success('Dashboard initialized successfully');
      } catch (error) {
        console.error('Dashboard initialization error:', error);
        toast.error('Failed to initialize dashboard');
      }
    };

    initDashboard();

    // Cleanup
    return () => {
      mevWebSocket.disconnect();
      controlWebSocket.disconnect();
      dataWebSocket.disconnect();
    };
  }, []);

  const loadSystemHealth = async () => {
    try {
      const [metrics, alerts, services] = await Promise.all([
        apiService.getSystemMetrics(),
        apiService.getAlerts(10),
        apiService.getServiceHealth()
      ]);

      const healthStatus = 
        metrics.decision_latency_p99_ms > 20 || metrics.bundle_land_rate < 55 ? 'critical' :
        metrics.decision_latency_p50_ms > 10 || metrics.bundle_land_rate < 65 ? 'degraded' :
        'healthy';

      setSystemHealth({
        status: healthStatus,
        metrics: {
          decisionLatencyP50: metrics.decision_latency_p50_ms || 0,
          decisionLatencyP99: metrics.decision_latency_p99_ms || 0,
          bundleLandRate: metrics.bundle_land_rate || 0,
          ingestionRate: metrics.ingestion_rate || 0,
          modelInference: 0,
          activeOpportunities: 0,
          profitRealized: 0,
          gasEfficiency: 0
        },
        alerts: alerts.map((a: any) => ({
          id: a.id || `alert-${Date.now()}`,
          type: a.type,
          message: a.message,
          timestamp: a.timestamp || Date.now()
        }))
      });
    } catch (error) {
      console.error('Failed to load system health:', error);
    }
  };

  const connectWebSockets = () => {
    // MEV WebSocket
    mevWebSocket.on('opportunity', (data) => {
      useMEVStore.getState().addOpportunity(data);
    });

    mevWebSocket.on('metrics', (data) => {
      setSystemHealth(prev => ({
        ...prev,
        metrics: { ...prev.metrics, ...data }
      }));
    });

    // Control WebSocket
    controlWebSocket.on('command_ack', (data) => {
      toast.success(`Command acknowledged: ${data.command_id}`);
    });

    controlWebSocket.on('alert', (data) => {
      setSystemHealth(prev => ({
        ...prev,
        alerts: [data, ...prev.alerts.slice(0, 9)]
      }));
      
      if (data.type === 'critical') {
        toast.error(data.message);
      } else if (data.type === 'warning') {
        toast.warning(data.message);
      }
    });

    // Data WebSocket for high-frequency updates
    dataWebSocket.on('tick', (data) => {
      // Handle high-frequency data updates
      if (data.type === 'latency') {
        useMEVStore.getState().addLatencyMetric(data.payload);
      }
    });

    // Connect all WebSockets
    mevWebSocket.connect();
    controlWebSocket.connect();
    dataWebSocket.connect();
  };

  const startPerformanceMonitoring = () => {
    // Monitor performance metrics
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.entryType === 'measure' && entry.name.startsWith('api-')) {
          console.debug(`API Performance: ${entry.name} took ${entry.duration.toFixed(2)}ms`);
        }
      }
    });
    
    observer.observe({ entryTypes: ['measure'] });

    // Track frame rate
    let lastTime = performance.now();
    let frames = 0;
    
    const trackFPS = () => {
      frames++;
      const currentTime = performance.now();
      
      if (currentTime >= lastTime + 1000) {
        const fps = Math.round((frames * 1000) / (currentTime - lastTime));
        if (fps < 30) {
          console.warn(`Low FPS detected: ${fps}`);
        }
        frames = 0;
        lastTime = currentTime;
      }
      
      requestAnimationFrame(trackFPS);
    };
    
    requestAnimationFrame(trackFPS);
  };

  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Ctrl/Cmd + K: Quick command
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        // Open command palette
      }
      
      // Ctrl/Cmd + F: Fullscreen
      if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
        e.preventDefault();
        toggleFullscreen();
      }
      
      // ESC: Emergency stop
      if (e.key === 'Escape' && e.shiftKey) {
        e.preventDefault();
        useControlStore.getState().emergencyStop();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [toggleFullscreen]);

  return (
    <div className={`min-h-screen ${darkMode ? 'bg-black text-white' : 'bg-white text-black'}`}>
      <Toaster 
        position="top-right"
        theme={darkMode ? 'dark' : 'light'}
        toastOptions={{
          style: {
            background: darkMode ? '#111' : '#fff',
            border: `1px solid ${darkMode ? '#333' : '#ddd'}`,
          },
        }}
      />

      {/* Header */}
      <motion.header
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        className="sticky top-0 z-50 backdrop-blur-xl bg-black/80 border-b border-cyan-500/20"
      >
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                MEV LEGENDARY v4.0
              </h1>
              
              {/* System Status Badge */}
              <div className={`px-3 py-1 rounded-full text-xs font-semibold ${
                systemHealth.status === 'healthy' ? 'bg-green-500/20 text-green-400' :
                systemHealth.status === 'degraded' ? 'bg-yellow-500/20 text-yellow-400' :
                'bg-red-500/20 text-red-400'
              }`}>
                {systemHealth.status.toUpperCase()}
              </div>

              {/* Key Metrics */}
              <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-1">
                  <span className="text-gray-400">P50:</span>
                  <span className={`font-mono ${
                    systemHealth.metrics.decisionLatencyP50 <= 8 ? 'text-green-400' :
                    systemHealth.metrics.decisionLatencyP50 <= 12 ? 'text-yellow-400' :
                    'text-red-400'
                  }`}>
                    {systemHealth.metrics.decisionLatencyP50.toFixed(1)}ms
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-gray-400">Land:</span>
                  <span className={`font-mono ${
                    systemHealth.metrics.bundleLandRate >= 65 ? 'text-green-400' :
                    systemHealth.metrics.bundleLandRate >= 55 ? 'text-yellow-400' :
                    'text-red-400'
                  }`}>
                    {systemHealth.metrics.bundleLandRate.toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-gray-400">Ingestion:</span>
                  <span className="font-mono text-cyan-400">
                    {(systemHealth.metrics.ingestionRate / 1000).toFixed(1)}k/s
                  </span>
                </div>
              </div>
            </div>

            {/* View Tabs */}
            <div className="flex items-center gap-2">
              {(['overview', 'mev', 'control', 'analytics'] as const).map(view => (
                <button
                  key={view}
                  onClick={() => setActiveView(view)}
                  className={`px-4 py-2 rounded capitalize transition-all ${
                    activeView === view
                      ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50'
                      : 'text-gray-400 hover:text-cyan-300 hover:bg-gray-900/50'
                  }`}
                >
                  {view}
                </button>
              ))}
              
              <button
                onClick={toggleFullscreen}
                className="ml-4 p-2 rounded text-gray-400 hover:text-cyan-300 hover:bg-gray-900/50 transition-all"
                title="Toggle Fullscreen (Ctrl+F)"
              >
                {isFullscreen ? '⊖' : '⊕'}
              </button>
            </div>
          </div>
        </div>
      </motion.header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        <motion.div
          key={activeView}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
        >
          {activeView === 'overview' && (
            <div className="grid grid-cols-12 gap-6">
              {/* MEV Opportunities - Full Width */}
              <div className="col-span-12">
                <MEVOpportunitiesPanel />
              </div>

              {/* Control Center - 2/3 Width */}
              <div className="col-span-8">
                <MEVControlCenter />
              </div>

              {/* System Alerts - 1/3 Width */}
              <div className="col-span-4">
                <div className="bg-black/90 backdrop-blur-xl rounded-lg border border-cyan-500/20 p-6">
                  <h3 className="text-lg font-semibold text-cyan-400 mb-4">System Alerts</h3>
                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {systemHealth.alerts.length === 0 ? (
                      <div className="text-gray-500 text-sm">No active alerts</div>
                    ) : (
                      systemHealth.alerts.map((alert) => (
                        <div
                          key={alert.id}
                          className={`p-3 rounded-lg border ${
                            alert.type === 'critical' ? 'bg-red-500/10 border-red-500/30' :
                            alert.type === 'error' ? 'bg-orange-500/10 border-orange-500/30' :
                            alert.type === 'warning' ? 'bg-yellow-500/10 border-yellow-500/30' :
                            'bg-blue-500/10 border-blue-500/30'
                          }`}
                        >
                          <div className="text-sm">{alert.message}</div>
                          <div className="text-xs text-gray-500 mt-1">
                            {new Date(alert.timestamp).toLocaleTimeString()}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>

              {/* Latency Heatmap - Half Width */}
              <div className="col-span-6">
                <LatencyHeatmap />
              </div>

              {/* Decision DNA - Half Width */}
              <div className="col-span-6">
                <DecisionDNA />
              </div>

              {/* Bandit Dashboard - Full Width */}
              <div className="col-span-12">
                <BanditDashboard />
              </div>
            </div>
          )}

          {activeView === 'mev' && (
            <div className="grid grid-cols-12 gap-6">
              <div className="col-span-12">
                <MEVOpportunitiesPanel />
              </div>
              <div className="col-span-6">
                <LatencyHeatmap />
              </div>
              <div className="col-span-6">
                <DecisionDNA />
              </div>
            </div>
          )}

          {activeView === 'control' && (
            <div className="grid grid-cols-12 gap-6">
              <div className="col-span-12">
                <MEVControlCenter />
              </div>
              <div className="col-span-12">
                <BanditDashboard />
              </div>
            </div>
          )}

          {activeView === 'analytics' && (
            <div className="grid grid-cols-12 gap-6">
              <div className="col-span-6">
                <LatencyHeatmap />
              </div>
              <div className="col-span-6">
                <DecisionDNA />
              </div>
              <div className="col-span-12">
                <BanditDashboard />
              </div>
            </div>
          )}
        </motion.div>
      </main>

      {/* Footer Status Bar */}
      <motion.footer
        initial={{ y: 100 }}
        animate={{ y: 0 }}
        className="fixed bottom-0 left-0 right-0 bg-black/90 backdrop-blur-xl border-t border-cyan-500/20 px-4 py-2"
      >
        <div className="container mx-auto flex items-center justify-between text-xs">
          <div className="flex items-center gap-4">
            <span className="text-gray-400">WebSocket:</span>
            <span className={`${mevWebSocket.isConnected() ? 'text-green-400' : 'text-red-400'}`}>
              {mevWebSocket.isConnected() ? 'Connected' : 'Disconnected'}
            </span>
            <span className="text-gray-400">|</span>
            <span className="text-gray-400">Messages:</span>
            <span className="text-cyan-400">{mevWebSocket.getStats().messagesReceived}</span>
            <span className="text-gray-400">|</span>
            <span className="text-gray-400">Latency:</span>
            <span className="text-cyan-400">{mevWebSocket.getStats().latency}ms</span>
          </div>
          
          <div className="flex items-center gap-4">
            <span className="text-gray-400">Active Opportunities:</span>
            <span className="text-cyan-400">{systemHealth.metrics.activeOpportunities}</span>
            <span className="text-gray-400">|</span>
            <span className="text-gray-400">Profit Today:</span>
            <span className="text-green-400">${systemHealth.metrics.profitRealized.toFixed(2)}</span>
            <span className="text-gray-400">|</span>
            <span className="text-gray-400">Gas Efficiency:</span>
            <span className="text-cyan-400">{systemHealth.metrics.gasEfficiency.toFixed(1)}%</span>
          </div>
        </div>
      </motion.footer>
    </div>
  );
}