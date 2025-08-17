'use client';

import React, { useEffect, useState, useRef } from 'react';
import { ArbitrageScanner } from '@/components/mev/ArbitrageScanner';
import { JitoBundleTracker } from '@/components/mev/JitoBundleTracker';
import { LatencyHeatmap } from '@/components/mev/LatencyHeatmap';
import { ProfitDashboard } from '@/components/mev/ProfitDashboard';
import { initializeMEVWebSocket } from '@/services/mev-websocket';
import { useMEVStore } from '@/stores/mev-store';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Activity, 
  DollarSign, 
  Zap, 
  Package, 
  BarChart3,
  Settings,
  Maximize2,
  Minimize2,
  RefreshCw,
  Download,
  Upload,
  Wifi,
  WifiOff,
  AlertTriangle,
  CheckCircle,
  XCircle,
  ChevronRight
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useHotkeys } from 'react-hotkeys-hook';

// Performance monitoring hook
const usePerformanceMonitor = () => {
  const [fps, setFps] = useState(60);
  const frameCount = useRef(0);
  const lastTime = useRef(performance.now());
  
  useEffect(() => {
    let animationId: number;
    
    const measureFPS = () => {
      frameCount.current++;
      const currentTime = performance.now();
      
      if (currentTime >= lastTime.current + 1000) {
        setFps(Math.round((frameCount.current * 1000) / (currentTime - lastTime.current)));
        frameCount.current = 0;
        lastTime.current = currentTime;
      }
      
      animationId = requestAnimationFrame(measureFPS);
    };
    
    animationId = requestAnimationFrame(measureFPS);
    
    return () => cancelAnimationFrame(animationId);
  }, []);
  
  return fps;
};

export default function MEVDashboard() {
  const { 
    isConnected, 
    connectionLatency, 
    nodeHealth,
    totalOpportunities,
    successfulTrades,
    profitMetrics,
    alerts,
    clearOldData,
    reset
  } = useMEVStore();
  
  const [activeTab, setActiveTab] = useState('arbitrage');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const wsRef = useRef<ReturnType<typeof initializeMEVWebSocket> | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const fps = usePerformanceMonitor();
  
  // Initialize WebSocket and Worker
  useEffect(() => {
    // Initialize WebSocket connection
    wsRef.current = initializeMEVWebSocket(
      process.env.NEXT_PUBLIC_MEV_WS_URL || 'ws://localhost:8080/mev'
    );
    
    // Initialize Web Worker
    if (typeof Worker !== 'undefined') {
      workerRef.current = new Worker('/workers/mev-processor.worker.js');
      
      workerRef.current.onmessage = (e) => {
        console.log('Worker result:', e.data);
      };
    }
    
    // Cleanup function
    return () => {
      wsRef.current?.destroy();
      workerRef.current?.terminate();
    };
  }, []);
  
  // Auto-refresh data
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      // Clear old data (older than 1 hour)
      clearOldData(Date.now() - 3600000);
    }, 60000); // Every minute
    
    return () => clearInterval(interval);
  }, [autoRefresh, clearOldData]);
  
  // Keyboard shortcuts
  useHotkeys('ctrl+1', () => setActiveTab('arbitrage'));
  useHotkeys('ctrl+2', () => setActiveTab('bundles'));
  useHotkeys('ctrl+3', () => setActiveTab('latency'));
  useHotkeys('ctrl+4', () => setActiveTab('profit'));
  useHotkeys('ctrl+r', () => window.location.reload());
  useHotkeys('ctrl+f', () => setIsFullscreen(!isFullscreen));
  useHotkeys('ctrl+s', () => setShowSettings(!showSettings));
  
  // Handle fullscreen
  useEffect(() => {
    if (isFullscreen) {
      document.documentElement.requestFullscreen?.();
    } else {
      if (document.fullscreenElement) {
        document.exitFullscreen?.();
      }
    }
  }, [isFullscreen]);
  
  // Export data function
  const exportData = () => {
    const data = {
      timestamp: Date.now(),
      opportunities: totalOpportunities,
      successfulTrades,
      profitMetrics,
      alerts: alerts.slice(0, 100)
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mev-data-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  
  const connectionStatus = {
    color: isConnected ? 'text-green-500' : 'text-red-500',
    icon: isConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />,
    text: isConnected ? 'Connected' : 'Disconnected'
  };
  
  const healthStatus = {
    healthy: { color: 'text-green-500', icon: <CheckCircle className="w-4 h-4" /> },
    degraded: { color: 'text-yellow-500', icon: <AlertTriangle className="w-4 h-4" /> },
    unhealthy: { color: 'text-red-500', icon: <XCircle className="w-4 h-4" /> }
  }[nodeHealth];
  
  return (
    <div className="min-h-screen bg-background p-4">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-6"
      >
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h1 className="text-2xl font-bold flex items-center gap-2">
                <Zap className="w-8 h-8 text-yellow-500" />
                MEV Dashboard
              </h1>
              
              <div className="flex items-center gap-3">
                <Badge variant="outline" className={connectionStatus.color}>
                  {connectionStatus.icon}
                  <span className="ml-1">{connectionStatus.text}</span>
                  {isConnected && (
                    <span className="ml-1 text-xs">({connectionLatency.toFixed(0)}ms)</span>
                  )}
                </Badge>
                
                <Badge variant="outline" className={healthStatus.color}>
                  {healthStatus.icon}
                  <span className="ml-1">Node {nodeHealth}</span>
                </Badge>
                
                <Badge variant="outline" className={fps < 30 ? 'text-red-500' : 'text-green-500'}>
                  <Activity className="w-3 h-3" />
                  <span className="ml-1">{fps} FPS</span>
                </Badge>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={autoRefresh ? 'text-green-500' : ''}
              >
                <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
              </Button>
              
              <Button
                size="sm"
                variant="outline"
                onClick={exportData}
              >
                <Download className="w-4 h-4" />
              </Button>
              
              <Button
                size="sm"
                variant="outline"
                onClick={() => setShowSettings(!showSettings)}
              >
                <Settings className="w-4 h-4" />
              </Button>
              
              <Button
                size="sm"
                variant="outline"
                onClick={() => setIsFullscreen(!isFullscreen)}
              >
                {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
              </Button>
            </div>
          </div>
          
          {/* Quick Stats */}
          <div className="grid grid-cols-6 gap-4 mt-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-500">{totalOpportunities}</div>
              <div className="text-xs text-gray-500">Opportunities</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-500">{successfulTrades}</div>
              <div className="text-xs text-gray-500">Successful</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-500">
                ${profitMetrics.totalProfit.toFixed(2)}
              </div>
              <div className="text-xs text-gray-500">Total Profit</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-500">
                ${profitMetrics.hourlyProfit.toFixed(2)}
              </div>
              <div className="text-xs text-gray-500">Hourly Profit</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-cyan-500">
                {(profitMetrics.successRate * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-gray-500">Success Rate</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-500">
                {alerts.filter(a => !a.acknowledged).length}
              </div>
              <div className="text-xs text-gray-500">Active Alerts</div>
            </div>
          </div>
        </Card>
      </motion.div>
      
      {/* Main Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid grid-cols-4 w-full">
          <TabsTrigger value="arbitrage" className="flex items-center gap-2">
            <Activity className="w-4 h-4" />
            Arbitrage Scanner
          </TabsTrigger>
          <TabsTrigger value="bundles" className="flex items-center gap-2">
            <Package className="w-4 h-4" />
            Jito Bundles
          </TabsTrigger>
          <TabsTrigger value="latency" className="flex items-center gap-2">
            <Zap className="w-4 h-4" />
            Latency Monitor
          </TabsTrigger>
          <TabsTrigger value="profit" className="flex items-center gap-2">
            <DollarSign className="w-4 h-4" />
            Profit & Loss
          </TabsTrigger>
        </TabsList>
        
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.2 }}
          >
            <TabsContent value="arbitrage" className="mt-0">
              <ArbitrageScanner />
            </TabsContent>
            
            <TabsContent value="bundles" className="mt-0">
              <JitoBundleTracker />
            </TabsContent>
            
            <TabsContent value="latency" className="mt-0">
              <LatencyHeatmap />
            </TabsContent>
            
            <TabsContent value="profit" className="mt-0">
              <ProfitDashboard />
            </TabsContent>
          </motion.div>
        </AnimatePresence>
      </Tabs>
      
      {/* Settings Panel */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0, x: 300 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 300 }}
            className="fixed right-0 top-0 h-full w-80 bg-background border-l shadow-lg p-4 z-50"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">Settings</h2>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setShowSettings(false)}
              >
                <XCircle className="w-4 h-4" />
              </Button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium">Auto Execute</label>
                <select className="w-full mt-1 px-2 py-1 border rounded">
                  <option value="off">Off</option>
                  <option value="conservative">Conservative</option>
                  <option value="aggressive">Aggressive</option>
                </select>
              </div>
              
              <div>
                <label className="text-sm font-medium">Min Profit Threshold</label>
                <input
                  type="number"
                  className="w-full mt-1 px-2 py-1 border rounded"
                  defaultValue="0.01"
                  step="0.001"
                />
              </div>
              
              <div>
                <label className="text-sm font-medium">Max Slippage</label>
                <input
                  type="number"
                  className="w-full mt-1 px-2 py-1 border rounded"
                  defaultValue="2"
                  step="0.1"
                />
              </div>
              
              <div>
                <label className="text-sm font-medium">Update Frequency</label>
                <select className="w-full mt-1 px-2 py-1 border rounded">
                  <option value="realtime">Real-time</option>
                  <option value="1s">1 second</option>
                  <option value="5s">5 seconds</option>
                  <option value="10s">10 seconds</option>
                </select>
              </div>
              
              <Button 
                className="w-full"
                onClick={() => reset()}
              >
                Reset All Data
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Keyboard Shortcuts Help */}
      <div className="fixed bottom-4 right-4 text-xs text-gray-500">
        <div className="flex items-center gap-2">
          <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded">Ctrl</kbd>
          <span>+</span>
          <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded">1-4</kbd>
          <span>Switch tabs</span>
        </div>
      </div>
    </div>
  );
}