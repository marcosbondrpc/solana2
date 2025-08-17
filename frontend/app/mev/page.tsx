'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'sonner';
import { mevStore, mevActions, startBatchFlushing, stopBatchFlushing } from '../../stores/mevStore';
import { mevWebSocket } from '../../lib/enhanced-websocket';
import { useSnapshot } from 'valtio';
import OpportunityFeed from '../../components/mev/OpportunityFeed';
import ExecutionMonitor from '../../components/mev/ExecutionMonitor';
import LatencyHeatmap from '../../components/mev/LatencyHeatmap';
import PnLDashboard from '../../components/mev/PnLDashboard';
import RiskMonitor from '../../components/mev/RiskMonitor';
import SystemStatus from '../../components/mev/SystemStatus';
import ControlPanel from '../../components/mev/ControlPanel';

// Dynamic import of new MEV Dashboard
const MEVDashboard = dynamic(() => import('../../components/MEVDashboard'), {
  ssr: false,
});

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000,
      refetchOnWindowFocus: false,
    },
  },
});

export default function MEVControlCenter() {
  const store = useSnapshot(mevStore);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [activePanel, setActivePanel] = useState<string | null>(null);
  const [showNewDashboard, setShowNewDashboard] = useState(false);

  useEffect(() => {
    // Initialize WebSocket connection
    const initializeConnection = async () => {
      try {
        await mevWebSocket.connect();
        mevActions.setConnectionStatus('ws', 'connected');
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        mevActions.setConnectionStatus('ws', 'error');
      }
    };

    initializeConnection();

    // Subscribe to WebSocket messages
    const unsubscribers = [
      mevWebSocket.subscribe('opportunity', (data) => {
        mevActions.addTransaction({
          ...data,
          timestamp: new Date().toISOString(),
        });
      }),
      mevWebSocket.subscribe('decision', (data) => {
        // Update decision metrics
        if (data.latency_ms) {
          mevStore.currentLatency = data.latency_ms;
        }
      }),
      mevWebSocket.subscribe('execution', (data) => {
        // Update execution outcomes
        if (data.success && data.actual_profit) {
          mevStore.totalProfit += data.actual_profit;
        }
      }),
      mevWebSocket.subscribe('metrics', (data) => {
        mevActions.updateSystemMetrics(data.metrics || []);
      }),
      mevWebSocket.subscribe('bundle_stats', (data) => {
        mevActions.updateBundleStats(data.stats || []);
      }),
    ];

    // Start batch flushing for performance
    startBatchFlushing();

    // Keyboard shortcuts
    const handleKeyPress = (e: KeyboardEvent) => {
      switch (e.key.toLowerCase()) {
        case ' ':
          e.preventDefault();
          mevActions.updateSettings({ autoScroll: !store.settings.autoScroll });
          break;
        case 'e':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            handleEmergencyStop();
          }
          break;
        case 'r':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            mevActions.reset();
          }
          break;
        case 'f':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            toggleFullscreen();
          }
          break;
      }
    };

    document.addEventListener('keydown', handleKeyPress);

    // Cleanup
    return () => {
      stopBatchFlushing();
      unsubscribers.forEach(unsub => unsub());
      mevWebSocket.disconnect();
      document.removeEventListener('keydown', handleKeyPress);
    };
  }, []);

  const handleEmergencyStop = async () => {
    if (confirm('Emergency stop will halt all MEV operations. Continue?')) {
      try {
        const response = await fetch('http://localhost:8000/api/control/emergency-stop', {
          method: 'POST',
        });
        if (response.ok) {
          console.log('Emergency stop activated');
        }
      } catch (error) {
        console.error('Failed to activate emergency stop:', error);
      }
    }
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const connectionColor = store.wsStatus === 'connected' ? 'text-green-500' : 
                         store.wsStatus === 'connecting' ? 'text-yellow-500' : 'text-red-500';

  // Show new integrated dashboard if toggled
  if (showNewDashboard) {
    return (
      <QueryClientProvider client={queryClient}>
        <Toaster 
          position="top-right" 
          theme="dark"
          toastOptions={{
            style: {
              background: '#1a1a1a',
              color: '#fff',
              border: '1px solid #333',
            },
          }}
        />
        <div className="relative">
          <button
            onClick={() => setShowNewDashboard(false)}
            className="absolute top-4 right-4 z-10 px-3 py-1 bg-zinc-900 border border-zinc-800 rounded text-xs hover:bg-zinc-800 transition-colors"
          >
            Switch to Classic View
          </button>
          <MEVDashboard />
        </div>
      </QueryClientProvider>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <h1 className="text-2xl font-bold text-brand-600">MEV Control Center</h1>
          <div className="flex items-center gap-2 text-xs">
            <span className={`${connectionColor} font-mono`}>
              {store.wsStatus.toUpperCase()}
            </span>
            <span className="text-zinc-500">|</span>
            <span className="font-mono">
              {store.health.ingestionRate.toLocaleString()}/s
            </span>
            <span className="text-zinc-500">|</span>
            <span className={`font-mono ${store.health.isHealthy ? 'text-green-500' : 'text-red-500'}`}>
              {store.health.isHealthy ? 'HEALTHY' : 'DEGRADED'}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowNewDashboard(true)}
            className="px-3 py-1 bg-green-900 border border-green-700 rounded text-xs hover:bg-green-800 transition-colors"
          >
            New Integrated Dashboard
          </button>
          <button
            onClick={toggleFullscreen}
            className="px-3 py-1 bg-zinc-900 border border-zinc-800 rounded text-xs hover:bg-zinc-800 transition-colors"
          >
            {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'} (Ctrl+F)
          </button>
          <button
            onClick={handleEmergencyStop}
            className="px-4 py-1 bg-red-900 border border-red-700 rounded text-xs font-bold hover:bg-red-800 transition-colors"
          >
            EMERGENCY STOP (Ctrl+E)
          </button>
        </div>
      </div>

      {/* System Status Bar */}
      <SystemStatus />

      {/* Main Grid Layout */}
      <div className="grid grid-cols-12 gap-4 mt-4">
        {/* Left Column - Opportunity Feed */}
        <div className="col-span-4">
          <div className="bg-zinc-950 border border-zinc-900 rounded-lg p-4 h-[600px]">
            <OpportunityFeed />
          </div>
        </div>

        {/* Middle Column - Execution & Latency */}
        <div className="col-span-4 space-y-4">
          <div className="bg-zinc-950 border border-zinc-900 rounded-lg p-4 h-[290px]">
            <ExecutionMonitor />
          </div>
          <div className="bg-zinc-950 border border-zinc-900 rounded-lg p-4 h-[290px]">
            <LatencyHeatmap />
          </div>
        </div>

        {/* Right Column - PnL & Risk */}
        <div className="col-span-4 space-y-4">
          <div className="bg-zinc-950 border border-zinc-900 rounded-lg p-4 h-[290px]">
            <PnLDashboard />
          </div>
          <div className="bg-zinc-950 border border-zinc-900 rounded-lg p-4 h-[290px]">
            <RiskMonitor />
          </div>
        </div>
      </div>

      {/* Control Panel */}
      <div className="mt-4">
        <ControlPanel />
      </div>

      {/* Keyboard Shortcuts Help */}
      <div className="fixed bottom-4 right-4 text-xs text-zinc-600">
        <div>Space: Pause/Resume Feed</div>
        <div>Ctrl+E: Emergency Stop</div>
        <div>Ctrl+R: Reset Dashboard</div>
        <div>Ctrl+F: Toggle Fullscreen</div>
      </div>
    </div>
  );
}