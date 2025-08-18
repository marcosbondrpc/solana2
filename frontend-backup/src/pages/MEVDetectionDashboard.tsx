import React, { useState, useEffect, useMemo } from 'react';
import { BehavioralSpectrum } from '../components/BehavioralSpectrum';
import { DetectionModels } from '../components/DetectionModels';
import { EntityAnalytics } from '../components/EntityAnalytics';
import { EconomicImpact } from '../components/EconomicImpact';
import { HypothesisTesting } from '../components/HypothesisTesting';
import { DecisionDNA } from '../components/DecisionDNA';
import { motion, AnimatePresence } from 'framer-motion';

// Mock WebSocket connection for real-time data
class DetectionWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private listeners: Map<string, Set<Function>> = new Map();

  connect(url: string = 'ws://localhost:8800/ws') {
    try {
      this.ws = new WebSocket(url);
      
      this.ws.onopen = () => {
        console.log('Connected to detection API');
        this.emit('connected', { timestamp: Date.now() });
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.emit('data', data);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.emit('disconnected', { timestamp: Date.now() });
        this.reconnect();
      };
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      this.reconnect();
    }
  }

  private reconnect() {
    if (this.reconnectTimeout) clearTimeout(this.reconnectTimeout);
    this.reconnectTimeout = setTimeout(() => {
      console.log('Attempting to reconnect...');
      this.connect();
    }, 5000);
  }

  on(event: string, callback: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  off(event: string, callback: Function) {
    this.listeners.get(event)?.delete(callback);
  }

  private emit(event: string, data: any) {
    this.listeners.get(event)?.forEach(callback => callback(data));
  }

  disconnect() {
    if (this.reconnectTimeout) clearTimeout(this.reconnectTimeout);
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

export const MEVDetectionDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('behavioral');
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('connecting');
  const [realtimeStats, setRealtimeStats] = useState({
    eventsPerSecond: 0,
    detectionRate: 0,
    avgLatency: 0,
    activeEntities: 0
  });
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');

  const ws = useMemo(() => new DetectionWebSocket(), []);

  useEffect(() => {
    // Connect to WebSocket
    ws.connect();

    ws.on('connected', () => setConnectionStatus('connected'));
    ws.on('disconnected', () => setConnectionStatus('disconnected'));
    ws.on('error', () => setConnectionStatus('disconnected'));
    
    ws.on('data', (data: any) => {
      // Update real-time stats
      if (data.type === 'stats') {
        setRealtimeStats(data.stats);
      }
    });

    // Simulate real-time updates
    const interval = setInterval(() => {
      setRealtimeStats({
        eventsPerSecond: Math.floor(Math.random() * 1000) + 500,
        detectionRate: Math.random() * 100,
        avgLatency: Math.random() * 20 + 5,
        activeEntities: Math.floor(Math.random() * 50) + 20
      });
    }, 2000);

    return () => {
      ws.disconnect();
      clearInterval(interval);
    };
  }, [ws]);

  // Mock data for components
  const mockEntityProfiles = useMemo(() => [
    {
      address: 'B91piBSfCBRs5rUxCMRdJEGv7tNEnFxweWcdQJHJoFpi',
      attackStyle: 'surgical' as const,
      riskAppetite: 85,
      feePosture: 'aggressive' as const,
      uptimeCadence: Array(24).fill(0).map(() => Math.random()),
      successRate: 78,
      avgLatency: 12,
      volumeSOL: 12500,
      sandwichCount: 342,
      arbitrageCount: 567,
      liquidationCount: 89,
      jitCount: 234
    },
    {
      address: '6gAnjderE13TGGFeqdPVQ438jp2FPVeyXAszxKu9y338',
      attackStyle: 'hybrid' as const,
      riskAppetite: 65,
      feePosture: 'adaptive' as const,
      uptimeCadence: Array(24).fill(0).map(() => Math.random()),
      successRate: 82,
      avgLatency: 15,
      volumeSOL: 8900,
      sandwichCount: 256,
      arbitrageCount: 789,
      liquidationCount: 45,
      jitCount: 178
    },
    {
      address: 'CaCZgxpiEDZtCgXABN9X8PTwkRk81Y9PKdc6s66frH7q',
      attackStyle: 'shotgun' as const,
      riskAppetite: 95,
      feePosture: 'aggressive' as const,
      uptimeCadence: Array(24).fill(0).map(() => Math.random()),
      successRate: 62,
      avgLatency: 8,
      volumeSOL: 25000,
      sandwichCount: 890,
      arbitrageCount: 234,
      liquidationCount: 167,
      jitCount: 456
    }
  ], []);

  const mockModels = useMemo(() => [
    {
      name: 'GNN-Alpha',
      type: 'GNN' as const,
      accuracy: 92.5,
      precision: 0.91,
      recall: 0.88,
      f1Score: 0.895,
      auc: 0.945,
      latencyP50: 12,
      latencyP95: 28,
      latencyP99: 45,
      confusionMatrix: {
        truePositive: 8234,
        falsePositive: 567,
        trueNegative: 12456,
        falseNegative: 892
      },
      rocCurve: Array(100).fill(0).map((_, i) => ({
        fpr: i / 100,
        tpr: 1 - Math.exp(-3 * (i / 100))
      })),
      predictions: []
    },
    {
      name: 'Transformer-Beta',
      type: 'Transformer' as const,
      accuracy: 89.3,
      precision: 0.87,
      recall: 0.92,
      f1Score: 0.894,
      auc: 0.932,
      latencyP50: 18,
      latencyP95: 42,
      latencyP99: 78,
      confusionMatrix: {
        truePositive: 7892,
        falsePositive: 892,
        trueNegative: 11234,
        falseNegative: 567
      },
      rocCurve: Array(100).fill(0).map((_, i) => ({
        fpr: i / 100,
        tpr: 1 - Math.exp(-2.5 * (i / 100))
      })),
      predictions: []
    },
    {
      name: 'Hybrid-Gamma',
      type: 'Hybrid' as const,
      accuracy: 94.7,
      precision: 0.93,
      recall: 0.91,
      f1Score: 0.92,
      auc: 0.961,
      latencyP50: 15,
      latencyP95: 35,
      latencyP99: 62,
      confusionMatrix: {
        truePositive: 9123,
        falsePositive: 423,
        trueNegative: 13567,
        falseNegative: 678
      },
      rocCurve: Array(100).fill(0).map((_, i) => ({
        fpr: i / 100,
        tpr: 1 - Math.exp(-3.5 * (i / 100))
      })),
      predictions: []
    }
  ], []);

  const tabs = [
    { id: 'behavioral', label: 'Behavioral Spectrum', icon: '🎯' },
    { id: 'models', label: 'Detection Models', icon: '🤖' },
    { id: 'entities', label: 'Entity Analytics', icon: '👁' },
    { id: 'economic', label: 'Economic Impact', icon: '💰' },
    { id: 'hypothesis', label: 'Hypothesis Testing', icon: '🔬' },
    { id: 'dna', label: 'Decision DNA', icon: '🧬' }
  ];

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Grid background pattern */}
      <div className="fixed inset-0 grid-pattern opacity-10 pointer-events-none" />
      
      {/* Header */}
      <header className="glass border-b border-zinc-800 sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                MEV Detection System
              </h1>
              <span className="px-3 py-1 bg-zinc-800 rounded-full text-xs font-medium text-zinc-400">
                DETECTION ONLY
              </span>
            </div>
            
            {/* Connection Status */}
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${
                  connectionStatus === 'connected' ? 'bg-green-500 animate-pulse' :
                  connectionStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' :
                  'bg-red-500'
                }`} />
                <span className="text-sm text-zinc-400">
                  {connectionStatus === 'connected' ? 'Live' :
                   connectionStatus === 'connecting' ? 'Connecting...' :
                   'Offline'}
                </span>
              </div>
              
              {/* Time Range Selector */}
              <div className="flex gap-1 bg-zinc-900 rounded-lg p-1">
                {(['1h', '24h', '7d', '30d'] as const).map(range => (
                  <button
                    key={range}
                    onClick={() => setTimeRange(range)}
                    className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                      timeRange === range
                        ? 'bg-zinc-700 text-white'
                        : 'text-zinc-500 hover:text-zinc-300'
                    }`}
                  >
                    {range}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Real-time Stats Bar */}
      <div className="glass border-b border-zinc-800">
        <div className="container mx-auto px-6 py-3">
          <div className="grid grid-cols-4 gap-6">
            <div className="flex items-center gap-3">
              <div className="text-xs text-zinc-500">Events/sec</div>
              <div className="text-lg font-semibold text-cyan-400">
                {realtimeStats.eventsPerSecond.toLocaleString()}
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="text-xs text-zinc-500">Detection Rate</div>
              <div className="text-lg font-semibold text-purple-400">
                {realtimeStats.detectionRate.toFixed(1)}%
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="text-xs text-zinc-500">Avg Latency</div>
              <div className={`text-lg font-semibold ${
                realtimeStats.avgLatency < 10 ? 'text-green-400' :
                realtimeStats.avgLatency < 20 ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                {realtimeStats.avgLatency.toFixed(1)}ms
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="text-xs text-zinc-500">Active Entities</div>
              <div className="text-lg font-semibold text-orange-400">
                {realtimeStats.activeEntities}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="glass border-b border-zinc-800">
        <div className="container mx-auto px-6">
          <div className="flex gap-1 py-2">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 ${
                  activeTab === tab.id
                    ? 'bg-gradient-to-r from-cyan-600/20 to-purple-600/20 text-white border border-cyan-500/30'
                    : 'text-zinc-400 hover:text-white hover:bg-zinc-800'
                }`}
              >
                <span>{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {activeTab === 'behavioral' && (
              <BehavioralSpectrum
                profiles={mockEntityProfiles}
                selectedAddress={selectedEntity}
                onAddressSelect={setSelectedEntity}
              />
            )}
            
            {activeTab === 'models' && (
              <DetectionModels
                models={mockModels}
              />
            )}
            
            {activeTab === 'entities' && (
              <EntityAnalytics
                onEntitySelect={setSelectedEntity}
              />
            )}
            
            {activeTab === 'economic' && (
              <EconomicImpact />
            )}
            
            {activeTab === 'hypothesis' && (
              <HypothesisTesting />
            )}
            
            {activeTab === 'dna' && (
              <DecisionDNA />
            )}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="glass border-t border-zinc-800 mt-12">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between text-xs text-zinc-500">
            <div>MEV Detection System v1.0 - Read-Only Analytics</div>
            <div>© 2025 Solana MEV Infrastructure</div>
          </div>
        </div>
      </footer>
    </div>
  );
};