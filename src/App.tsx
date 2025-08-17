import React, { useEffect, useState, useMemo } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster, toast } from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';
import EntityBehavioralSpectrum from './components/EntityBehavioralSpectrum';
import DetectionStream from './components/DetectionStream';
import ModelPerformance from './components/ModelPerformance';
import DecisionDNAExplorer from './components/DecisionDNAExplorer';
import { 
  getWebSocketManager, 
  DetectionEvent, 
  EntityProfile, 
  ModelMetrics 
} from './services/websocket';
import { theme } from './theme';
import './App.css';

const queryClient = new QueryClient();

// Key entities to monitor
const FOCUS_ENTITIES = [
  'B91piBSfCBRs5rUxCMRdJEGv7tNEnFxweWcdQJHJoFpi',
  '6gAnjderE13TGGFeqdPVQ438jp2FPVeyXAszxKu9y338',
  'E6YoRP3adE5XYneSseLee15wJshDxCsmyD2WtLvAmfLi',
  'CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C', // Raydium
  'pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA', // PumpSwap
];

function App() {
  const [detectionEvents, setDetectionEvents] = useState<DetectionEvent[]>([]);
  const [entities, setEntities] = useState<EntityProfile[]>([]);
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics[]>([]);
  const [dnaNodes, setDnaNodes] = useState<any[]>([]);
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'entities' | 'detection' | 'models' | 'dna'>('overview');
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');

  // Initialize WebSocket connection
  useEffect(() => {
    const wsManager = getWebSocketManager();
    
    const init = async () => {
      try {
        await wsManager.initialize();
        
        // Subscribe to channels
        wsManager.subscribe('detection');
        wsManager.subscribe('entities');
        wsManager.subscribe('models');
        wsManager.subscribe('dna');
        
        // Set up event listeners
        wsManager.on('connection', (data) => {
          setConnectionStatus(data.status);
          if (data.status === 'connected') {
            toast.success('Connected to MEV Detection System');
          } else if (data.status === 'disconnected') {
            toast.error('Disconnected from MEV Detection System');
          }
        });
        
        wsManager.on('detectionBatch', (events: DetectionEvent[]) => {
          setDetectionEvents(prev => [...events, ...prev].slice(0, 1000));
        });
        
        wsManager.on('entity', (entity: EntityProfile) => {
          setEntities(prev => {
            const existing = prev.findIndex(e => e.address === entity.address);
            if (existing >= 0) {
              const updated = [...prev];
              updated[existing] = entity;
              return updated;
            }
            return [entity, ...prev];
          });
        });
        
        wsManager.on('metrics', (metrics: ModelMetrics) => {
          setModelMetrics(prev => {
            const existing = prev.findIndex(m => m.layerId === metrics.layerId);
            if (existing >= 0) {
              const updated = [...prev];
              updated[existing] = metrics;
              return updated;
            }
            return [...prev, metrics];
          });
        });
        
        wsManager.on('error', (error) => {
          console.error('WebSocket error:', error);
          toast.error('Connection error occurred');
        });
        
      } catch (error) {
        console.error('Failed to initialize WebSocket:', error);
        toast.error('Failed to connect to detection system');
      }
    };
    
    init();
    
    // Generate mock data for demonstration
    generateMockData();
    
    return () => {
      wsManager.disconnect();
    };
  }, []);

  // Generate mock data for demonstration
  const generateMockData = () => {
    // Mock entities
    const mockEntities: EntityProfile[] = FOCUS_ENTITIES.map((address, idx) => ({
      address,
      style: ['SURGICAL', 'SHOTGUN', 'HYBRID'][idx % 3] as any,
      riskAppetite: 0.3 + Math.random() * 0.7,
      feePosture: ['AGGRESSIVE', 'CONSERVATIVE', 'ADAPTIVE'][idx % 3] as any,
      uptime: {
        hours: Math.floor(Math.random() * 24),
        pattern: ['CONSISTENT', 'SPORADIC', 'SCHEDULED'][idx % 3] as any,
      },
      attackVolume: Math.floor(Math.random() * 10000),
      successRate: 60 + Math.random() * 35,
      avgProfit: 1000 + Math.random() * 9000,
      preferredVenues: idx < 3 ? ['Raydium', 'Orca'] : ['PumpSwap', 'Jupiter'],
      knownAssociates: [],
      behavioral_embedding: [Math.random() * 10 - 5, Math.random() * 10 - 5],
    }));
    setEntities(mockEntities);

    // Mock detection events
    const mockEvents: DetectionEvent[] = Array.from({ length: 50 }, (_, i) => ({
      id: `event-${i}`,
      timestamp: Date.now() - i * 60000,
      type: ['SANDWICH', 'FRONTRUN', 'BACKRUN', 'ARBITRAGE', 'LIQUIDATION'][i % 5] as any,
      severity: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'][Math.floor(Math.random() * 4)] as any,
      actors: {
        attacker: FOCUS_ENTITIES[i % FOCUS_ENTITIES.length],
        victim: i % 2 === 0 ? `victim-${i}` : undefined,
      },
      metrics: {
        profitEstimate: 100 + Math.random() * 9900,
        gasUsed: 100000 + Math.random() * 400000,
        latency: 5 + Math.random() * 15,
        confidence: 0.7 + Math.random() * 0.3,
      },
      signatures: {
        ed25519: `sig-${Math.random().toString(36).substring(7)}`,
        merkleRoot: `merkle-${Math.random().toString(36).substring(7)}`,
      },
      venue: ['Raydium', 'PumpSwap', 'Orca', 'Jupiter'][i % 4],
      txHash: `0x${Math.random().toString(16).substring(2, 66)}`,
      blockHeight: 250000000 + i * 10,
    }));
    setDetectionEvents(mockEvents);

    // Mock model metrics
    const mockMetrics: ModelMetrics[] = [
      'sandwich-detector',
      'arbitrage-classifier',
      'liquidation-predictor',
      'ensemble-voter',
    ].map(layerId => ({
      layerId,
      accuracy: 0.9 + Math.random() * 0.09,
      precision: 0.85 + Math.random() * 0.14,
      recall: 0.88 + Math.random() * 0.11,
      f1Score: 0.86 + Math.random() * 0.13,
      rocAuc: 0.92 + Math.random() * 0.07,
      latencyP50: 5 + Math.random() * 5,
      latencyP95: 10 + Math.random() * 8,
      latencyP99: 15 + Math.random() * 10,
      confusionMatrix: {
        tp: Math.floor(800 + Math.random() * 200),
        fp: Math.floor(20 + Math.random() * 30),
        tn: Math.floor(900 + Math.random() * 100),
        fn: Math.floor(15 + Math.random() * 25),
      },
    }));
    setModelMetrics(mockMetrics);

    // Mock DNA nodes
    const mockDnaNodes = Array.from({ length: 20 }, (_, i) => ({
      id: `node-${i}`,
      hash: `0x${Math.random().toString(16).substring(2, 66)}`,
      timestamp: Date.now() - i * 300000,
      features: Array.from({ length: 5 }, () => `feature-${Math.random().toString(36).substring(7)}`),
      signature: `ed25519-${Math.random().toString(36).substring(7)}`,
      parent: i > 0 ? `node-${Math.max(0, i - Math.ceil(Math.random() * 3))}` : undefined,
      children: [],
      depth: Math.floor(i / 5),
      confidence: 0.7 + Math.random() * 0.3,
    }));
    setDnaNodes(mockDnaNodes);
  };

  const stats = useMemo(() => ({
    totalDetections: detectionEvents.length,
    criticalAlerts: detectionEvents.filter(e => e.severity === 'CRITICAL').length,
    avgLatency: modelMetrics.reduce((acc, m) => acc + m.latencyP50, 0) / Math.max(modelMetrics.length, 1),
    activeEntities: entities.length,
  }), [detectionEvents, modelMetrics, entities]);

  return (
    <QueryClientProvider client={queryClient}>
      <div style={{
        minHeight: '100vh',
        background: `linear-gradient(180deg, ${theme.colors.bg.primary} 0%, ${theme.colors.bg.secondary} 100%)`,
        color: theme.colors.text.primary,
      }}>
        <Toaster
          position="top-right"
          toastOptions={{
            style: {
              background: theme.colors.bg.tertiary,
              color: theme.colors.text.primary,
              border: `1px solid ${theme.colors.border.primary}`,
            },
          }}
        />

        {/* Header */}
        <motion.header
          initial={{ y: -100 }}
          animate={{ y: 0 }}
          style={{
            background: theme.colors.bg.glass,
            backdropFilter: 'blur(20px)',
            borderBottom: `1px solid ${theme.colors.border.glass}`,
            padding: theme.spacing.lg,
          }}
        >
          <div style={{ 
            maxWidth: '1600px', 
            margin: '0 auto',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: theme.spacing.lg }}>
              <h1 style={{ 
                fontSize: theme.fontSize['3xl'],
                fontWeight: 'bold',
                background: theme.colors.gradients.primary,
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                textShadow: theme.effects.neon.text,
              }}>
                MEV Detection System
              </h1>
              <span style={{
                padding: `${theme.spacing.xs} ${theme.spacing.sm}`,
                background: theme.colors.bg.tertiary,
                borderRadius: theme.borderRadius.sm,
                fontSize: theme.fontSize.xs,
                color: theme.colors.text.muted,
                fontWeight: 'bold',
              }}>
                DETECTION ONLY
              </span>
            </div>

            {/* Connection Status */}
            <div style={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <motion.div
                animate={{ 
                  opacity: connectionStatus === 'connected' ? [0.5, 1, 0.5] : 1,
                }}
                transition={{ duration: 2, repeat: connectionStatus === 'connected' ? Infinity : 0 }}
                style={{
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  background: connectionStatus === 'connected' 
                    ? theme.colors.primary 
                    : connectionStatus === 'connecting'
                    ? theme.colors.warning
                    : theme.colors.danger,
                  boxShadow: `0 0 10px ${
                    connectionStatus === 'connected' 
                      ? theme.colors.primary 
                      : connectionStatus === 'connecting'
                      ? theme.colors.warning
                      : theme.colors.danger
                  }`,
                }}
              />
              <span style={{ 
                color: theme.colors.text.secondary,
                fontSize: theme.fontSize.sm,
                textTransform: 'capitalize',
              }}>
                {connectionStatus}
              </span>
            </div>
          </div>
        </motion.header>

        {/* Stats Bar */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          style={{
            background: theme.colors.bg.tertiary,
            padding: theme.spacing.md,
            borderBottom: `1px solid ${theme.colors.border.glass}`,
          }}
        >
          <div style={{ 
            maxWidth: '1600px', 
            margin: '0 auto',
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: theme.spacing.lg,
          }}>
            <div>
              <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                Total Detections
              </span>
              <p style={{ 
                color: theme.colors.text.primary,
                fontSize: theme.fontSize['2xl'],
                fontWeight: 'bold',
              }}>
                {stats.totalDetections.toLocaleString()}
              </p>
            </div>
            <div>
              <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                Critical Alerts
              </span>
              <p style={{ 
                color: theme.colors.danger,
                fontSize: theme.fontSize['2xl'],
                fontWeight: 'bold',
              }}>
                {stats.criticalAlerts}
              </p>
            </div>
            <div>
              <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                Avg Latency
              </span>
              <p style={{ 
                color: stats.avgLatency < 8 ? theme.colors.primary : theme.colors.warning,
                fontSize: theme.fontSize['2xl'],
                fontWeight: 'bold',
              }}>
                {stats.avgLatency.toFixed(1)}ms
              </p>
            </div>
            <div>
              <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                Active Entities
              </span>
              <p style={{ 
                color: theme.colors.secondary,
                fontSize: theme.fontSize['2xl'],
                fontWeight: 'bold',
              }}>
                {stats.activeEntities}
              </p>
            </div>
          </div>
        </motion.div>

        {/* Navigation Tabs */}
        <div style={{
          background: theme.colors.bg.secondary,
          padding: `${theme.spacing.md} 0`,
          borderBottom: `1px solid ${theme.colors.border.glass}`,
        }}>
          <div style={{ 
            maxWidth: '1600px', 
            margin: '0 auto',
            display: 'flex',
            gap: theme.spacing.md,
            paddingLeft: theme.spacing.lg,
            paddingRight: theme.spacing.lg,
          }}>
            {[
              { id: 'overview', label: 'Overview', icon: '📊' },
              { id: 'entities', label: 'Entity Analysis', icon: '🎯' },
              { id: 'detection', label: 'Detection Stream', icon: '🚨' },
              { id: 'models', label: 'Model Performance', icon: '🧠' },
              { id: 'dna', label: 'Decision DNA', icon: '🧬' },
            ].map(tab => (
              <motion.button
                key={tab.id}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setActiveTab(tab.id as any)}
                style={{
                  padding: `${theme.spacing.sm} ${theme.spacing.lg}`,
                  background: activeTab === tab.id 
                    ? `linear-gradient(135deg, ${theme.colors.primary}22, ${theme.colors.secondary}22)`
                    : 'transparent',
                  border: activeTab === tab.id 
                    ? `1px solid ${theme.colors.primary}`
                    : `1px solid ${theme.colors.border.glass}`,
                  borderRadius: theme.borderRadius.md,
                  color: activeTab === tab.id 
                    ? theme.colors.primary
                    : theme.colors.text.secondary,
                  fontSize: theme.fontSize.sm,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: theme.spacing.xs,
                  transition: 'all 0.3s',
                }}
              >
                <span>{tab.icon}</span>
                {tab.label}
              </motion.button>
            ))}
          </div>
        </div>

        {/* Main Content */}
        <div style={{ 
          maxWidth: '1600px', 
          margin: '0 auto',
          padding: theme.spacing.lg,
        }}>
          <AnimatePresence mode="wait">
            {activeTab === 'overview' && (
              <motion.div
                key="overview"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                style={{ display: 'grid', gap: theme.spacing.lg }}
              >
                <DetectionStream events={detectionEvents} maxEvents={10} />
                <EntityBehavioralSpectrum entities={entities} focusEntity={selectedEntity} />
              </motion.div>
            )}

            {activeTab === 'entities' && (
              <motion.div
                key="entities"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
              >
                <EntityBehavioralSpectrum 
                  entities={entities} 
                  focusEntity={selectedEntity}
                />
              </motion.div>
            )}

            {activeTab === 'detection' && (
              <motion.div
                key="detection"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
              >
                <DetectionStream events={detectionEvents} />
              </motion.div>
            )}

            {activeTab === 'models' && (
              <motion.div
                key="models"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
              >
                <ModelPerformance metrics={modelMetrics} />
              </motion.div>
            )}

            {activeTab === 'dna' && (
              <motion.div
                key="dna"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
              >
                <DecisionDNAExplorer nodes={dnaNodes} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Focus Entity Selector */}
        <motion.div
          initial={{ x: -100 }}
          animate={{ x: 0 }}
          style={{
            position: 'fixed',
            left: 0,
            top: '50%',
            transform: 'translateY(-50%)',
            background: theme.colors.bg.glass,
            backdropFilter: 'blur(10px)',
            borderRadius: `0 ${theme.borderRadius.lg} ${theme.borderRadius.lg} 0`,
            padding: theme.spacing.md,
            border: `1px solid ${theme.colors.border.primary}`,
            borderLeft: 'none',
          }}
        >
          <h3 style={{ 
            color: theme.colors.text.secondary,
            fontSize: theme.fontSize.sm,
            marginBottom: theme.spacing.sm,
          }}>
            Focus Entities
          </h3>
          {FOCUS_ENTITIES.map(entity => (
            <motion.button
              key={entity}
              whileHover={{ scale: 1.05 }}
              onClick={() => {
                setSelectedEntity(entity);
                setActiveTab('entities');
              }}
              style={{
                display: 'block',
                width: '100%',
                padding: theme.spacing.xs,
                background: selectedEntity === entity 
                  ? theme.colors.primary 
                  : 'transparent',
                color: selectedEntity === entity 
                  ? theme.colors.bg.primary 
                  : theme.colors.text.muted,
                border: 'none',
                borderRadius: theme.borderRadius.sm,
                fontSize: theme.fontSize.xs,
                fontFamily: 'monospace',
                cursor: 'pointer',
                marginBottom: theme.spacing.xs,
                textAlign: 'left',
              }}
            >
              {entity.slice(0, 6)}...{entity.slice(-4)}
            </motion.button>
          ))}
        </motion.div>
      </div>
    </QueryClientProvider>
  );
}

export default App;