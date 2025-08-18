import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import * as Tabs from '@radix-ui/react-tabs';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { useWebSocket } from '../hooks/use-websocket';
import { 
  Power, 
  PowerOff, 
  RotateCcw, 
  AlertTriangle, 
  ChevronUp, 
  ChevronDown,
  Zap,
  Activity,
  Server,
  Cpu,
  Network,
  Shield,
  Database,
  Download
} from 'lucide-react';

// Import sub-components
import OverviewTab from '../components/node/OverviewTab';
import LowLatencyJitoTab from '../components/node/LowLatencyJitoTab';
import TransportTPUTab from '../components/node/TransportTPUTab';
import NodePowerDrawer from '../components/node/NodePowerDrawer';
import { DatasetExporter } from '../components/node/DatasetExporter';

// Types
export interface NodeStatus {
  client: {
    version: string;
    commit: string;
    cluster: string;
    identity: string;
    votePubkey: string;
  };
  consensus: {
    ledgerHeight: number;
    slot: number;
    distanceToTip: number;
    epochProgress: number;
    upcomingLeaderSlots: number[];
    voteCredits: number;
    delinquent: boolean;
    recentBlockProduction: {
      success: number;
      skipped: number;
      total: number;
    };
  };
  cluster: {
    tps: number[];
    confirmationTime: {
      p50: number;
      p90: number;
      p99: number;
    };
    networkMetrics: {
      totalStake: number;
      activeValidators: number;
      epochTimeRemaining: number;
    };
  };
  jito: {
    regions: string[];
    minTip: number;
    auctionTick: number;
    bundleAuthUuid: string;
    shredstreamProxy: boolean;
    tipFeed: number[];
    bundleMetrics: {
      acceptanceRate: number;
      rejectionReasons: Record<string, number>;
      landingDelay: number[];
      multiRegionFailoverRate: number;
    };
    shredstream: {
      packetsPerSec: number;
      gaps: number;
      reorders: number;
      decodeTime: number;
    };
  };
  rpc: {
    methods: Record<string, { p50: number; p99: number }>;
    subscriptions: {
      count: number;
      lag: number;
    };
  };
  transport: {
    quic: {
      handshakeSuccessRate: number;
      concurrentConnections: number;
      openStreamsPerPeer: Record<string, number>;
      throttlingEvents: number;
      ppsRateLimiting: number[];
    };
    qos: {
      whitelistedRpcs: Array<{ address: string; stake: number }>;
      virtualStake: number;
      leaderTpuPort: number;
      pinPeeringStatus: boolean;
    };
    gossip: {
      peers: number;
      shredsSent: number;
      shredsReceived: number;
      retransmits: number[];
    };
  };
}

const DEFAULT_STATUS: NodeStatus = {
  client: {
    version: '1.17.28',
    commit: 'a8b3f4d',
    cluster: 'mainnet-beta',
    identity: '...',
    votePubkey: '...',
  },
  consensus: {
    ledgerHeight: 0,
    slot: 0,
    distanceToTip: 0,
    epochProgress: 0,
    upcomingLeaderSlots: [],
    voteCredits: 0,
    delinquent: false,
    recentBlockProduction: {
      success: 0,
      skipped: 0,
      total: 0,
    },
  },
  cluster: {
    tps: [],
    confirmationTime: {
      p50: 0,
      p90: 0,
      p99: 0,
    },
    networkMetrics: {
      totalStake: 0,
      activeValidators: 0,
      epochTimeRemaining: 0,
    },
  },
  jito: {
    regions: [],
    minTip: 0,
    auctionTick: 50,
    bundleAuthUuid: '',
    shredstreamProxy: false,
    tipFeed: [],
    bundleMetrics: {
      acceptanceRate: 0,
      rejectionReasons: {},
      landingDelay: [],
      multiRegionFailoverRate: 0,
    },
    shredstream: {
      packetsPerSec: 0,
      gaps: 0,
      reorders: 0,
      decodeTime: 0,
    },
  },
  rpc: {
    methods: {},
    subscriptions: {
      count: 0,
      lag: 0,
    },
  },
  transport: {
    quic: {
      handshakeSuccessRate: 0,
      concurrentConnections: 0,
      openStreamsPerPeer: {},
      throttlingEvents: 0,
      ppsRateLimiting: [],
    },
    qos: {
      whitelistedRpcs: [],
      virtualStake: 0,
      leaderTpuPort: 0,
      pinPeeringStatus: false,
    },
    gossip: {
      peers: 0,
      shredsSent: 0,
      shredsReceived: 0,
      retransmits: [],
    },
  },
};

export default function NodePage() {
  const [nodeStatus, setNodeStatus] = useState<NodeStatus>(DEFAULT_STATUS);
  const [activeTab, setActiveTab] = useState('overview');
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [nodeRunning, setNodeRunning] = useState(true);
  const [twoManRuleEnabled, setTwoManRuleEnabled] = useState(false);
  const [dryRunMode, setDryRunMode] = useState(false);
  const [exportModalOpen, setExportModalOpen] = useState(false);

  const { data, isConnected } = ((useWebSocket as any)('node-status') as any);

  useEffect(() => {
    if (data) {
      setNodeStatus(prev => ({
        ...prev,
        ...data,
      }));
    }
  }, [data]);

  const handleNodeStart = useCallback(async () => {
    if (dryRunMode) {
      console.log('[DRY RUN] Starting node...');
      return;
    }
    // Implement actual node start logic
    setNodeRunning(true);
  }, [dryRunMode]);

  const handleNodeStop = useCallback(async () => {
    if (dryRunMode) {
      console.log('[DRY RUN] Stopping node...');
      return;
    }
    // Implement actual node stop logic
    setNodeRunning(false);
  }, [dryRunMode]);

  const handleNodeRestart = useCallback(async () => {
    if (dryRunMode) {
      console.log('[DRY RUN] Restarting node...');
      return;
    }
    await handleNodeStop();
    setTimeout(() => handleNodeStart(), 2000);
  }, [dryRunMode, handleNodeStart, handleNodeStop]);

  return (
    <div className="min-h-screen bg-[#0a0a0a] relative pb-24">
      {/* Animated Background Grid */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute inset-0 bg-gradient-to-b from-purple-900/5 via-transparent to-cyan-900/5" />
        <svg className="absolute inset-0 w-full h-full opacity-10">
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" strokeWidth="0.5" className="text-purple-500" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
      </div>

      {/* Main Content */}
      <div className="relative z-10 container mx-auto px-6 py-8">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-cyan-500 blur-xl opacity-50" />
                <h1 className="relative text-5xl font-black bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent">
                  MISSION CONTROL
                </h1>
              </div>
              <Badge 
                className={`${isConnected ? 'bg-green-500/20 text-green-400 border-green-500' : 'bg-red-500/20 text-red-400 border-red-500'} animate-pulse`}
              >
                {isConnected ? 'ONLINE' : 'OFFLINE'}
              </Badge>
            </div>
            <div className="flex items-center gap-3">
              <Badge className="bg-purple-500/20 text-purple-400 border-purple-500">
                {nodeStatus.client.cluster}
              </Badge>
              <Badge className="bg-cyan-500/20 text-cyan-400 border-cyan-500">
                v{nodeStatus.client.version}
              </Badge>
              <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500">
                Slot: {nodeStatus.consensus.slot.toLocaleString()}
              </Badge>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setExportModalOpen(true)}
                className="px-4 py-2 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 hover:from-cyan-500/30 hover:to-blue-500/30 border border-cyan-500/30 rounded-lg flex items-center gap-2 text-cyan-400 font-medium transition-all shadow-lg hover:shadow-cyan-500/20"
              >
                <Database className="w-4 h-4" />
                Export Dataset
              </motion.button>
            </div>
          </div>
          <p className="text-gray-500 mt-2 text-sm uppercase tracking-wider">
            Solana Private Node Operations Center
          </p>
        </motion.div>

        {/* Tabs */}
        <Tabs.Root value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <Tabs.List className="flex gap-2 p-1 bg-gray-900/50 backdrop-blur-sm rounded-lg border border-gray-800">
            <Tabs.Trigger
              value="overview"
              className={`flex-1 px-6 py-3 rounded-md font-medium transition-all ${
                activeTab === 'overview'
                  ? 'bg-gradient-to-r from-purple-500 to-blue-500 text-white shadow-lg'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4" />
                Overview
              </div>
            </Tabs.Trigger>
            <Tabs.Trigger
              value="lowlatency"
              className={`flex-1 px-6 py-3 rounded-md font-medium transition-all ${
                activeTab === 'lowlatency'
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-lg'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4" />
                Low-latency + Jito
              </div>
            </Tabs.Trigger>
            <Tabs.Trigger
              value="transport"
              className={`flex-1 px-6 py-3 rounded-md font-medium transition-all ${
                activeTab === 'transport'
                  ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white shadow-lg'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              <div className="flex items-center gap-2">
                <Network className="w-4 h-4" />
                Transport & TPU
              </div>
            </Tabs.Trigger>
          </Tabs.List>

          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
            >
              <Tabs.Content value="overview">
                <OverviewTab nodeStatus={nodeStatus} />
              </Tabs.Content>
              <Tabs.Content value="lowlatency">
                <LowLatencyJitoTab nodeStatus={nodeStatus} />
              </Tabs.Content>
              <Tabs.Content value="transport">
                <TransportTPUTab nodeStatus={nodeStatus} />
              </Tabs.Content>
            </motion.div>
          </AnimatePresence>
        </Tabs.Root>
      </div>

      {/* Node Power Drawer */}
      <NodePowerDrawer
        isOpen={drawerOpen}
        onToggle={() => setDrawerOpen(!drawerOpen)}
        nodeRunning={nodeRunning}
        nodeStatus={nodeStatus}
        twoManRuleEnabled={twoManRuleEnabled}
        dryRunMode={dryRunMode}
        onTwoManRuleToggle={setTwoManRuleEnabled}
        onDryRunToggle={setDryRunMode}
        onStart={handleNodeStart}
        onStop={handleNodeStop}
        onRestart={handleNodeRestart}
      />

      {/* Dataset Export Modal */}
      <DatasetExporter
        isOpen={exportModalOpen}
        onClose={() => setExportModalOpen(false)}
      />

      {/* Floating Export Button */}
      <motion.div
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.5, type: 'spring' }}
        className="fixed bottom-24 right-8 z-40"
      >
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => setExportModalOpen(true)}
          className="relative p-4 bg-gradient-to-br from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 rounded-full shadow-2xl shadow-cyan-500/30 group"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-cyan-400 to-blue-400 rounded-full blur-xl opacity-50 group-hover:opacity-75 transition-opacity" />
          <Download className="relative w-6 h-6 text-white" />
          <div className="absolute -top-8 right-0 px-3 py-1 bg-gray-900 border border-cyan-500/30 rounded-lg text-xs text-cyan-400 font-medium opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
            Export ML Dataset
          </div>
        </motion.button>
      </motion.div>
    </div>
  );
}