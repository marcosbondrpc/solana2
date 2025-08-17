import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { 
  Power, 
  PowerOff, 
  RotateCcw, 
  ChevronUp, 
  ChevronDown,
  AlertTriangle,
  Shield,
  Terminal,
  CheckCircle,
  XCircle,
  Loader2,
  Lock,
  Unlock,
  PlayCircle,
  Database,
  HardDrive,
  Cpu
} from 'lucide-react';
import type { NodeStatus } from '../../pages/Node';

interface NodePowerDrawerProps {
  isOpen: boolean;
  onToggle: () => void;
  nodeRunning: boolean;
  nodeStatus: NodeStatus;
  twoManRuleEnabled: boolean;
  dryRunMode: boolean;
  onTwoManRuleToggle: (enabled: boolean) => void;
  onDryRunToggle: (enabled: boolean) => void;
  onStart: () => Promise<void>;
  onStop: () => Promise<void>;
  onRestart: () => Promise<void>;
}

export default function NodePowerDrawer({
  isOpen,
  onToggle,
  nodeRunning,
  nodeStatus,
  twoManRuleEnabled,
  dryRunMode,
  onTwoManRuleToggle,
  onDryRunToggle,
  onStart,
  onStop,
  onRestart,
}: NodePowerDrawerProps) {
  const [confirmAction, setConfirmAction] = useState<'start' | 'stop' | 'restart' | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [secondKeyEntered, setSecondKeyEntered] = useState(false);

  const handleAction = async (action: 'start' | 'stop' | 'restart') => {
    if (twoManRuleEnabled && !secondKeyEntered) {
      setConfirmAction(action);
      return;
    }

    setIsExecuting(true);
    try {
      switch (action) {
        case 'start':
          await onStart();
          break;
        case 'stop':
          await onStop();
          break;
        case 'restart':
          await onRestart();
          break;
      }
    } finally {
      setIsExecuting(false);
      setConfirmAction(null);
      setSecondKeyEntered(false);
    }
  };

  const confirmTwoManRule = () => {
    setSecondKeyEntered(true);
    if (confirmAction) {
      handleAction(confirmAction);
    }
  };

  // Preflight checks
  const preflightChecks = {
    ledgerFsync: true,
    snapshotsReady: true,
    catchUpDebt: nodeStatus.consensus.distanceToTip < 100,
    gossipConnected: nodeStatus.transport.gossip.peers > 0,
    jitoConnected: nodeStatus.jito.regions.length > 0,
  };

  const allChecksPass = Object.values(preflightChecks).every(v => v);

  return (
    <>
      {/* Toggle Button */}
      <motion.div
        className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50"
        initial={{ y: 100 }}
        animate={{ y: 0 }}
        transition={{ type: "spring", damping: 20 }}
      >
        <Button
          onClick={onToggle}
          className={`
            px-6 py-3 rounded-full shadow-2xl backdrop-blur-sm
            bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-700 hover:to-cyan-700
            text-white font-bold flex items-center gap-3
            border border-purple-500/30
          `}
        >
          <Power className="w-5 h-5" />
          NODE POWER
          {isOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
        </Button>
      </motion.div>

      {/* Drawer */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ y: "100%" }}
            animate={{ y: 0 }}
            exit={{ y: "100%" }}
            transition={{ type: "spring", damping: 25 }}
            className="fixed bottom-0 left-0 right-0 z-40 bg-gray-900/95 backdrop-blur-xl border-t border-gray-800 shadow-2xl"
          >
            <div className="container mx-auto px-6 py-6">
              <div className="grid grid-cols-12 gap-6">
                {/* Power Controls */}
                <div className="col-span-4">
                  <Card className="bg-gray-800/50 border-gray-700 p-6">
                    <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                      <Power className="w-5 h-5 text-purple-400" />
                      Node Control
                    </h3>

                    <div className="space-y-4">
                      {/* Power Buttons */}
                      <div className="grid grid-cols-3 gap-3">
                        <Button
                          onClick={() => handleAction('start')}
                          disabled={nodeRunning || isExecuting || !allChecksPass}
                          className={`
                            h-20 flex flex-col items-center justify-center gap-2
                            ${nodeRunning 
                              ? 'bg-gray-700 text-gray-500 cursor-not-allowed' 
                              : 'bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white'
                            }
                          `}
                        >
                          {isExecuting && confirmAction === 'start' ? (
                            <Loader2 className="w-6 h-6 animate-spin" />
                          ) : (
                            <PlayCircle className="w-6 h-6" />
                          )}
                          <span className="text-xs font-bold">START</span>
                        </Button>

                        <Button
                          onClick={() => handleAction('stop')}
                          disabled={!nodeRunning || isExecuting}
                          className={`
                            h-20 flex flex-col items-center justify-center gap-2
                            ${!nodeRunning 
                              ? 'bg-gray-700 text-gray-500 cursor-not-allowed' 
                              : 'bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700 text-white'
                            }
                          `}
                        >
                          {isExecuting && confirmAction === 'stop' ? (
                            <Loader2 className="w-6 h-6 animate-spin" />
                          ) : (
                            <PowerOff className="w-6 h-6" />
                          )}
                          <span className="text-xs font-bold">STOP</span>
                        </Button>

                        <Button
                          onClick={() => handleAction('restart')}
                          disabled={!nodeRunning || isExecuting}
                          className={`
                            h-20 flex flex-col items-center justify-center gap-2
                            ${!nodeRunning 
                              ? 'bg-gray-700 text-gray-500 cursor-not-allowed' 
                              : 'bg-gradient-to-r from-yellow-600 to-orange-600 hover:from-yellow-700 hover:to-orange-700 text-white'
                            }
                          `}
                        >
                          {isExecuting && confirmAction === 'restart' ? (
                            <Loader2 className="w-6 h-6 animate-spin" />
                          ) : (
                            <RotateCcw className="w-6 h-6" />
                          )}
                          <span className="text-xs font-bold">RESTART</span>
                        </Button>
                      </div>

                      {/* Safety Options */}
                      <div className="space-y-3">
                        <div className="flex items-center justify-between p-3 bg-gray-900/50 rounded-lg">
                          <div className="flex items-center gap-2">
                            <Shield className="w-4 h-4 text-yellow-400" />
                            <span className="text-sm text-gray-300">2-Man Rule</span>
                          </div>
                          <button
                            onClick={() => onTwoManRuleToggle(!twoManRuleEnabled)}
                            className={`
                              w-12 h-6 rounded-full transition-colors relative
                              ${twoManRuleEnabled ? 'bg-yellow-600' : 'bg-gray-600'}
                            `}
                          >
                            <div 
                              className={`
                                w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform
                                ${twoManRuleEnabled ? 'translate-x-6' : 'translate-x-0.5'}
                              `}
                            />
                          </button>
                        </div>

                        <div className="flex items-center justify-between p-3 bg-gray-900/50 rounded-lg">
                          <div className="flex items-center gap-2">
                            <Terminal className="w-4 h-4 text-cyan-400" />
                            <span className="text-sm text-gray-300">Dry Run Mode</span>
                          </div>
                          <button
                            onClick={() => onDryRunToggle(!dryRunMode)}
                            className={`
                              w-12 h-6 rounded-full transition-colors relative
                              ${dryRunMode ? 'bg-cyan-600' : 'bg-gray-600'}
                            `}
                          >
                            <div 
                              className={`
                                w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform
                                ${dryRunMode ? 'translate-x-6' : 'translate-x-0.5'}
                              `}
                            />
                          </button>
                        </div>
                      </div>

                      {dryRunMode && (
                        <div className="p-3 bg-cyan-900/20 border border-cyan-800/30 rounded-lg">
                          <p className="text-xs text-cyan-400 flex items-center gap-2">
                            <AlertTriangle className="w-3 h-3" />
                            Commands will be simulated only
                          </p>
                        </div>
                      )}
                    </div>
                  </Card>
                </div>

                {/* Preflight Checks */}
                <div className="col-span-4">
                  <Card className="bg-gray-800/50 border-gray-700 p-6">
                    <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                      <CheckCircle className="w-5 h-5 text-green-400" />
                      Preflight Checks
                    </h3>

                    <div className="space-y-3">
                      <div className="flex items-center justify-between p-3 bg-gray-900/50 rounded-lg">
                        <div className="flex items-center gap-2">
                          <HardDrive className="w-4 h-4 text-gray-400" />
                          <span className="text-sm text-gray-300">Ledger Fsync</span>
                        </div>
                        {preflightChecks.ledgerFsync ? (
                          <CheckCircle className="w-5 h-5 text-green-400" />
                        ) : (
                          <XCircle className="w-5 h-5 text-red-400" />
                        )}
                      </div>

                      <div className="flex items-center justify-between p-3 bg-gray-900/50 rounded-lg">
                        <div className="flex items-center gap-2">
                          <Database className="w-4 h-4 text-gray-400" />
                          <span className="text-sm text-gray-300">Snapshots Ready</span>
                        </div>
                        {preflightChecks.snapshotsReady ? (
                          <CheckCircle className="w-5 h-5 text-green-400" />
                        ) : (
                          <XCircle className="w-5 h-5 text-red-400" />
                        )}
                      </div>

                      <div className="flex items-center justify-between p-3 bg-gray-900/50 rounded-lg">
                        <div className="flex items-center gap-2">
                          <Cpu className="w-4 h-4 text-gray-400" />
                          <span className="text-sm text-gray-300">Catch-up Debt</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-gray-500">{nodeStatus.consensus.distanceToTip} slots</span>
                          {preflightChecks.catchUpDebt ? (
                            <CheckCircle className="w-5 h-5 text-green-400" />
                          ) : (
                            <AlertTriangle className="w-5 h-5 text-yellow-400" />
                          )}
                        </div>
                      </div>

                      <div className="flex items-center justify-between p-3 bg-gray-900/50 rounded-lg">
                        <div className="flex items-center gap-2">
                          <Shield className="w-4 h-4 text-gray-400" />
                          <span className="text-sm text-gray-300">Gossip Connected</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-gray-500">{nodeStatus.transport.gossip.peers} peers</span>
                          {preflightChecks.gossipConnected ? (
                            <CheckCircle className="w-5 h-5 text-green-400" />
                          ) : (
                            <XCircle className="w-5 h-5 text-red-400" />
                          )}
                        </div>
                      </div>

                      <div className="flex items-center justify-between p-3 bg-gray-900/50 rounded-lg">
                        <div className="flex items-center gap-2">
                          <Shield className="w-4 h-4 text-gray-400" />
                          <span className="text-sm text-gray-300">Jito Connected</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-gray-500">{nodeStatus.jito.regions.length} regions</span>
                          {preflightChecks.jitoConnected ? (
                            <CheckCircle className="w-5 h-5 text-green-400" />
                          ) : (
                            <XCircle className="w-5 h-5 text-red-400" />
                          )}
                        </div>
                      </div>
                    </div>

                    {!allChecksPass && (
                      <div className="mt-4 p-3 bg-red-900/20 border border-red-800/30 rounded-lg">
                        <p className="text-xs text-red-400 flex items-center gap-2">
                          <AlertTriangle className="w-3 h-3" />
                          Some checks failed. Review before starting.
                        </p>
                      </div>
                    )}
                  </Card>
                </div>

                {/* Real-time Status */}
                <div className="col-span-4">
                  <Card className="bg-gray-800/50 border-gray-700 p-6">
                    <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                      <Terminal className="w-5 h-5 text-cyan-400" />
                      Real-time Status
                    </h3>

                    <div className="bg-black/50 rounded-lg p-3 font-mono text-xs text-green-400 h-64 overflow-y-auto">
                      <pre className="whitespace-pre-wrap">
{JSON.stringify({
  node: {
    status: nodeRunning ? 'RUNNING' : 'STOPPED',
    uptime: nodeRunning ? '2h 34m 12s' : '0s',
    version: nodeStatus.client.version,
    commit: nodeStatus.client.commit,
  },
  consensus: {
    slot: nodeStatus.consensus.slot,
    ledgerHeight: nodeStatus.consensus.ledgerHeight,
    distanceToTip: nodeStatus.consensus.distanceToTip,
    delinquent: nodeStatus.consensus.delinquent,
  },
  rpc: {
    subscriptions: nodeStatus.rpc.subscriptions.count,
    lag: `${nodeStatus.rpc.subscriptions.lag}ms`,
  },
  quic: {
    handshakeRate: `${nodeStatus.transport.quic.handshakeSuccessRate.toFixed(1)}%`,
    connections: nodeStatus.transport.quic.concurrentConnections,
    throttling: nodeStatus.transport.quic.throttlingEvents,
  },
  jito: {
    regions: nodeStatus.jito.regions,
    acceptanceRate: `${nodeStatus.jito.bundleMetrics.acceptanceRate.toFixed(1)}%`,
    shredstream: nodeStatus.jito.shredstreamProxy ? 'ACTIVE' : 'INACTIVE',
  },
  geyser: {
    throughput: `${(Math.random() * 50000).toFixed(0)} msg/s`,
  }
}, null, 2)}
                      </pre>
                    </div>
                  </Card>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Two-Man Rule Confirmation Modal */}
      <AnimatePresence>
        {confirmAction && twoManRuleEnabled && !secondKeyEntered && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-gray-900 border border-yellow-600 rounded-lg p-6 max-w-md"
            >
              <div className="flex items-center gap-3 mb-4">
                <Lock className="w-6 h-6 text-yellow-400" />
                <h3 className="text-xl font-bold text-white">Two-Man Rule Active</h3>
              </div>
              
              <p className="text-gray-400 mb-6">
                This action requires confirmation from a second operator.
                Action to confirm: <span className="text-yellow-400 font-bold uppercase">{confirmAction}</span>
              </p>

              <div className="flex gap-3">
                <Button
                  onClick={confirmTwoManRule}
                  className="flex-1 bg-gradient-to-r from-yellow-600 to-orange-600 hover:from-yellow-700 hover:to-orange-700 text-white"
                >
                  <Unlock className="w-4 h-4 mr-2" />
                  Confirm as Second Operator
                </Button>
                <Button
                  onClick={() => setConfirmAction(null)}
                  className="flex-1 bg-gray-700 hover:bg-gray-600 text-white"
                >
                  Cancel
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}