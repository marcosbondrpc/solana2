import React, { useState, useMemo, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import * as d3 from 'd3';

interface EntityData {
  address: string;
  label: string;
  type: 'bot' | 'dex' | 'suspicious';
  totalVolume: number;
  transactionCount: number;
  avgTransactionSize: number;
  sandwichAttacks: number;
  arbitrageCount: number;
  liquidations: number;
  profitSOL: number;
  victimCount: number;
  gasSpent: number;
  successRate: number;
  firstSeen: Date;
  lastSeen: Date;
  riskScore: number;
  activityHours: number[];
  walletNetwork: string[];
  preferredDEX: string[];
}

const TARGET_ADDRESSES: { address: string; label: string; type: 'bot' | 'dex' | 'suspicious' }[] = [
  { address: 'B91piBSfCBRs5rUxCMRdJEGv7tNEnFxweWcdQJHJoFpi', label: 'Alpha Bot', type: 'bot' },
  { address: '6gAnjderE13TGGFeqdPVQ438jp2FPVeyXAszxKu9y338', label: 'Arbitrage Pro', type: 'bot' },
  { address: 'CaCZgxpiEDZtCgXABN9X8PTwkRk81Y9PKdc6s66frH7q', label: 'MEV Hunter', type: 'bot' },
  { address: 'D9Akv6CQyExvjtEA22g6AAEgCs4GYWzxVoC5UA4fWXEC', label: 'Sandwich King', type: 'suspicious' },
  { address: 'E6YoRP3adE5XYneSseLee15wJshDxCsmyD2WtLvAmfLi', label: 'Flash Loan Expert', type: 'bot' },
  { address: 'GGG4BBhgAYKXwjpHjTJyKXWaUnPnHvJ5fnXpG1jvZcNQ', label: 'Liquidator Prime', type: 'bot' },
  { address: 'EAJ1DPoYR9GhaBbSWHmFiyAtssy8Qi7dPvTzBdwkCYMW', label: 'JIT Master', type: 'suspicious' },
  { address: '2brXWR3RYHXsAcbo58LToDV9KVd7D2dyH9iP3qa9PZTy', label: 'Volume Bot', type: 'bot' },
  { address: 'CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C', label: 'Raydium', type: 'dex' },
  { address: 'pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA', label: 'PumpSwap', type: 'dex' }
];

interface EntityAnalyticsProps {
  entityData?: EntityData[];
  onEntitySelect?: (address: string) => void;
}

export const EntityAnalytics: React.FC<EntityAnalyticsProps> = ({
  entityData = [],
  onEntitySelect
}) => {
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d'>('30d');
  const [sortBy, setSortBy] = useState<'volume' | 'profit' | 'risk' | 'victims'>('volume');
  const [animatedValues, setAnimatedValues] = useState<{ [key: string]: number }>({});

  // Generate mock data if not provided
  const entities = useMemo(() => {
    if (entityData.length > 0) return entityData;
    
    return TARGET_ADDRESSES.map(target => ({
      address: target.address,
      label: target.label,
      type: target.type,
      totalVolume: Math.random() * 50000 + 1000,
      transactionCount: Math.floor(Math.random() * 10000 + 100),
      avgTransactionSize: Math.random() * 100 + 10,
      sandwichAttacks: Math.floor(Math.random() * 500),
      arbitrageCount: Math.floor(Math.random() * 1000),
      liquidations: Math.floor(Math.random() * 200),
      profitSOL: Math.random() * 5000 - 500,
      victimCount: Math.floor(Math.random() * 2000),
      gasSpent: Math.random() * 1000,
      successRate: Math.random() * 100,
      firstSeen: new Date(Date.now() - Math.random() * 90 * 24 * 60 * 60 * 1000),
      lastSeen: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000),
      riskScore: Math.random() * 100,
      activityHours: Array(24).fill(0).map(() => Math.random()),
      walletNetwork: Array(Math.floor(Math.random() * 5 + 1)).fill(0).map((_, i) => 
        `Wallet${i + 1}_${Math.random().toString(36).substr(2, 9)}`
      ),
      preferredDEX: ['Raydium', 'Orca', 'PumpSwap'].filter(() => Math.random() > 0.5)
    }));
  }, [entityData]);

  // Sort entities
  const sortedEntities = useMemo(() => {
    return [...entities].sort((a, b) => {
      switch (sortBy) {
        case 'volume': return b.totalVolume - a.totalVolume;
        case 'profit': return b.profitSOL - a.profitSOL;
        case 'risk': return b.riskScore - a.riskScore;
        case 'victims': return b.victimCount - a.victimCount;
        default: return 0;
      }
    });
  }, [entities, sortBy]);

  // Animate values
  useEffect(() => {
    const interval = setInterval(() => {
      setAnimatedValues(prev => {
        const next = { ...prev };
        entities.forEach(entity => {
          const key = `${entity.address}-volume`;
          if (!next[key] || next[key] < entity.totalVolume) {
            next[key] = Math.min((next[key] || 0) + entity.totalVolume / 50, entity.totalVolume);
          }
        });
        return next;
      });
    }, 50);

    return () => clearInterval(interval);
  }, [entities]);

  const getRiskColor = (score: number) => {
    if (score < 30) return 'text-green-400';
    if (score < 70) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getRiskBg = (score: number) => {
    if (score < 30) return 'bg-green-500/10';
    if (score < 70) return 'bg-yellow-500/10';
    return 'bg-red-500/10';
  };

  return (
    <div className="space-y-6">
      {/* Header Controls */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-zinc-100">
          Entity Analytics & Tracking
        </h2>
        <div className="flex items-center gap-4">
          <div className="flex gap-2">
            {(['7d', '30d', '90d'] as const).map(range => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-3 py-1 rounded-lg text-sm transition-all ${
                  timeRange === range
                    ? 'bg-gradient-to-r from-cyan-600 to-purple-600 text-white'
                    : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
                }`}
              >
                {range}
              </button>
            ))}
          </div>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="bg-zinc-800 text-zinc-300 px-3 py-1 rounded-lg text-sm border border-zinc-700 focus:outline-none focus:border-cyan-500"
          >
            <option value="volume">Sort by Volume</option>
            <option value="profit">Sort by Profit</option>
            <option value="risk">Sort by Risk</option>
            <option value="victims">Sort by Victims</option>
          </select>
        </div>
      </div>

      {/* Entity Cards Grid */}
      <div className="grid grid-cols-2 gap-4">
        {sortedEntities.map((entity, index) => (
          <motion.div
            key={entity.address}
            className={`glass rounded-xl p-6 cursor-pointer transition-all ${
              selectedEntity === entity.address ? 'ring-2 ring-cyan-500' : ''
            }`}
            onClick={() => {
              setSelectedEntity(entity.address);
              onEntitySelect?.(entity.address);
            }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            whileHover={{ scale: 1.02 }}
          >
            {/* Entity Header */}
            <div className="flex items-start justify-between mb-4">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <h3 className="font-semibold text-zinc-100">{entity.label}</h3>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                    entity.type === 'bot' ? 'bg-blue-500/20 text-blue-400' :
                    entity.type === 'dex' ? 'bg-green-500/20 text-green-400' :
                    'bg-red-500/20 text-red-400'
                  }`}>
                    {entity.type.toUpperCase()}
                  </span>
                </div>
                <div className="font-mono text-xs text-zinc-500">
                  {entity.address.slice(0, 8)}...{entity.address.slice(-6)}
                </div>
              </div>
              <div className={`text-right ${getRiskBg(entity.riskScore)} px-2 py-1 rounded`}>
                <div className={`text-lg font-bold ${getRiskColor(entity.riskScore)}`}>
                  {entity.riskScore.toFixed(0)}
                </div>
                <div className="text-xs text-zinc-500">Risk</div>
              </div>
            </div>

            {/* Key Metrics */}
            <div className="grid grid-cols-3 gap-3 mb-4">
              <div>
                <div className="text-xs text-zinc-500 mb-1">Volume</div>
                <div className="text-lg font-semibold text-zinc-200">
                  {(animatedValues[`${entity.address}-volume`] || 0).toFixed(0)} SOL
                </div>
              </div>
              <div>
                <div className="text-xs text-zinc-500 mb-1">Profit</div>
                <div className={`text-lg font-semibold ${
                  entity.profitSOL > 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {entity.profitSOL > 0 ? '+' : ''}{entity.profitSOL.toFixed(1)} SOL
                </div>
              </div>
              <div>
                <div className="text-xs text-zinc-500 mb-1">Success</div>
                <div className="text-lg font-semibold text-zinc-200">
                  {entity.successRate.toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Attack Types */}
            <div className="flex gap-2 mb-4">
              {entity.sandwichAttacks > 0 && (
                <div className="bg-orange-500/10 px-2 py-1 rounded text-xs text-orange-400">
                  🥪 {entity.sandwichAttacks}
                </div>
              )}
              {entity.arbitrageCount > 0 && (
                <div className="bg-blue-500/10 px-2 py-1 rounded text-xs text-blue-400">
                  ⚡ {entity.arbitrageCount}
                </div>
              )}
              {entity.liquidations > 0 && (
                <div className="bg-red-500/10 px-2 py-1 rounded text-xs text-red-400">
                  💀 {entity.liquidations}
                </div>
              )}
            </div>

            {/* Activity Sparkline */}
            <div className="h-8 flex items-end gap-0.5">
              {entity.activityHours.slice(-24).map((activity, i) => (
                <div
                  key={i}
                  className="flex-1 bg-gradient-to-t from-cyan-600 to-purple-600 rounded-t opacity-80"
                  style={{ height: `${activity * 100}%` }}
                  title={`${i}:00 - Activity: ${(activity * 100).toFixed(0)}%`}
                />
              ))}
            </div>

            {/* Footer Stats */}
            <div className="flex items-center justify-between mt-4 pt-4 border-t border-zinc-800">
              <div className="flex items-center gap-4 text-xs text-zinc-500">
                <span>👁 {entity.victimCount} victims</span>
                <span>⛽ {entity.gasSpent.toFixed(1)} SOL gas</span>
              </div>
              <div className="flex items-center gap-1">
                {entity.preferredDEX.map(dex => (
                  <span key={dex} className="px-2 py-0.5 bg-zinc-800 rounded text-xs text-zinc-400">
                    {dex}
                  </span>
                ))}
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Detailed View for Selected Entity */}
      <AnimatePresence>
        {selectedEntity && (
          <motion.div
            className="glass rounded-xl p-6"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            {(() => {
              const entity = entities.find(e => e.address === selectedEntity);
              if (!entity) return null;

              return (
                <>
                  <h3 className="text-lg font-semibold text-zinc-100 mb-4">
                    Detailed Analysis: {entity.label}
                  </h3>
                  
                  {/* Wallet Network */}
                  <div className="mb-4">
                    <h4 className="text-sm font-medium text-zinc-400 mb-2">Connected Wallets</h4>
                    <div className="flex flex-wrap gap-2">
                      {entity.walletNetwork.map(wallet => (
                        <span
                          key={wallet}
                          className="px-3 py-1 bg-zinc-800 rounded-lg text-xs font-mono text-zinc-400 hover:bg-zinc-700 cursor-pointer"
                        >
                          {wallet}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Time Analysis */}
                  <div className="grid grid-cols-3 gap-4">
                    <div className="bg-zinc-900/50 rounded-lg p-3">
                      <div className="text-xs text-zinc-500 mb-1">First Seen</div>
                      <div className="text-sm text-zinc-300">
                        {entity.firstSeen.toLocaleDateString()}
                      </div>
                    </div>
                    <div className="bg-zinc-900/50 rounded-lg p-3">
                      <div className="text-xs text-zinc-500 mb-1">Last Active</div>
                      <div className="text-sm text-zinc-300">
                        {entity.lastSeen.toLocaleString()}
                      </div>
                    </div>
                    <div className="bg-zinc-900/50 rounded-lg p-3">
                      <div className="text-xs text-zinc-500 mb-1">Days Active</div>
                      <div className="text-sm text-zinc-300">
                        {Math.floor((Date.now() - entity.firstSeen.getTime()) / (24 * 60 * 60 * 1000))}
                      </div>
                    </div>
                  </div>
                </>
              );
            })()}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};