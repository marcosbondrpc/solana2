/**
 * MEV Opportunities Panel - Real-time opportunity tracking
 * Displays arbitrage, liquidation, and sandwich opportunities
 */

'use client';

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FixedSizeList as List } from 'react-window';
import { useMEVStore } from '../../stores/mevStore';
import { apiService, type MEVOpportunity } from '../../lib/api-service';
import { mevWebSocket } from '../../lib/enhanced-websocket';
import { toast } from 'sonner';

interface OpportunityFilter {
  type: string[];
  minProfit: number;
  maxGas: number;
  minConfidence: number;
}

interface OpportunityStats {
  total: number;
  arbitrage: number;
  liquidation: number;
  sandwich: number;
  jit: number;
  totalProfit: number;
  avgProfit: number;
  successRate: number;
}

const OpportunityRow: React.FC<{
  opportunity: MEVOpportunity;
  onExecute: (id: string) => void;
  isExecuting: boolean;
}> = React.memo(({ opportunity, onExecute, isExecuting }) => {
  const profitColor = 
    opportunity.expected_profit > 1000 ? 'text-green-400' :
    opportunity.expected_profit > 100 ? 'text-yellow-400' :
    'text-orange-400';

  const confidenceColor =
    opportunity.confidence > 0.8 ? 'text-green-400' :
    opportunity.confidence > 0.6 ? 'text-yellow-400' :
    'text-orange-400';

  const typeColors = {
    arbitrage: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    liquidation: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
    sandwich: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
    jit: 'bg-green-500/20 text-green-400 border-green-500/30'
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      className="p-3 bg-gray-900/50 hover:bg-gray-900/70 rounded-lg border border-gray-700 hover:border-cyan-500/30 transition-all cursor-pointer group"
      onClick={() => onExecute(opportunity.id)}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          {/* Type Badge */}
          <div className={`px-2 py-1 rounded text-xs font-semibold border ${typeColors[opportunity.type]}`}>
            {opportunity.type.toUpperCase()}
          </div>

          {/* Token Pair */}
          <div className="text-sm font-mono text-cyan-300">
            {opportunity.token_pair}
          </div>

          {/* Route */}
          <div className="text-xs text-gray-400">
            {opportunity.dex_a} → {opportunity.dex_b}
          </div>

          {/* Decision DNA */}
          {opportunity.decision_dna && (
            <div className="text-xs text-gray-500 font-mono" title="Decision DNA">
              {opportunity.decision_dna.slice(0, 8)}...
            </div>
          )}
        </div>

        <div className="flex items-center gap-6">
          {/* Expected Profit */}
          <div className="text-right">
            <div className="text-xs text-gray-400">Profit</div>
            <div className={`font-bold ${profitColor}`}>
              ${opportunity.expected_profit.toFixed(2)}
            </div>
          </div>

          {/* Confidence */}
          <div className="text-right">
            <div className="text-xs text-gray-400">Confidence</div>
            <div className={`font-bold ${confidenceColor}`}>
              {(opportunity.confidence * 100).toFixed(1)}%
            </div>
          </div>

          {/* Gas Estimate */}
          <div className="text-right">
            <div className="text-xs text-gray-400">Gas</div>
            <div className="font-mono text-sm text-gray-300">
              {opportunity.gas_estimate.toFixed(4)} SOL
            </div>
          </div>

          {/* Execution Window */}
          <div className="text-right">
            <div className="text-xs text-gray-400">Window</div>
            <div className="font-mono text-sm text-cyan-400">
              {opportunity.execution_window_ms}ms
            </div>
          </div>

          {/* Execute Button */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              onExecute(opportunity.id);
            }}
            disabled={isExecuting}
            className="px-3 py-1 bg-cyan-500/20 text-cyan-400 rounded hover:bg-cyan-500/30 transition-colors text-sm font-semibold disabled:opacity-50 disabled:cursor-not-allowed opacity-0 group-hover:opacity-100"
          >
            {isExecuting ? 'Executing...' : 'Execute'}
          </button>
        </div>
      </div>

      {/* Slippage Warning */}
      {opportunity.slippage_tolerance > 0.05 && (
        <div className="mt-2 text-xs text-yellow-400 flex items-center gap-1">
          ⚠ High slippage: {(opportunity.slippage_tolerance * 100).toFixed(1)}%
        </div>
      )}
    </motion.div>
  );
});

OpportunityRow.displayName = 'OpportunityRow';

export const MEVOpportunitiesPanel: React.FC = () => {
  const { opportunities, addOpportunity, removeOpportunity } = useMEVStore();
  const [filter, setFilter] = useState<OpportunityFilter>({
    type: [],
    minProfit: 0,
    maxGas: 1000,
    minConfidence: 0
  });
  const [sortBy, setSortBy] = useState<'profit' | 'confidence' | 'time'>('profit');
  const [executingIds, setExecutingIds] = useState<Set<string>>(new Set());
  const [autoExecute, setAutoExecute] = useState(false);
  const [stats, setStats] = useState<OpportunityStats>({
    total: 0,
    arbitrage: 0,
    liquidation: 0,
    sandwich: 0,
    jit: 0,
    totalProfit: 0,
    avgProfit: 0,
    successRate: 0
  });

  // Load opportunities from API
  useEffect(() => {
    const loadOpportunities = async () => {
      try {
        const opps = await apiService.getMEVOpportunities(100);
        opps.forEach(opp => addOpportunity(opp));
      } catch (error) {
        console.error('Failed to load opportunities:', error);
      }
    };

    loadOpportunities();
    const interval = setInterval(loadOpportunities, 5000);
    return () => clearInterval(interval);
  }, [addOpportunity]);

  // WebSocket real-time updates
  useEffect(() => {
    const handleOpportunity = (data: any) => {
      if (data.type === 'new_opportunity') {
        addOpportunity(data.opportunity);
        
        // Auto-execute if enabled and meets criteria
        if (autoExecute && 
            data.opportunity.expected_profit > filter.minProfit &&
            data.opportunity.confidence > filter.minConfidence) {
          executeOpportunity(data.opportunity.id);
        }
      } else if (data.type === 'opportunity_expired') {
        removeOpportunity(data.id);
      }
    };

    const unsubscribe = mevWebSocket.subscribe('mev_opportunity', handleOpportunity);
    return unsubscribe;
  }, [addOpportunity, removeOpportunity, autoExecute, filter]);

  // Filter and sort opportunities
  const filteredOpportunities = useMemo(() => {
    let filtered = Array.from(opportunities.values());

    // Apply filters
    if (filter.type.length > 0) {
      filtered = filtered.filter(opp => filter.type.includes(opp.type));
    }
    if (filter.minProfit > 0) {
      filtered = filtered.filter(opp => opp.expected_profit >= filter.minProfit);
    }
    if (filter.maxGas < 1000) {
      filtered = filtered.filter(opp => opp.gas_estimate <= filter.maxGas);
    }
    if (filter.minConfidence > 0) {
      filtered = filtered.filter(opp => opp.confidence >= filter.minConfidence);
    }

    // Sort
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'profit':
          return b.expected_profit - a.expected_profit;
        case 'confidence':
          return b.confidence - a.confidence;
        case 'time':
          return b.timestamp - a.timestamp;
        default:
          return 0;
      }
    });

    return filtered;
  }, [opportunities, filter, sortBy]);

  // Calculate statistics
  useEffect(() => {
    const opps = Array.from(opportunities.values());
    const newStats: OpportunityStats = {
      total: opps.length,
      arbitrage: opps.filter(o => o.type === 'arbitrage').length,
      liquidation: opps.filter(o => o.type === 'liquidation').length,
      sandwich: opps.filter(o => o.type === 'sandwich').length,
      jit: opps.filter(o => o.type === 'jit').length,
      totalProfit: opps.reduce((sum, o) => sum + o.expected_profit, 0),
      avgProfit: opps.length > 0 ? opps.reduce((sum, o) => sum + o.expected_profit, 0) / opps.length : 0,
      successRate: 0 // Would need execution history
    };
    setStats(newStats);
  }, [opportunities]);

  // Execute opportunity
  const executeOpportunity = useCallback(async (id: string) => {
    const opportunity = opportunities.get(id);
    if (!opportunity || executingIds.has(id)) return;

    setExecutingIds(prev => new Set(prev).add(id));

    try {
      // Prepare bundle
      const bundle = {
        bundle_id: `bundle-${Date.now()}`,
        transactions: [], // Would be populated with actual transactions
        tip_amount: opportunity.expected_profit * 0.1, // 10% tip
        relay_url: 'https://mainnet.block-engine.jito.wtf'
      };

      const result = await apiService.submitJitoBundle(bundle);
      
      if (result.landed) {
        toast.success(`Opportunity executed! Profit: $${opportunity.expected_profit.toFixed(2)}`);
        removeOpportunity(id);
      } else {
        toast.error('Bundle failed to land');
      }
    } catch (error: any) {
      toast.error(`Execution failed: ${error.message}`);
    } finally {
      setExecutingIds(prev => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }
  }, [opportunities, executingIds, removeOpportunity]);

  // Row renderer for virtual list
  const Row = useCallback(({ index, style }: { index: number; style: React.CSSProperties }) => {
    const opportunity = filteredOpportunities[index];
    if (!opportunity) return null;

    return (
      <div style={style}>
        <OpportunityRow
          opportunity={opportunity}
          onExecute={executeOpportunity}
          isExecuting={executingIds.has(opportunity.id)}
        />
      </div>
    );
  }, [filteredOpportunities, executeOpportunity, executingIds]);

  return (
    <div className="w-full bg-black/90 backdrop-blur-xl rounded-lg border border-cyan-500/20 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <h2 className="text-2xl font-bold text-cyan-400">MEV Opportunities</h2>
          
          {/* Stats */}
          <div className="flex items-center gap-3 text-sm">
            <div className="px-3 py-1 bg-gray-900/50 rounded">
              <span className="text-gray-400">Total:</span>
              <span className="ml-1 font-bold text-cyan-400">{stats.total}</span>
            </div>
            <div className="px-3 py-1 bg-gray-900/50 rounded">
              <span className="text-gray-400">Profit:</span>
              <span className="ml-1 font-bold text-green-400">${stats.totalProfit.toFixed(2)}</span>
            </div>
            <div className="px-3 py-1 bg-gray-900/50 rounded">
              <span className="text-gray-400">Avg:</span>
              <span className="ml-1 font-bold text-yellow-400">${stats.avgProfit.toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-4">
          {/* Auto Execute Toggle */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={autoExecute}
              onChange={(e) => setAutoExecute(e.target.checked)}
              className="w-4 h-4 rounded border-gray-600 text-cyan-500 focus:ring-cyan-500 focus:ring-offset-0 bg-gray-900"
            />
            <span className="text-sm text-gray-300">Auto Execute</span>
          </label>

          {/* Sort Dropdown */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-3 py-1 bg-gray-900/50 border border-gray-700 rounded text-sm text-gray-300 focus:outline-none focus:border-cyan-500"
          >
            <option value="profit">Sort by Profit</option>
            <option value="confidence">Sort by Confidence</option>
            <option value="time">Sort by Time</option>
          </select>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4 mb-4 pb-4 border-b border-gray-800">
        {/* Type Filter */}
        <div className="flex items-center gap-2">
          {(['arbitrage', 'liquidation', 'sandwich', 'jit'] as const).map(type => (
            <button
              key={type}
              onClick={() => {
                setFilter(prev => ({
                  ...prev,
                  type: prev.type.includes(type)
                    ? prev.type.filter(t => t !== type)
                    : [...prev.type, type]
                }));
              }}
              className={`px-3 py-1 rounded text-xs font-semibold transition-all ${
                filter.type.includes(type)
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50'
                  : 'bg-gray-900/50 text-gray-400 border border-gray-700 hover:border-cyan-500/30'
              }`}
            >
              {type.toUpperCase()}
            </button>
          ))}
        </div>

        {/* Min Profit Filter */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">Min Profit:</span>
          <input
            type="number"
            value={filter.minProfit}
            onChange={(e) => setFilter(prev => ({ ...prev, minProfit: parseFloat(e.target.value) || 0 }))}
            className="w-20 px-2 py-1 bg-gray-900/50 border border-gray-700 rounded text-sm text-gray-300 focus:outline-none focus:border-cyan-500"
            placeholder="0"
          />
        </div>

        {/* Min Confidence Filter */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">Min Confidence:</span>
          <input
            type="number"
            value={filter.minConfidence * 100}
            onChange={(e) => setFilter(prev => ({ ...prev, minConfidence: (parseFloat(e.target.value) || 0) / 100 }))}
            className="w-16 px-2 py-1 bg-gray-900/50 border border-gray-700 rounded text-sm text-gray-300 focus:outline-none focus:border-cyan-500"
            placeholder="0"
            min="0"
            max="100"
          />
          <span className="text-xs text-gray-400">%</span>
        </div>

        {/* Clear Filters */}
        {(filter.type.length > 0 || filter.minProfit > 0 || filter.minConfidence > 0) && (
          <button
            onClick={() => setFilter({ type: [], minProfit: 0, maxGas: 1000, minConfidence: 0 })}
            className="px-3 py-1 bg-red-500/20 text-red-400 rounded text-xs hover:bg-red-500/30 transition-colors"
          >
            Clear Filters
          </button>
        )}
      </div>

      {/* Opportunities List */}
      <div className="relative">
        {filteredOpportunities.length === 0 ? (
          <div className="flex items-center justify-center h-64 text-gray-500">
            No opportunities matching filters
          </div>
        ) : (
          <List
            height={600}
            itemCount={filteredOpportunities.length}
            itemSize={100}
            width="100%"
            className="scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent"
          >
            {Row}
          </List>
        )}
      </div>

      {/* Type Distribution */}
      <div className="mt-4 pt-4 border-t border-gray-800 flex items-center justify-between">
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
            <span className="text-gray-400">Arbitrage: {stats.arbitrage}</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
            <span className="text-gray-400">Liquidation: {stats.liquidation}</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
            <span className="text-gray-400">Sandwich: {stats.sandwich}</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span className="text-gray-400">JIT: {stats.jit}</span>
          </div>
        </div>

        <div className="text-xs text-gray-500">
          Updated: {new Date().toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
};