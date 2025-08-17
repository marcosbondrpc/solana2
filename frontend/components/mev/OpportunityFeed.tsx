'use client';

import { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { useSnapshot } from 'valtio';
import { mevStore } from '../../stores/mevStore';
import { FixedSizeList as List } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';

interface Opportunity {
  id: string;
  timestamp: number;
  kind: 'arbitrage' | 'sandwich';
  route: 'direct' | 'jito_main' | 'jito_alt';
  expected_profit: number;
  confidence: number;
  risk_score: number;
  tip: number;
  status?: 'pending' | 'executing' | 'success' | 'failed';
}

export default function OpportunityFeed() {
  const store = useSnapshot(mevStore);
  const [isPaused, setIsPaused] = useState(false);
  const [filter, setFilter] = useState<'all' | 'arbitrage' | 'sandwich'>('all');
  const listRef = useRef<List>(null);
  const [hoveredRow, setHoveredRow] = useState<number | null>(null);

  // Convert transactions to opportunities
  const opportunities = useMemo(() => {
    const txs = store.transactions.map((tx, idx) => ({
      id: tx.transaction_hash || `opp-${idx}`,
      timestamp: new Date(tx.timestamp).getTime(),
      kind: tx.transaction_type === 'sandwich' ? 'sandwich' : 'arbitrage' as any,
      route: tx.route || 'direct' as any,
      expected_profit: tx.profit_amount || 0,
      confidence: tx.confidence_score || 0,
      risk_score: tx.risk_score || 0,
      tip: tx.tip_amount || 0,
      status: tx.success ? 'success' : tx.success === false ? 'failed' : 'pending' as any,
    }));

    if (filter === 'all') return txs;
    return txs.filter(opp => opp.kind === filter);
  }, [store.transactions, filter]);

  // Auto-scroll to bottom when new items arrive
  useEffect(() => {
    if (!isPaused && store.settings.autoScroll && listRef.current && opportunities.length > 0) {
      listRef.current.scrollToItem(opportunities.length - 1, 'end');
    }
  }, [opportunities.length, isPaused, store.settings.autoScroll]);

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toTimeString().slice(0, 8) + '.' + date.getMilliseconds().toString().padStart(3, '0');
  };

  const formatProfit = (profit: number) => {
    if (profit >= 1000) return `$${(profit / 1000).toFixed(1)}k`;
    if (profit >= 100) return `$${profit.toFixed(0)}`;
    return `$${profit.toFixed(2)}`;
  };

  const getOpportunityColor = (opp: Opportunity) => {
    if (opp.status === 'success') return 'text-green-400';
    if (opp.status === 'failed') return 'text-red-400';
    if (opp.kind === 'sandwich') return 'text-yellow-400';
    if (opp.kind === 'arbitrage') return 'text-blue-400';
    return 'text-zinc-400';
  };

  const getRouteColor = (route: string) => {
    switch (route) {
      case 'direct': return 'text-purple-400';
      case 'jito_main': return 'text-cyan-400';
      case 'jito_alt': return 'text-pink-400';
      default: return 'text-zinc-400';
    }
  };

  const getRiskColor = (risk: number) => {
    if (risk < 0.3) return 'text-green-400';
    if (risk < 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const Row = useCallback(({ index, style }: { index: number; style: React.CSSProperties }) => {
    const opp = opportunities[index];
    if (!opp) return null;

    const isHovered = hoveredRow === index;
    const rowClass = `px-2 py-1 font-mono text-xs flex items-center gap-2 transition-colors ${
      isHovered ? 'bg-zinc-900' : ''
    } ${opp.status === 'executing' ? 'animate-pulse' : ''}`;

    return (
      <div 
        style={style} 
        className={rowClass}
        onMouseEnter={() => setHoveredRow(index)}
        onMouseLeave={() => setHoveredRow(null)}
      >
        <span className="text-zinc-600 w-24">{formatTime(opp.timestamp)}</span>
        <span className={`w-16 ${getOpportunityColor(opp)}`}>
          {opp.kind.toUpperCase().slice(0, 3)}
        </span>
        <span className={`w-20 ${getRouteColor(opp.route)}`}>
          {opp.route.replace('_', ' ').toUpperCase()}
        </span>
        <span className="w-16 text-green-400 text-right">{formatProfit(opp.expected_profit)}</span>
        <span className={`w-12 text-right ${getRiskColor(opp.risk_score)}`}>
          {(opp.risk_score * 100).toFixed(0)}%
        </span>
        <span className="w-12 text-amber-400 text-right">{opp.tip.toFixed(1)}</span>
        <span className="w-12 text-cyan-400 text-right">{(opp.confidence * 100).toFixed(0)}%</span>
        {opp.status && (
          <span className={`ml-auto text-[10px] px-1 rounded ${
            opp.status === 'success' ? 'bg-green-900 text-green-300' :
            opp.status === 'failed' ? 'bg-red-900 text-red-300' :
            opp.status === 'executing' ? 'bg-yellow-900 text-yellow-300' :
            'bg-zinc-800 text-zinc-400'
          }`}>
            {opp.status.toUpperCase()}
          </span>
        )}
      </div>
    );
  }, [opportunities, hoveredRow]);

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-zinc-400">OPPORTUNITY FEED</h3>
        <div className="flex items-center gap-2">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as any)}
            className="bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs"
          >
            <option value="all">All</option>
            <option value="arbitrage">Arbitrage</option>
            <option value="sandwich">Sandwich</option>
          </select>
          <button
            onClick={() => setIsPaused(!isPaused)}
            className={`px-2 py-1 text-xs rounded transition-colors ${
              isPaused 
                ? 'bg-yellow-900 text-yellow-400 border border-yellow-800' 
                : 'bg-zinc-900 text-zinc-400 border border-zinc-800'
            }`}
          >
            {isPaused ? 'PAUSED' : 'LIVE'}
          </button>
        </div>
      </div>

      {/* Column Headers */}
      <div className="px-2 py-1 border-b border-zinc-800 font-mono text-[10px] text-zinc-600 flex items-center gap-2">
        <span className="w-24">TIME</span>
        <span className="w-16">TYPE</span>
        <span className="w-20">ROUTE</span>
        <span className="w-16 text-right">PROFIT</span>
        <span className="w-12 text-right">RISK</span>
        <span className="w-12 text-right">TIP</span>
        <span className="w-12 text-right">CONF</span>
      </div>

      {/* Virtual List */}
      <div className="flex-1 min-h-0">
        <AutoSizer>
          {({ height, width }) => (
            <List
              ref={listRef}
              height={height}
              itemCount={opportunities.length}
              itemSize={24}
              width={width}
              overscanCount={10}
            >
              {Row}
            </List>
          )}
        </AutoSizer>
      </div>

      {/* Footer Stats */}
      <div className="pt-2 border-t border-zinc-800 flex items-center justify-between text-[10px] text-zinc-600">
        <span>{opportunities.length} opportunities</span>
        <span>{store.activeOpportunities} active</span>
        <span className="text-green-400">
          ${(store.totalProfit / 1000).toFixed(1)}k total
        </span>
      </div>
    </div>
  );
}