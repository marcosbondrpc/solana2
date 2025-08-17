'use client';

import { memo, useEffect, useRef, useMemo, useCallback } from 'react';
import { FixedSizeList as List } from 'react-window';
import { motion, AnimatePresence } from 'framer-motion';
import { snapshot } from 'valtio';
import { mevStore } from '@/stores/mevStore';

interface SandwichAttack {
  id: string;
  timestamp: number;
  victimTx: string;
  frontrunTx: string;
  backrunTx: string;
  confidence: number;
  profitUSD: number;
  victimLoss: number;
  attackerAddress: string;
  dexPair: string;
  blockNumber: number;
  gasUsed: number;
  status: 'detected' | 'confirmed' | 'failed';
}

// Virtual row renderer with extreme optimization
const Row = memo(({ index, style, data }: any) => {
  const item = data[index];
  
  const confidenceColor = useMemo(() => {
    if (item.confidence >= 0.9) return 'text-red-500';
    if (item.confidence >= 0.7) return 'text-yellow-500';
    return 'text-gray-500';
  }, [item.confidence]);
  
  const profitColor = useMemo(() => {
    if (item.profitUSD >= 1000) return 'text-green-400 font-bold';
    if (item.profitUSD >= 100) return 'text-green-500';
    return 'text-gray-400';
  }, [item.profitUSD]);
  
  return (
    <div 
      style={style} 
      className="flex items-center px-4 py-2 border-b border-gray-800 hover:bg-gray-900/50 transition-colors duration-75"
    >
      {/* Status Indicator */}
      <div className="w-2 h-2 mr-3">
        <div 
          className={`w-full h-full rounded-full ${
            item.status === 'confirmed' ? 'bg-red-500' : 
            item.status === 'detected' ? 'bg-yellow-500 animate-pulse' : 
            'bg-gray-600'
          }`}
        />
      </div>
      
      {/* Timestamp */}
      <div className="w-20 text-xs text-gray-500">
        {new Date(item.timestamp).toLocaleTimeString('en-US', { 
          hour12: false,
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
          fractionalSecondDigits: 3
        })}
      </div>
      
      {/* Block */}
      <div className="w-24 text-xs font-mono text-gray-400 ml-2">
        #{item.blockNumber}
      </div>
      
      {/* DEX Pair */}
      <div className="w-32 text-sm font-medium truncate ml-2">
        {item.dexPair}
      </div>
      
      {/* Confidence Score */}
      <div className={`w-20 text-sm font-mono ml-2 ${confidenceColor}`}>
        {(item.confidence * 100).toFixed(1)}%
      </div>
      
      {/* Profit */}
      <div className={`w-24 text-sm font-mono ml-2 ${profitColor}`}>
        ${item.profitUSD.toFixed(2)}
      </div>
      
      {/* Victim Loss */}
      <div className="w-24 text-sm font-mono text-red-400 ml-2">
        -${item.victimLoss.toFixed(2)}
      </div>
      
      {/* Attacker */}
      <div className="flex-1 text-xs font-mono text-gray-500 ml-2 truncate">
        {item.attackerAddress}
      </div>
      
      {/* Transactions */}
      <div className="flex gap-1 ml-2">
        <div 
          className="w-2 h-2 bg-blue-500 rounded-full" 
          title={`Front: ${item.frontrunTx.slice(0, 8)}`}
        />
        <div 
          className="w-2 h-2 bg-purple-500 rounded-full" 
          title={`Victim: ${item.victimTx.slice(0, 8)}`}
        />
        <div 
          className="w-2 h-2 bg-green-500 rounded-full" 
          title={`Back: ${item.backrunTx.slice(0, 8)}`}
        />
      </div>
    </div>
  );
});

Row.displayName = 'SandwichRow';

export const SandwichDetectionFeed = memo(() => {
  const listRef = useRef<List>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [attacks, setAttacks] = useState<SandwichAttack[]>([]);
  const [autoScroll, setAutoScroll] = useState(true);
  
  // Generate mock data for demonstration
  useEffect(() => {
    const generateMockAttack = (): SandwichAttack => ({
      id: Math.random().toString(36).substr(2, 9),
      timestamp: Date.now(),
      victimTx: '0x' + Math.random().toString(36).substr(2, 64),
      frontrunTx: '0x' + Math.random().toString(36).substr(2, 64),
      backrunTx: '0x' + Math.random().toString(36).substr(2, 64),
      confidence: 0.5 + Math.random() * 0.5,
      profitUSD: Math.random() * 5000,
      victimLoss: Math.random() * 2000,
      attackerAddress: '0x' + Math.random().toString(36).substr(2, 40),
      dexPair: ['USDC/SOL', 'RAY/USDT', 'BONK/SOL', 'JTO/USDC'][Math.floor(Math.random() * 4)],
      blockNumber: Math.floor(250000000 + Math.random() * 1000),
      gasUsed: Math.floor(100000 + Math.random() * 500000),
      status: Math.random() > 0.3 ? 'confirmed' : 'detected'
    });
    
    // Initial batch
    const initialAttacks = Array.from({ length: 100 }, generateMockAttack);
    setAttacks(initialAttacks);
    
    // Real-time updates
    const interval = setInterval(() => {
      setAttacks(prev => {
        const newAttack = generateMockAttack();
        const updated = [newAttack, ...prev];
        return updated.slice(0, 10000); // Keep max 10k items
      });
      
      if (autoScroll && listRef.current) {
        listRef.current.scrollToItem(0, 'start');
      }
    }, 500 + Math.random() * 1500);
    
    return () => clearInterval(interval);
  }, [autoScroll]);
  
  // Stats calculation with memoization
  const stats = useMemo(() => {
    const last100 = attacks.slice(0, 100);
    const totalProfit = last100.reduce((sum, a) => sum + a.profitUSD, 0);
    const totalLoss = last100.reduce((sum, a) => sum + a.victimLoss, 0);
    const avgConfidence = last100.reduce((sum, a) => sum + a.confidence, 0) / (last100.length || 1);
    const confirmedCount = last100.filter(a => a.status === 'confirmed').length;
    
    return {
      totalProfit,
      totalLoss,
      avgConfidence,
      confirmedRate: (confirmedCount / (last100.length || 1)) * 100
    };
  }, [attacks]);
  
  return (
    <div className="h-full flex flex-col">
      {/* Header with Stats */}
      <div className="flex justify-between items-center mb-4">
        <div className="flex gap-6">
          <div>
            <div className="text-xs text-gray-500">Total Extracted</div>
            <div className="text-lg font-bold text-green-400">
              ${stats.totalProfit.toFixed(0)}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Victim Losses</div>
            <div className="text-lg font-bold text-red-400">
              ${stats.totalLoss.toFixed(0)}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Avg Confidence</div>
            <div className="text-lg font-bold text-yellow-400">
              {(stats.avgConfidence * 100).toFixed(1)}%
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Confirmed Rate</div>
            <div className="text-lg font-bold text-purple-400">
              {stats.confirmedRate.toFixed(1)}%
            </div>
          </div>
        </div>
        
        <button
          onClick={() => setAutoScroll(!autoScroll)}
          className={`px-3 py-1 text-xs rounded ${
            autoScroll 
              ? 'bg-blue-600 text-white' 
              : 'bg-gray-700 text-gray-400'
          } transition-colors`}
        >
          Auto-Scroll: {autoScroll ? 'ON' : 'OFF'}
        </button>
      </div>
      
      {/* Column Headers */}
      <div className="flex items-center px-4 py-2 border-b border-gray-700 text-xs text-gray-500 font-medium">
        <div className="w-2 mr-3"></div>
        <div className="w-20">Time</div>
        <div className="w-24 ml-2">Block</div>
        <div className="w-32 ml-2">DEX Pair</div>
        <div className="w-20 ml-2">Confidence</div>
        <div className="w-24 ml-2">Profit</div>
        <div className="w-24 ml-2">Loss</div>
        <div className="flex-1 ml-2">Attacker</div>
        <div className="w-12 ml-2">TXs</div>
      </div>
      
      {/* Virtual List */}
      <div ref={containerRef} className="flex-1 bg-gray-950">
        <List
          ref={listRef}
          height={600}
          itemCount={attacks.length}
          itemSize={40}
          width="100%"
          itemData={attacks}
          overscanCount={10}
        >
          {Row}
        </List>
      </div>
      
      {/* Footer */}
      <div className="mt-2 flex justify-between items-center text-xs text-gray-500">
        <div>Showing {attacks.length} attacks</div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 bg-blue-500 rounded-full" />
            <span>Frontrun</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 bg-purple-500 rounded-full" />
            <span>Victim</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 bg-green-500 rounded-full" />
            <span>Backrun</span>
          </div>
        </div>
      </div>
    </div>
  );
});

SandwichDetectionFeed.displayName = 'SandwichDetectionFeed';

// Missing import fix
import { useState } from 'react';