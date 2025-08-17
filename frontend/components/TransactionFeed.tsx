'use client';

import { useSnapshot } from 'valtio';
import { mevStore } from '../stores/mevStore';
import { FixedSizeList as List } from 'react-window';
import { memo, useCallback, useRef, useEffect } from 'react';
import { clsx } from 'clsx';
import numeral from 'numeral';
import { motion, AnimatePresence } from 'framer-motion';
import { format } from 'date-fns';
import { MEVTransaction } from '../lib/clickhouse';

// Memoized transaction row component
const TransactionRow = memo(({ 
  index, 
  style, 
  data 
}: { 
  index: number; 
  style: React.CSSProperties; 
  data: MEVTransaction[] 
}) => {
  const tx = data[index];
  
  if (!tx) return null;
  
  const isProfit = tx.profit_amount > 0;
  const latencyStatus = 
    tx.latency_ms <= 8 ? 'excellent' : 
    tx.latency_ms <= 20 ? 'good' : 'poor';
  
  return (
    <div style={style} className="px-4">
      <motion.div 
        className={clsx(
          "flex items-center gap-4 p-3 rounded-lg border transition-all duration-200",
          "hover:bg-zinc-900/50",
          tx.bundle_landed 
            ? "border-green-500/20 bg-green-500/5" 
            : "border-red-500/20 bg-red-500/5"
        )}
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.2, delay: index * 0.02 }}
      >
        {/* Status Indicator */}
        <div className="flex flex-col items-center">
          <div className={clsx(
            "w-2 h-2 rounded-full",
            tx.bundle_landed ? "bg-green-500" : "bg-red-500"
          )} />
          <span className="text-xs text-zinc-600 mt-1">
            {format(new Date(tx.timestamp), 'HH:mm:ss')}
          </span>
        </div>
        
        {/* Transaction Type */}
        <div className="w-20">
          <span className={clsx(
            "text-xs px-2 py-0.5 rounded-full uppercase",
            tx.transaction_type === 'arbitrage' && "bg-blue-500/10 text-blue-400",
            tx.transaction_type === 'liquidation' && "bg-orange-500/10 text-orange-400",
            tx.transaction_type === 'sandwich' && "bg-purple-500/10 text-purple-400",
            tx.transaction_type === 'jit' && "bg-cyan-500/10 text-cyan-400"
          )}>
            {tx.transaction_type}
          </span>
        </div>
        
        {/* Signature */}
        <div className="flex-1">
          <code className="text-xs text-zinc-500">
            {tx.transaction_signature.slice(0, 8)}...{tx.transaction_signature.slice(-8)}
          </code>
        </div>
        
        {/* Route */}
        <div className="hidden lg:block flex-1">
          <code className="text-xs text-zinc-600 truncate block">
            {tx.route_path.join(' → ')}
          </code>
        </div>
        
        {/* Profit */}
        <div className="text-right">
          <div className={clsx(
            "text-sm font-medium",
            isProfit ? "text-green-400" : "text-red-400"
          )}>
            {isProfit ? '+' : ''}{numeral(tx.profit_amount).format('$0,0.000')}
          </div>
          <div className="text-xs text-zinc-600">
            {tx.profit_percentage.toFixed(2)}%
          </div>
        </div>
        
        {/* Latency */}
        <div className="text-right">
          <div className={clsx(
            "text-sm font-medium",
            `latency-${latencyStatus}`
          )}>
            {tx.latency_ms.toFixed(1)}ms
          </div>
          <div className="text-xs text-zinc-600">
            Gas: {numeral(tx.gas_used).format('0,0')}
          </div>
        </div>
        
        {/* DNA Fingerprint */}
        <div className="hidden xl:block">
          <code className="text-xs text-zinc-700" title={tx.decision_dna}>
            {tx.decision_dna.slice(0, 6)}...
          </code>
        </div>
      </motion.div>
    </div>
  );
});

TransactionRow.displayName = 'TransactionRow';

export default function TransactionFeed() {
  const snap = useSnapshot(mevStore);
  const listRef = useRef<List>(null);
  const autoScrollRef = useRef(true);
  
  // Auto-scroll to new transactions
  useEffect(() => {
    if (autoScrollRef.current && snap.transactions.length > 0 && listRef.current) {
      listRef.current.scrollToItem(0, 'start');
    }
  }, [snap.transactions.length]);
  
  const handleScroll = useCallback(({ scrollOffset }: { scrollOffset: number }) => {
    // Disable auto-scroll if user scrolls up
    autoScrollRef.current = scrollOffset === 0;
  }, []);
  
  const toggleAutoScroll = useCallback(() => {
    autoScrollRef.current = !autoScrollRef.current;
    if (autoScrollRef.current && listRef.current) {
      listRef.current.scrollToItem(0, 'start');
    }
  }, []);
  
  return (
    <div className="glass rounded-xl overflow-hidden">
      <div className="flex items-center justify-between p-4 border-b border-zinc-800">
        <div className="flex items-center gap-4">
          <h2 className="text-lg font-semibold">Transaction Feed</h2>
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">
              {snap.transactions.length} transactions
            </span>
            {snap.transactionBuffer.length > 0 && (
              <span className="text-xs text-yellow-400">
                +{snap.transactionBuffer.length} pending
              </span>
            )}
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={toggleAutoScroll}
            className={clsx(
              "btn btn-ghost text-xs",
              autoScrollRef.current && "bg-blue-500/10 text-blue-400"
            )}
          >
            {autoScrollRef.current ? 'Auto-Scroll ON' : 'Auto-Scroll OFF'}
          </button>
        </div>
      </div>
      
      <div className="h-96">
        {snap.transactions.length > 0 ? (
          <List
            ref={listRef}
            height={384}
            itemCount={snap.transactions.length}
            itemSize={80}
            width="100%"
            itemData={snap.transactions}
            onScroll={handleScroll}
            className="scrollbar-thin"
          >
            {TransactionRow}
          </List>
        ) : (
          <div className="flex items-center justify-center h-full text-zinc-600">
            <div className="text-center">
              <div className="text-4xl mb-2">📊</div>
              <div className="text-sm">Waiting for transactions...</div>
            </div>
          </div>
        )}
      </div>
      
      {/* Live indicator */}
      <div className="px-4 py-2 border-t border-zinc-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={clsx(
            "w-2 h-2 rounded-full",
            snap.wsStatus === 'connected' ? "bg-green-500 animate-pulse" : "bg-red-500"
          )} />
          <span className="text-xs text-zinc-500">
            {snap.wsStatus === 'connected' ? 'Live' : 'Disconnected'}
          </span>
        </div>
        
        {snap.lastTransactionTime > 0 && (
          <span className="text-xs text-zinc-600">
            Last update: {format(new Date(snap.lastTransactionTime), 'HH:mm:ss')}
          </span>
        )}
      </div>
    </div>
  );
}