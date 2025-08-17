'use client';

import React, { useEffect, useRef, useState, useMemo } from 'react';
import { useMEVStore, selectActiveOpportunities, ArbitrageOpportunity } from '@/stores/mev-store';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle, 
  XCircle,
  Activity,
  DollarSign,
  Zap,
  Target,
  GitBranch
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { useVirtualizer } from '@tanstack/react-virtual';

interface OpportunityCardProps {
  opportunity: ArbitrageOpportunity;
  onExecute: (id: string) => void;
}

const OpportunityCard: React.FC<OpportunityCardProps> = React.memo(({ opportunity, onExecute }) => {
  const profitColor = opportunity.expectedProfit > 0.1 ? 'text-green-500' : 
                      opportunity.expectedProfit > 0.05 ? 'text-yellow-500' : 'text-gray-500';
  
  const statusIcon = {
    pending: <Activity className="w-4 h-4 text-blue-500" />,
    executing: <Zap className="w-4 h-4 text-yellow-500 animate-pulse" />,
    completed: <CheckCircle className="w-4 h-4 text-green-500" />,
    failed: <XCircle className="w-4 h-4 text-red-500" />
  }[opportunity.status];
  
  const confidenceColor = opportunity.confidence > 0.8 ? 'bg-green-500' :
                          opportunity.confidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500';
  
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      transition={{ duration: 0.2 }}
      className="relative"
    >
      <Card className="p-4 hover:shadow-lg transition-all duration-200 border-l-4 border-l-blue-500">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              {statusIcon}
              <span className="text-sm font-medium">
                {opportunity.dexA} → {opportunity.dexB}
              </span>
              <Badge variant={opportunity.status === 'completed' ? 'default' : 'outline'}>
                {opportunity.status}
              </Badge>
              {opportunity.latency && (
                <span className="text-xs text-gray-500">
                  {opportunity.latency.toFixed(2)}ms
                </span>
              )}
            </div>
            
            <div className="grid grid-cols-2 gap-4 mb-3">
              <div>
                <div className="text-xs text-gray-500">Path</div>
                <div className="flex items-center gap-1 text-sm">
                  <GitBranch className="w-3 h-3" />
                  {opportunity.path.slice(0, 3).join(' → ')}
                  {opportunity.path.length > 3 && '...'}
                </div>
              </div>
              
              <div>
                <div className="text-xs text-gray-500">Tokens</div>
                <div className="text-sm">
                  {opportunity.tokenIn} → {opportunity.tokenOut}
                </div>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div>
                  <div className="text-xs text-gray-500">Expected Profit</div>
                  <div className={`text-lg font-bold ${profitColor}`}>
                    ${opportunity.expectedProfit.toFixed(4)}
                  </div>
                </div>
                
                {opportunity.actualProfit !== undefined && (
                  <div>
                    <div className="text-xs text-gray-500">Actual Profit</div>
                    <div className="text-lg font-bold text-green-500">
                      ${opportunity.actualProfit.toFixed(4)}
                    </div>
                  </div>
                )}
                
                <div>
                  <div className="text-xs text-gray-500">Amount</div>
                  <div className="text-sm">
                    ${opportunity.amountIn.toFixed(2)}
                  </div>
                </div>
                
                <div>
                  <div className="text-xs text-gray-500">Slippage</div>
                  <div className="text-sm">
                    {(opportunity.slippage * 100).toFixed(2)}%
                  </div>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <div className="flex flex-col items-center">
                  <div className="text-xs text-gray-500">Confidence</div>
                  <div className="w-20 h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div 
                      className={`h-full ${confidenceColor} transition-all duration-300`}
                      style={{ width: `${opportunity.confidence * 100}%` }}
                    />
                  </div>
                  <div className="text-xs mt-1">
                    {(opportunity.confidence * 100).toFixed(0)}%
                  </div>
                </div>
                
                {opportunity.status === 'pending' && (
                  <Button 
                    size="sm" 
                    onClick={() => onExecute(opportunity.id)}
                    className="ml-2"
                  >
                    Execute
                  </Button>
                )}
              </div>
            </div>
            
            {opportunity.txHash && (
              <div className="mt-2 pt-2 border-t">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500">Transaction:</span>
                  <a 
                    href={`https://solscan.io/tx/${opportunity.txHash}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-500 hover:underline font-mono"
                  >
                    {opportunity.txHash.slice(0, 8)}...{opportunity.txHash.slice(-8)}
                  </a>
                </div>
                {opportunity.gasUsed && (
                  <div className="flex items-center justify-between text-xs mt-1">
                    <span className="text-gray-500">Gas Used:</span>
                    <span>${opportunity.gasUsed.toFixed(4)}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
        
        <div className="absolute top-2 right-2 text-xs text-gray-500">
          {formatDistanceToNow(opportunity.timestamp, { addSuffix: true })}
        </div>
      </Card>
    </motion.div>
  );
});

OpportunityCard.displayName = 'OpportunityCard';

export const ArbitrageScanner: React.FC = () => {
  const opportunities = useMEVStore(selectActiveOpportunities);
  const { 
    totalOpportunities, 
    executedOpportunities, 
    successfulTrades,
    failedTrades,
    autoExecute,
    minProfitThreshold,
    updateArbitrageStatus,
    setConfiguration
  } = useMEVStore();
  
  const parentRef = useRef<HTMLDivElement>(null);
  const [sortBy, setSortBy] = useState<'profit' | 'confidence' | 'time'>('profit');
  const [filterDex, setFilterDex] = useState<string>('all');
  
  const sortedOpportunities = useMemo(() => {
    let sorted = [...opportunities];
    
    if (filterDex !== 'all') {
      sorted = sorted.filter(o => o.dexA === filterDex || o.dexB === filterDex);
    }
    
    switch (sortBy) {
      case 'profit':
        sorted.sort((a, b) => b.expectedProfit - a.expectedProfit);
        break;
      case 'confidence':
        sorted.sort((a, b) => b.confidence - a.confidence);
        break;
      case 'time':
        sorted.sort((a, b) => b.timestamp - a.timestamp);
        break;
    }
    
    return sorted;
  }, [opportunities, sortBy, filterDex]);
  
  const virtualizer = useVirtualizer({
    count: sortedOpportunities.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 180,
    overscan: 5,
  });
  
  const handleExecute = (id: string) => {
    updateArbitrageStatus(id, 'executing');
    // Actual execution logic would go here
  };
  
  const successRate = totalOpportunities > 0 
    ? ((successfulTrades / totalOpportunities) * 100).toFixed(1)
    : '0.0';
  
  const stats = [
    { label: 'Total Opportunities', value: totalOpportunities, icon: Target, color: 'text-blue-500' },
    { label: 'Executed', value: executedOpportunities, icon: Activity, color: 'text-yellow-500' },
    { label: 'Successful', value: successfulTrades, icon: CheckCircle, color: 'text-green-500' },
    { label: 'Failed', value: failedTrades, icon: XCircle, color: 'text-red-500' },
  ];
  
  return (
    <div className="space-y-4">
      {/* Header Stats */}
      <div className="grid grid-cols-4 gap-4">
        {stats.map((stat) => (
          <Card key={stat.label} className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-500">{stat.label}</div>
                <div className="text-2xl font-bold">{stat.value.toLocaleString()}</div>
              </div>
              <stat.icon className={`w-8 h-8 ${stat.color}`} />
            </div>
          </Card>
        ))}
      </div>
      
      {/* Success Rate Bar */}
      <Card className="p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium">Success Rate</span>
          <span className="text-2xl font-bold">{successRate}%</span>
        </div>
        <Progress value={parseFloat(successRate)} className="h-3" />
      </Card>
      
      {/* Controls */}
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">Auto Execute:</label>
              <Button
                size="sm"
                variant={autoExecute ? 'default' : 'outline'}
                onClick={() => setConfiguration({ autoExecute: !autoExecute })}
              >
                {autoExecute ? 'ON' : 'OFF'}
              </Button>
            </div>
            
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">Min Profit:</label>
              <input
                type="number"
                value={minProfitThreshold}
                onChange={(e) => setConfiguration({ minProfitThreshold: parseFloat(e.target.value) })}
                className="w-20 px-2 py-1 text-sm border rounded"
                step="0.001"
              />
            </div>
            
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">Sort:</label>
              <select 
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="px-2 py-1 text-sm border rounded"
              >
                <option value="profit">Profit</option>
                <option value="confidence">Confidence</option>
                <option value="time">Recent</option>
              </select>
            </div>
            
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">DEX:</label>
              <select 
                value={filterDex}
                onChange={(e) => setFilterDex(e.target.value)}
                className="px-2 py-1 text-sm border rounded"
              >
                <option value="all">All</option>
                <option value="raydium">Raydium</option>
                <option value="orca">Orca</option>
                <option value="phoenix">Phoenix</option>
                <option value="meteora">Meteora</option>
              </select>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-500">
              Showing {sortedOpportunities.length} active opportunities
            </span>
          </div>
        </div>
      </Card>
      
      {/* Opportunities List */}
      <Card className="p-4">
        <div className="mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            Active Arbitrage Opportunities
          </h3>
        </div>
        
        <div 
          ref={parentRef}
          className="h-[600px] overflow-auto space-y-2"
        >
          <div
            style={{
              height: `${virtualizer.getTotalSize()}px`,
              width: '100%',
              position: 'relative',
            }}
          >
            <AnimatePresence>
              {virtualizer.getVirtualItems().map((virtualItem) => {
                const opportunity = sortedOpportunities[virtualItem.index];
                return (
                  <div
                    key={virtualItem.key}
                    style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: `${virtualItem.size}px`,
                      transform: `translateY(${virtualItem.start}px)`,
                    }}
                  >
                    <OpportunityCard
                      opportunity={opportunity}
                      onExecute={handleExecute}
                    />
                  </div>
                );
              })}
            </AnimatePresence>
          </div>
        </div>
      </Card>
    </div>
  );
};