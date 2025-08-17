'use client';

import { useEffect, useRef, useState, memo } from 'react';
import { FixedSizeList as List } from 'react-window';
import { Badge } from '@/components/ui/badge';
import { useDetectionFeed } from '@/hooks/useDetectionFeed';
import { formatRelativeTime, formatLamports } from '@/lib/utils';
import { AlertTriangle, TrendingUp, Clock } from 'lucide-react';

interface DetectionItem {
  id: string;
  timestamp: number;
  type: 'sandwich' | 'frontrun' | 'backrun';
  confidence: number;
  victimTx: string;
  attackerTx: string;
  profitLamports: number;
  gasUsed: number;
  blockSlot: number;
  evidence: {
    timeDelta: number;
    priceImpact: number;
    slippage: number;
  };
}

const DetectionRow = memo(({ index, style, data }: any) => {
  const item: DetectionItem = data[index];
  
  const confidenceColor = 
    item.confidence >= 0.9 ? 'text-red-500' :
    item.confidence >= 0.7 ? 'text-orange-500' :
    'text-yellow-500';
  
  return (
    <div style={style} className="px-4 py-2 border-b border-border/50">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <Badge 
              variant={item.type === 'sandwich' ? 'destructive' : 'secondary'}
              className="text-xs"
            >
              {item.type.toUpperCase()}
            </Badge>
            <span className={`text-sm font-medium ${confidenceColor}`}>
              {(item.confidence * 100).toFixed(1)}% confidence
            </span>
            <span className="text-xs text-muted-foreground">
              {formatRelativeTime(item.timestamp)}
            </span>
          </div>
          
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-muted-foreground">Victim: </span>
              <span className="font-mono">{item.victimTx.slice(0, 8)}...</span>
            </div>
            <div>
              <span className="text-muted-foreground">Attacker: </span>
              <span className="font-mono">{item.attackerTx.slice(0, 8)}...</span>
            </div>
          </div>
          
          <div className="flex items-center gap-4 mt-2">
            <div className="flex items-center gap-1">
              <TrendingUp className="h-3 w-3 text-mev-profit" />
              <span className="text-xs font-medium text-mev-profit">
                {formatLamports(item.profitLamports)} SOL
              </span>
            </div>
            <div className="flex items-center gap-1">
              <Clock className="h-3 w-3 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">
                {item.evidence.timeDelta}ms delta
              </span>
            </div>
            <div className="flex items-center gap-1">
              <AlertTriangle className="h-3 w-3 text-yellow-500" />
              <span className="text-xs text-muted-foreground">
                {(item.evidence.priceImpact * 100).toFixed(2)}% impact
              </span>
            </div>
          </div>
        </div>
        
        <div className="text-right">
          <div className="text-xs text-muted-foreground">Block</div>
          <div className="text-sm font-mono">{item.blockSlot.toLocaleString()}</div>
        </div>
      </div>
    </div>
  );
});

DetectionRow.displayName = 'DetectionRow';

export function SandwichDetectionFeed() {
  const listRef = useRef<List>(null);
  const { data: detections, isLoading } = useDetectionFeed();
  const [autoScroll, setAutoScroll] = useState(true);
  
  useEffect(() => {
    if (autoScroll && listRef.current && detections.length > 0) {
      listRef.current.scrollToItem(0, 'start');
    }
  }, [detections, autoScroll]);
  
  if (isLoading) {
    return (
      <div className="h-[600px] flex items-center justify-center">
        <div className="text-muted-foreground">Loading detection feed...</div>
      </div>
    );
  }
  
  return (
    <div className="relative">
      {/* Auto-scroll toggle */}
      <div className="absolute top-2 right-2 z-10">
        <button
          onClick={() => setAutoScroll(!autoScroll)}
          className={`px-3 py-1 text-xs rounded-md transition-colors ${
            autoScroll 
              ? 'bg-primary/20 text-primary' 
              : 'bg-secondary text-secondary-foreground'
          }`}
        >
          {autoScroll ? 'Auto-scroll ON' : 'Auto-scroll OFF'}
        </button>
      </div>
      
      {/* Virtual list */}
      <List
        ref={listRef}
        height={600}
        itemCount={detections.length}
        itemSize={100}
        width="100%"
        itemData={detections}
        className="custom-scrollbar"
      >
        {DetectionRow}
      </List>
      
      {/* Stats footer */}
      <div className="mt-4 p-3 bg-secondary/20 rounded-lg">
        <div className="flex justify-between text-xs">
          <div>
            <span className="text-muted-foreground">Total Detections: </span>
            <span className="font-medium">{detections.length.toLocaleString()}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Avg Confidence: </span>
            <span className="font-medium">
              {(detections.reduce((acc, d) => acc + d.confidence, 0) / detections.length * 100).toFixed(1)}%
            </span>
          </div>
          <div>
            <span className="text-muted-foreground">Total Profit Extracted: </span>
            <span className="font-medium text-mev-profit">
              {formatLamports(detections.reduce((acc, d) => acc + d.profitLamports, 0))} SOL
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}