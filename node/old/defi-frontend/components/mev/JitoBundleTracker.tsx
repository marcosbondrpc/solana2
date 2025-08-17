'use client';

import React, { useState, useMemo, useRef, useEffect } from 'react';
import { useMEVStore, selectRecentBundles, JitoBundle } from '@/stores/mev-store';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import {
  Package,
  CheckCircle,
  XCircle,
  Clock,
  DollarSign,
  TrendingUp,
  Send,
  Layers,
  Cpu,
  BarChart3,
  Timer,
  Gauge
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactECharts from 'echarts-for-react';
import { formatDistanceToNow } from 'date-fns';

interface BundleMetrics {
  totalBundles: number;
  landedBundles: number;
  failedBundles: number;
  pendingBundles: number;
  avgTip: number;
  totalTips: number;
  landRate: number;
  avgLatency: number;
  profitability: number;
}

const BundleCard: React.FC<{ bundle: JitoBundle }> = React.memo(({ bundle }) => {
  const statusColors = {
    pending: 'bg-yellow-500',
    landed: 'bg-green-500',
    failed: 'bg-red-500',
    expired: 'bg-gray-500'
  };
  
  const statusIcons = {
    pending: <Clock className="w-4 h-4" />,
    landed: <CheckCircle className="w-4 h-4" />,
    failed: <XCircle className="w-4 h-4" />,
    expired: <Timer className="w-4 h-4" />
  };
  
  const mevTypeColors = {
    arb: 'text-blue-500',
    liquidation: 'text-red-500',
    sandwich: 'text-yellow-500',
    jit: 'text-purple-500'
  };
  
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.15 }}
    >
      <Card className={`p-3 border-l-4 ${statusColors[bundle.status]} border-opacity-50`}>
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              {statusIcons[bundle.status]}
              <span className="font-mono text-xs text-gray-500">
                {bundle.bundleId.slice(0, 8)}...
              </span>
              <Badge variant="outline" className={mevTypeColors[bundle.mevType]}>
                {bundle.mevType.toUpperCase()}
              </Badge>
              <span className="text-xs text-gray-500">
                {bundle.transactions.length} txs
              </span>
            </div>
            
            <div className="grid grid-cols-4 gap-2 text-xs">
              <div>
                <span className="text-gray-500">Tip:</span>
                <span className="ml-1 font-bold text-green-500">
                  {bundle.tip.toFixed(4)} SOL
                </span>
              </div>
              
              <div>
                <span className="text-gray-500">Latency:</span>
                <span className="ml-1 font-mono">
                  {bundle.submitLatency.toFixed(0)}ms
                </span>
              </div>
              
              {bundle.landingSlot && (
                <div>
                  <span className="text-gray-500">Slot:</span>
                  <span className="ml-1 font-mono">
                    {bundle.landingSlot.toLocaleString()}
                  </span>
                </div>
              )}
              
              {bundle.profitability !== undefined && (
                <div>
                  <span className="text-gray-500">Profit:</span>
                  <span className={`ml-1 font-bold ${bundle.profitability > 0 ? 'text-green-500' : 'text-red-500'}`}>
                    ${bundle.profitability.toFixed(2)}
                  </span>
                </div>
              )}
            </div>
            
            <div className="mt-2 text-xs text-gray-500">
              Relay: {new URL(bundle.relayUrl).hostname}
            </div>
          </div>
          
          <div className="text-xs text-gray-400">
            {formatDistanceToNow(bundle.timestamp, { addSuffix: true })}
          </div>
        </div>
      </Card>
    </motion.div>
  );
});

BundleCard.displayName = 'BundleCard';

export const JitoBundleTracker: React.FC = () => {
  const bundles = useMEVStore(selectRecentBundles);
  const { jitoBundles } = useMEVStore();
  const [selectedRelay, setSelectedRelay] = useState<string>('all');
  const [selectedMevType, setSelectedMevType] = useState<string>('all');
  const chartRef = useRef<any>(null);
  
  const metrics = useMemo<BundleMetrics>(() => {
    const allBundles = Array.from(jitoBundles.values());
    const landed = allBundles.filter(b => b.status === 'landed');
    const failed = allBundles.filter(b => b.status === 'failed');
    const pending = allBundles.filter(b => b.status === 'pending');
    
    const totalTips = allBundles.reduce((sum, b) => sum + b.tip, 0);
    const avgTip = allBundles.length > 0 ? totalTips / allBundles.length : 0;
    const landRate = allBundles.length > 0 ? (landed.length / allBundles.length) * 100 : 0;
    
    const avgLatency = allBundles.length > 0
      ? allBundles.reduce((sum, b) => sum + b.submitLatency, 0) / allBundles.length
      : 0;
    
    const totalProfitability = landed
      .filter(b => b.profitability !== undefined)
      .reduce((sum, b) => sum + (b.profitability || 0), 0);
    
    return {
      totalBundles: allBundles.length,
      landedBundles: landed.length,
      failedBundles: failed.length,
      pendingBundles: pending.length,
      avgTip,
      totalTips,
      landRate,
      avgLatency,
      profitability: totalProfitability
    };
  }, [jitoBundles]);
  
  const filteredBundles = useMemo(() => {
    let filtered = bundles;
    
    if (selectedRelay !== 'all') {
      filtered = filtered.filter(b => new URL(b.relayUrl).hostname === selectedRelay);
    }
    
    if (selectedMevType !== 'all') {
      filtered = filtered.filter(b => b.mevType === selectedMevType);
    }
    
    return filtered.slice(0, 50); // Show top 50
  }, [bundles, selectedRelay, selectedMevType]);
  
  const relayStats = useMemo(() => {
    const stats = new Map<string, { total: number; landed: number; avgTip: number }>();
    
    Array.from(jitoBundles.values()).forEach(bundle => {
      const relay = new URL(bundle.relayUrl).hostname;
      const existing = stats.get(relay) || { total: 0, landed: 0, avgTip: 0 };
      
      existing.total++;
      if (bundle.status === 'landed') existing.landed++;
      existing.avgTip = (existing.avgTip * (existing.total - 1) + bundle.tip) / existing.total;
      
      stats.set(relay, existing);
    });
    
    return Array.from(stats.entries())
      .map(([relay, data]) => ({
        relay,
        ...data,
        landRate: data.total > 0 ? (data.landed / data.total) * 100 : 0
      }))
      .sort((a, b) => b.landRate - a.landRate);
  }, [jitoBundles]);
  
  const landRateChartOptions = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      }
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      top: '10%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: relayStats.map(s => s.relay.split('.')[0]),
      axisLabel: {
        rotate: 45,
        fontSize: 10
      }
    },
    yAxis: [
      {
        type: 'value',
        name: 'Land Rate %',
        position: 'left',
        axisLabel: {
          formatter: '{value}%'
        }
      },
      {
        type: 'value',
        name: 'Avg Tip (SOL)',
        position: 'right'
      }
    ],
    series: [
      {
        name: 'Land Rate',
        type: 'bar',
        data: relayStats.map(s => s.landRate.toFixed(2)),
        itemStyle: {
          color: '#10b981'
        }
      },
      {
        name: 'Avg Tip',
        type: 'line',
        yAxisIndex: 1,
        data: relayStats.map(s => s.avgTip.toFixed(4)),
        itemStyle: {
          color: '#f59e0b'
        }
      }
    ]
  };
  
  const tipDistributionOptions = {
    tooltip: {
      trigger: 'item',
      formatter: '{b}: {c} ({d}%)'
    },
    legend: {
      orient: 'vertical',
      left: 'left',
      top: 'center'
    },
    series: [
      {
        name: 'MEV Type',
        type: 'pie',
        radius: ['40%', '70%'],
        avoidLabelOverlap: false,
        itemStyle: {
          borderRadius: 10,
          borderColor: '#fff',
          borderWidth: 2
        },
        label: {
          show: false,
          position: 'center'
        },
        emphasis: {
          label: {
            show: true,
            fontSize: 20,
            fontWeight: 'bold'
          }
        },
        labelLine: {
          show: false
        },
        data: ['arb', 'liquidation', 'sandwich', 'jit'].map(type => ({
          name: type.toUpperCase(),
          value: Array.from(jitoBundles.values())
            .filter(b => b.mevType === type)
            .reduce((sum, b) => sum + b.tip, 0).toFixed(4)
        }))
      }
    ]
  };
  
  return (
    <div className="space-y-4">
      {/* Metrics Overview */}
      <div className="grid grid-cols-5 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-500">Total Bundles</div>
              <div className="text-2xl font-bold">{metrics.totalBundles}</div>
            </div>
            <Package className="w-8 h-8 text-blue-500" />
          </div>
        </Card>
        
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-500">Land Rate</div>
              <div className="text-2xl font-bold text-green-500">
                {metrics.landRate.toFixed(1)}%
              </div>
            </div>
            <Gauge className="w-8 h-8 text-green-500" />
          </div>
          <Progress value={metrics.landRate} className="mt-2 h-2" />
        </Card>
        
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-500">Avg Tip</div>
              <div className="text-2xl font-bold text-yellow-500">
                {metrics.avgTip.toFixed(4)} SOL
              </div>
            </div>
            <DollarSign className="w-8 h-8 text-yellow-500" />
          </div>
        </Card>
        
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-500">Avg Latency</div>
              <div className="text-2xl font-bold">
                {metrics.avgLatency.toFixed(0)}ms
              </div>
            </div>
            <Timer className="w-8 h-8 text-purple-500" />
          </div>
        </Card>
        
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-500">Total Profit</div>
              <div className={`text-2xl font-bold ${metrics.profitability >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                ${metrics.profitability.toFixed(2)}
              </div>
            </div>
            <TrendingUp className="w-8 h-8 text-green-500" />
          </div>
        </Card>
      </div>
      
      {/* Charts Row */}
      <div className="grid grid-cols-2 gap-4">
        <Card className="p-4">
          <h3 className="text-sm font-semibold mb-2 flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            Relay Performance
          </h3>
          <ReactECharts 
            option={landRateChartOptions} 
            style={{ height: '250px' }}
            ref={chartRef}
          />
        </Card>
        
        <Card className="p-4">
          <h3 className="text-sm font-semibold mb-2 flex items-center gap-2">
            <Layers className="w-4 h-4" />
            Tip Distribution by MEV Type
          </h3>
          <ReactECharts 
            option={tipDistributionOptions} 
            style={{ height: '250px' }}
          />
        </Card>
      </div>
      
      {/* Filters */}
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">Relay:</label>
              <select
                value={selectedRelay}
                onChange={(e) => setSelectedRelay(e.target.value)}
                className="px-2 py-1 text-sm border rounded"
              >
                <option value="all">All Relays</option>
                {relayStats.map(stat => (
                  <option key={stat.relay} value={stat.relay}>
                    {stat.relay}
                  </option>
                ))}
              </select>
            </div>
            
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">MEV Type:</label>
              <select
                value={selectedMevType}
                onChange={(e) => setSelectedMevType(e.target.value)}
                className="px-2 py-1 text-sm border rounded"
              >
                <option value="all">All Types</option>
                <option value="arb">Arbitrage</option>
                <option value="liquidation">Liquidation</option>
                <option value="sandwich">Sandwich</option>
                <option value="jit">JIT</option>
              </select>
            </div>
          </div>
          
          <div className="flex items-center gap-4 text-sm">
            <Badge variant="outline" className="gap-1">
              <CheckCircle className="w-3 h-3 text-green-500" />
              {metrics.landedBundles} Landed
            </Badge>
            <Badge variant="outline" className="gap-1">
              <Clock className="w-3 h-3 text-yellow-500" />
              {metrics.pendingBundles} Pending
            </Badge>
            <Badge variant="outline" className="gap-1">
              <XCircle className="w-3 h-3 text-red-500" />
              {metrics.failedBundles} Failed
            </Badge>
          </div>
        </div>
      </Card>
      
      {/* Bundle List */}
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Send className="w-5 h-5" />
          Recent Bundles
        </h3>
        
        <div className="space-y-2 max-h-[500px] overflow-y-auto">
          <AnimatePresence mode="popLayout">
            {filteredBundles.map((bundle) => (
              <BundleCard key={bundle.id} bundle={bundle} />
            ))}
          </AnimatePresence>
        </div>
      </Card>
    </div>
  );
};