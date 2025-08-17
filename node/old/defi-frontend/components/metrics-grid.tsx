'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useNodeStore } from '@/lib/store';
import { formatNumber, formatPercentage, formatBytes, formatLatency } from '@/lib/utils';
import {
  Activity,
  Blocks,
  Cpu,
  Database,
  Globe,
  HardDrive,
  Heart,
  Layers,
  MemoryStick,
  Network,
  Percent,
  Server,
  TrendingUp,
  Zap,
} from 'lucide-react';
import { cn } from '@/lib/utils';

export function MetricsGrid() {
  const nodeMetrics = useNodeStore((state) => state.nodeMetrics);
  const systemMetrics = useNodeStore((state) => state.systemMetrics);
  const jitoMetrics = useNodeStore((state) => state.jitoMetrics);
  const rpcMetrics = useNodeStore((state) => state.rpcMetrics);

  const metrics = [
    {
      title: 'Node Health',
      value: nodeMetrics?.health || 'unknown',
      icon: Heart,
      color: nodeMetrics?.health === 'healthy' ? 'text-green-500' : 
             nodeMetrics?.health === 'warning' ? 'text-yellow-500' : 
             'text-red-500',
      format: (v: string) => v.toUpperCase(),
    },
    {
      title: 'Current Slot',
      value: nodeMetrics?.slot || 0,
      icon: Blocks,
      color: 'text-blue-500',
      format: (v: number) => formatNumber(v),
    },
    {
      title: 'Epoch',
      value: nodeMetrics?.epoch || 0,
      icon: Layers,
      color: 'text-purple-500',
      format: (v: number) => v.toString(),
      subtitle: `${formatPercentage((nodeMetrics?.slotIndex || 0) / (nodeMetrics?.slotsInEpoch || 1) * 100)} complete`,
    },
    {
      title: 'Skip Rate',
      value: nodeMetrics?.skipRate || 0,
      icon: Percent,
      color: nodeMetrics?.skipRate && nodeMetrics.skipRate > 5 ? 'text-yellow-500' : 'text-green-500',
      format: (v: number) => formatPercentage(v),
    },
    {
      title: 'CPU Usage',
      value: systemMetrics?.cpuUsage || 0,
      icon: Cpu,
      color: systemMetrics?.cpuUsage && systemMetrics.cpuUsage > 80 ? 'text-red-500' : 'text-blue-500',
      format: (v: number) => formatPercentage(v),
    },
    {
      title: 'Memory Usage',
      value: systemMetrics?.memoryPercent || 0,
      icon: MemoryStick,
      color: systemMetrics?.memoryPercent && systemMetrics.memoryPercent > 80 ? 'text-red-500' : 'text-blue-500',
      format: (v: number) => formatPercentage(v),
      subtitle: systemMetrics ? `${formatBytes(systemMetrics.memoryUsed * 1024 * 1024)} / ${formatBytes(systemMetrics.memoryTotal * 1024 * 1024)}` : undefined,
    },
    {
      title: 'Disk Usage',
      value: systemMetrics?.diskPercent || 0,
      icon: HardDrive,
      color: systemMetrics?.diskPercent && systemMetrics.diskPercent > 80 ? 'text-red-500' : 'text-blue-500',
      format: (v: number) => formatPercentage(v),
      subtitle: systemMetrics ? `${formatBytes(systemMetrics.diskUsed * 1024 * 1024 * 1024)} / ${formatBytes(systemMetrics.diskTotal * 1024 * 1024 * 1024)}` : undefined,
    },
    {
      title: 'Network RX',
      value: systemMetrics?.networkRx || 0,
      icon: Network,
      color: 'text-green-500',
      format: (v: number) => `${formatBytes(v)}/s`,
    },
    {
      title: 'Network TX',
      value: systemMetrics?.networkTx || 0,
      icon: Network,
      color: 'text-blue-500',
      format: (v: number) => `${formatBytes(v)}/s`,
    },
    {
      title: 'RPC RPS',
      value: rpcMetrics?.requestsPerSecond || 0,
      icon: Server,
      color: 'text-purple-500',
      format: (v: number) => formatNumber(v),
    },
    {
      title: 'RPC Latency',
      value: rpcMetrics?.avgResponseTime || 0,
      icon: Activity,
      color: rpcMetrics?.avgResponseTime && rpcMetrics.avgResponseTime > 100 ? 'text-yellow-500' : 'text-green-500',
      format: (v: number) => formatLatency(v),
    },
    {
      title: 'Jito Bundles',
      value: jitoMetrics?.bundlesLanded || 0,
      icon: Zap,
      color: 'text-solana-purple',
      format: (v: number) => formatNumber(v),
      subtitle: jitoMetrics ? `${formatPercentage(jitoMetrics.bundleRate)} land rate` : undefined,
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      {metrics.map((metric) => {
        const Icon = metric.icon;
        return (
          <Card key={metric.title} className="relative overflow-hidden">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                {metric.title}
              </CardTitle>
              <Icon className={cn('h-4 w-4', metric.color)} />
            </CardHeader>
            <CardContent>
              <div className={cn('text-2xl font-bold', metric.color)}>
                {metric.format(metric.value)}
              </div>
              {metric.subtitle && (
                <p className="text-xs text-muted-foreground mt-1">
                  {metric.subtitle}
                </p>
              )}
              <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-current to-transparent opacity-20" />
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}