'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { useNodeStore } from '@/lib/store';
import { formatBytes, formatPercentage } from '@/lib/utils';
import { Cpu, MemoryStick, HardDrive, Network, Activity, Clock } from 'lucide-react';
import { ChartContainer } from '@/components/chart-container';

export function SystemMonitor() {
  const systemMetrics = useNodeStore((state) => state.systemMetrics);
  const cpuHistory = useNodeStore((state) => state.cpuHistory);
  const memoryHistory = useNodeStore((state) => state.memoryHistory);
  const networkHistory = useNodeStore((state) => state.networkHistory);

  if (!systemMetrics) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>System Monitor</CardTitle>
          <CardDescription>Waiting for system metrics...</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Resource Usage Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">CPU Usage</CardTitle>
              <Cpu className="h-4 w-4 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-2xl font-bold">{formatPercentage(systemMetrics.cpuUsage)}</div>
            <Progress value={systemMetrics.cpuUsage} className="h-2" />
            <div className="text-xs text-muted-foreground">
              {systemMetrics.cpuCores} cores @ {systemMetrics.cpuFreq?.toFixed(2)} GHz
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Memory</CardTitle>
              <MemoryStick className="h-4 w-4 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-2xl font-bold">{formatPercentage(systemMetrics.memoryPercent)}</div>
            <Progress value={systemMetrics.memoryPercent} className="h-2" />
            <div className="text-xs text-muted-foreground">
              {formatBytes(systemMetrics.memoryUsed * 1024 * 1024)} / {formatBytes(systemMetrics.memoryTotal * 1024 * 1024)}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Disk</CardTitle>
              <HardDrive className="h-4 w-4 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-2xl font-bold">{formatPercentage(systemMetrics.diskPercent)}</div>
            <Progress value={systemMetrics.diskPercent} className="h-2" />
            <div className="text-xs text-muted-foreground">
              {formatBytes(systemMetrics.diskUsed * 1024 * 1024 * 1024)} / {formatBytes(systemMetrics.diskTotal * 1024 * 1024 * 1024)}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Load Average</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-2xl font-bold">{systemMetrics.loadAverage[0].toFixed(2)}</div>
            <div className="text-xs text-muted-foreground">
              1m: {systemMetrics.loadAverage[0].toFixed(2)} | 
              5m: {systemMetrics.loadAverage[1].toFixed(2)} | 
              15m: {systemMetrics.loadAverage[2].toFixed(2)}
            </div>
            <div className="text-xs text-muted-foreground">
              {systemMetrics.processes} processes
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartContainer
          title="CPU Usage History"
          description="CPU utilization over time"
          dataKey="cpuHistory"
          color="hsl(var(--primary))"
        />
        <ChartContainer
          title="Memory Usage History"
          description="Memory utilization over time"
          dataKey="memoryHistory"
          color="hsl(var(--accent))"
        />
      </div>

      {/* Network Stats */}
      <Card>
        <CardHeader>
          <CardTitle>Network Statistics</CardTitle>
          <CardDescription>Real-time network throughput</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Download</span>
                <Network className="h-4 w-4 text-green-500" />
              </div>
              <div className="text-2xl font-bold text-green-500">
                {formatBytes(systemMetrics.networkRx)}/s
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Upload</span>
                <Network className="h-4 w-4 text-blue-500" />
              </div>
              <div className="text-2xl font-bold text-blue-500">
                {formatBytes(systemMetrics.networkTx)}/s
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* System Info */}
      <Card>
        <CardHeader>
          <CardTitle>System Information</CardTitle>
          <CardDescription>Hardware and OS details</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div className="text-xs text-muted-foreground">Uptime</div>
              <div className="font-mono text-sm">
                {Math.floor(systemMetrics.uptime / 86400)}d {Math.floor((systemMetrics.uptime % 86400) / 3600)}h
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">CPU Cores</div>
              <div className="font-mono text-sm">{systemMetrics.cpuCores}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Total RAM</div>
              <div className="font-mono text-sm">{formatBytes(systemMetrics.memoryTotal * 1024 * 1024)}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Total Disk</div>
              <div className="font-mono text-sm">{formatBytes(systemMetrics.diskTotal * 1024 * 1024 * 1024)}</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}