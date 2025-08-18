'use client';

import { useMemo, memo } from 'react';
import { useMonitoringStore } from '@/lib/monitoring-store';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Progress } from '@/components/ui/Progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Brush,
  ScatterChart,
  Scatter,
} from 'recharts';
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Zap,
  Clock,
  Package,
  Layers,
  GitBranch,
  Server,
  Database,
  AlertCircle,
  CheckCircle,
} from 'lucide-react';

export const PerformancePanel = memo(() => {
  const performance = useMonitoringStore((state) => state.performance) as any;
  const performanceHistory = useMonitoringStore((state) => state.performanceHistory) as any;
  
  // Prepare TPS chart data
  const tpsData = useMemo(() => {
    return performanceHistory.data.slice(-100).map((point) => ({
      time: new Date(point.timestamp).toLocaleTimeString(),
      current: point.value.currentTPS || 0,
      average: point.value.averageTPS || 0,
      peak: point.value.peakTPS || 0,
    }));
  }, [performanceHistory]);
  
  // Stage latency data
  const stageLatencyData = useMemo(() => {
    if (!performance) return [];
    return [
      { stage: 'Banking', latency: performance.bankingStage.processingTime, packets: performance.bankingStage.bufferedPackets },
      { stage: 'Fetch', latency: performance.fetchStage.latency, packets: performance.fetchStage.packetsReceived },
      { stage: 'Vote', latency: performance.voteStage.voteLatency, packets: performance.voteStage.votesProcessed },
      { stage: 'Shred', latency: performance.shredStage.shredLatency, packets: performance.shredStage.shredsReceived },
      { stage: 'Replay', latency: performance.replayStage.replayLatency, packets: performance.replayStage.slotsReplayed },
    ];
  }, [performance]);
  
  // Packet flow data
  const packetFlowData = useMemo(() => {
    return performanceHistory.data.slice(-50).map((point) => ({
      time: new Date(point.timestamp).toLocaleTimeString(),
      received: point.value.fetchStage?.packetsReceived || 0,
      processed: point.value.fetchStage?.packetsProcessed || 0,
      forwarded: point.value.bankingStage?.forwardedPackets || 0,
      dropped: point.value.bankingStage?.droppedPackets || 0,
    }));
  }, [performanceHistory]);
  
  // Thread utilization
  const threadUtilization = useMemo(() => {
    if (!performance) return [];
    const total = 32; // Assuming 32 threads total
    return [
      { name: 'Active', value: performance.bankingStage.threadsActive, percentage: (performance.bankingStage.threadsActive / total) * 100 },
      { name: 'Idle', value: total - performance.bankingStage.threadsActive, percentage: ((total - performance.bankingStage.threadsActive) / total) * 100 },
    ];
  }, [performance]);
  
  // Confirmation time distribution
  const confirmationData = useMemo(() => {
    return performanceHistory.data.slice(-30).map((point) => ({
      time: new Date(point.timestamp).toLocaleTimeString(),
      slot: point.value.slotTime || 0,
      confirmation: point.value.confirmationTime || 0,
    }));
  }, [performanceHistory]);
  
  if (!performance) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-96">
          <p className="text-muted-foreground">Waiting for performance data...</p>
        </CardContent>
      </Card>
    );
  }
  
  const getTPSStatus = (current: number, average: number) => {
    const ratio = current / average;
    if (ratio > 1.2) return { color: 'text-green-500', icon: TrendingUp, text: 'Above Average' };
    if (ratio < 0.8) return { color: 'text-red-500', icon: TrendingDown, text: 'Below Average' };
    return { color: 'text-yellow-500', icon: Activity, text: 'Normal' };
  };
  
  const tpsStatus = getTPSStatus(performance.currentTPS, performance.averageTPS);
  const TPSIcon = tpsStatus.icon;
  
  return (
    <div className="space-y-4">
      {/* TPS Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Zap className="w-5 h-5" />
              Performance Metrics
            </span>
            <Badge variant="outline" className={tpsStatus.color}>
              <TPSIcon className="w-3 h-3 mr-1" />
              {tpsStatus.text}
            </Badge>
          </CardTitle>
          <CardDescription>
            Transaction processing and stage performance metrics
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Current TPS</p>
              <p className="text-3xl font-bold">{performance.currentTPS.toLocaleString()}</p>
              <Progress value={(performance.currentTPS / performance.peakTPS) * 100} className="h-1" />
            </div>
            
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Average TPS</p>
              <p className="text-3xl font-bold">{performance.averageTPS.toLocaleString()}</p>
              <Progress value={(performance.averageTPS / performance.peakTPS) * 100} className="h-1" />
            </div>
            
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Peak TPS</p>
              <p className="text-3xl font-bold">{performance.peakTPS.toLocaleString()}</p>
              <Badge variant="secondary" className="mt-1">Record</Badge>
            </div>
            
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Slot Time</p>
              <p className="text-3xl font-bold">{performance.slotTime}ms</p>
              <Progress value={Math.min((400 / performance.slotTime) * 100, 100)} className="h-1" />
            </div>
            
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Confirmation</p>
              <p className="text-3xl font-bold">{performance.confirmationTime}ms</p>
              <Progress value={Math.min((1000 / performance.confirmationTime) * 100, 100)} className="h-1" />
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* TPS Chart */}
      <Card>
        <CardHeader>
          <CardTitle>TPS History</CardTitle>
          <CardDescription>Transaction throughput over time</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={350}>
            <ComposedChart data={tpsData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Area
                type="monotone"
                dataKey="current"
                fill="#8884d8"
                stroke="#8884d8"
                fillOpacity={0.6}
              />
              <Line
                type="monotone"
                dataKey="average"
                stroke="#82ca9d"
                strokeWidth={2}
                strokeDasharray="5 5"
              />
              <Line
                type="monotone"
                dataKey="peak"
                stroke="#ff7300"
                strokeWidth={1}
                strokeDasharray="3 3"
              />
              <Brush dataKey="time" height={30} stroke="#8884d8" />
            </ComposedChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
      
      {/* Stage Performance Tabs */}
      <Card>
        <CardHeader>
          <CardTitle>Pipeline Stages</CardTitle>
          <CardDescription>Detailed stage-by-stage performance analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="overview" className="space-y-4">
            <TabsList className="grid grid-cols-6 w-full">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="banking">Banking</TabsTrigger>
              <TabsTrigger value="fetch">Fetch</TabsTrigger>
              <TabsTrigger value="vote">Vote</TabsTrigger>
              <TabsTrigger value="shred">Shred</TabsTrigger>
              <TabsTrigger value="replay">Replay</TabsTrigger>
            </TabsList>
            
            <TabsContent value="overview" className="space-y-4">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={stageLatencyData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="stage" />
                  <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
                  <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
                  <Tooltip />
                  <Legend />
                  <Bar yAxisId="left" dataKey="latency" fill="#8884d8" name="Latency (ms)" />
                  <Bar yAxisId="right" dataKey="packets" fill="#82ca9d" name="Packets" />
                </BarChart>
              </ResponsiveContainer>
            </TabsContent>
            
            <TabsContent value="banking" className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Buffered Packets</p>
                  <p className="text-2xl font-bold">{performance.bankingStage.bufferedPackets.toLocaleString()}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Forwarded</p>
                  <p className="text-2xl font-bold">{performance.bankingStage.forwardedPackets.toLocaleString()}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Dropped</p>
                  <p className="text-2xl font-bold text-red-500">{performance.bankingStage.droppedPackets.toLocaleString()}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Active Threads</p>
                  <p className="text-2xl font-bold">{performance.bankingStage.threadsActive}/32</p>
                </div>
              </div>
              
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={packetFlowData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area type="monotone" dataKey="processed" stackId="1" stroke="#82ca9d" fill="#82ca9d" />
                  <Area type="monotone" dataKey="forwarded" stackId="1" stroke="#8884d8" fill="#8884d8" />
                  <Area type="monotone" dataKey="dropped" stackId="1" stroke="#ff7300" fill="#ff7300" />
                </AreaChart>
              </ResponsiveContainer>
            </TabsContent>
            
            <TabsContent value="fetch" className="space-y-4">
              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Packets Received</p>
                  <p className="text-2xl font-bold">{performance.fetchStage.packetsReceived.toLocaleString()}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Packets Processed</p>
                  <p className="text-2xl font-bold">{performance.fetchStage.packetsProcessed.toLocaleString()}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Latency</p>
                  <p className="text-2xl font-bold">{performance.fetchStage.latency}ms</p>
                </div>
              </div>
              
              <Progress 
                value={(performance.fetchStage.packetsProcessed / performance.fetchStage.packetsReceived) * 100} 
                className="h-2"
              />
            </TabsContent>
            
            <TabsContent value="vote" className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Votes Processed</p>
                  <p className="text-2xl font-bold">{performance.voteStage.votesProcessed.toLocaleString()}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Vote Latency</p>
                  <p className="text-2xl font-bold">{performance.voteStage.voteLatency}ms</p>
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="shred" className="space-y-4">
              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Shreds Received</p>
                  <p className="text-2xl font-bold">{performance.shredStage.shredsReceived.toLocaleString()}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Shreds Inserted</p>
                  <p className="text-2xl font-bold">{performance.shredStage.shredsInserted.toLocaleString()}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Latency</p>
                  <p className="text-2xl font-bold">{performance.shredStage.shredLatency}ms</p>
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="replay" className="space-y-4">
              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Slots Replayed</p>
                  <p className="text-2xl font-bold">{performance.replayStage.slotsReplayed.toLocaleString()}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Fork Weight</p>
                  <p className="text-2xl font-bold">{performance.replayStage.forkWeight.toLocaleString()}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Replay Latency</p>
                  <p className="text-2xl font-bold">{performance.replayStage.replayLatency}ms</p>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
      
      {/* Confirmation Times */}
      <Card>
        <CardHeader>
          <CardTitle>Confirmation Performance</CardTitle>
          <CardDescription>Slot time and confirmation latency</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={confirmationData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="slot" stroke="#8884d8" name="Slot Time (ms)" strokeWidth={2} />
              <Line type="monotone" dataKey="confirmation" stroke="#82ca9d" name="Confirmation (ms)" strokeWidth={2} />
              <ReferenceLine y={400} stroke="red" strokeDasharray="3 3" label="Target" />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
});

PerformancePanel.displayName = 'PerformancePanel';