'use client';

import { useMemo, memo } from 'react';
import { useMonitoringStore } from '@/lib/monitoring-store';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Progress } from '@/components/ui/Progress';
import { Separator } from '@/components/ui/separator';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadialBarChart,
  RadialBar,
} from 'recharts';
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  Layers,
  TrendingUp,
  TrendingDown,
  Zap,
  Shield,
  Award,
  Target,
  Users,
  DollarSign,
} from 'lucide-react';

export const ConsensusPanel = memo(() => {
  const consensus = useMonitoringStore((state) => state.consensus);
  const consensusHistory = useMonitoringStore((state) => state.consensusHistory);
  
  // Calculate derived metrics
  const votingEfficiency = useMemo(() => {
    if (!consensus) return 0;
    return ((consensus.blocksProduced / Math.max(consensus.leaderSlots, 1)) * 100);
  }, [consensus]);
  
  const creditEfficiency = useMemo(() => {
    if (!consensus) return 0;
    return (consensus.credits / Math.max(consensus.epochCredits, 1)) * 100;
  }, [consensus]);
  
  const towerSync = useMemo(() => {
    if (!consensus) return 0;
    return ((consensus.towerRoot / Math.max(consensus.towerHeight, 1)) * 100);
  }, [consensus]);
  
  // Prepare chart data
  const slotProgressData = useMemo(() => {
    return consensusHistory.data.slice(-50).map((point) => ({
      time: new Date(point.timestamp).toLocaleTimeString(),
      slot: point.value.slot || 0,
      root: point.value.rootSlot || 0,
      optimistic: point.value.optimisticSlot || 0,
    }));
  }, [consensusHistory]);
  
  const skipRateData = useMemo(() => {
    return consensusHistory.data.slice(-20).map((point) => ({
      time: new Date(point.timestamp).toLocaleTimeString(),
      skipRate: point.value.skipRate || 0,
      target: 5, // Target skip rate
    }));
  }, [consensusHistory]);
  
  const stakeDistribution = useMemo(() => {
    if (!consensus) return [];
    const total = Number(consensus.validatorActiveStake + consensus.validatorDelinquentStake);
    return [
      { name: 'Active', value: Number(consensus.validatorActiveStake) / 1e9, percentage: (Number(consensus.validatorActiveStake) / total) * 100 },
      { name: 'Delinquent', value: Number(consensus.validatorDelinquentStake) / 1e9, percentage: (Number(consensus.validatorDelinquentStake) / total) * 100 },
    ];
  }, [consensus]);
  
  const performanceMetrics = useMemo(() => {
    if (!consensus) return [];
    return [
      { metric: 'Voting', value: votingEfficiency, fill: '#8884d8' },
      { metric: 'Credits', value: creditEfficiency, fill: '#82ca9d' },
      { metric: 'Tower Sync', value: towerSync, fill: '#ffc658' },
    ];
  }, [consensus, votingEfficiency, creditEfficiency, towerSync]);
  
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];
  
  if (!consensus) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-96">
          <p className="text-muted-foreground">Waiting for consensus data...</p>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <div className="space-y-4">
      {/* Status Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Layers className="w-5 h-5" />
              Consensus Status
            </span>
            <Badge
              variant={consensus.votingState === 'voting' ? 'default' : 'destructive'}
              className="text-sm"
            >
              {consensus.votingState.toUpperCase()}
            </Badge>
          </CardTitle>
          <CardDescription>
            Validator consensus participation and voting metrics
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Current Slot</span>
                <Clock className="w-4 h-4 text-muted-foreground" />
              </div>
              <p className="text-2xl font-bold">{consensus.lastVoteSlot.toLocaleString()}</p>
              <Progress value={75} className="h-1" />
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Root Slot</span>
                <Shield className="w-4 h-4 text-muted-foreground" />
              </div>
              <p className="text-2xl font-bold">{consensus.rootSlot.toLocaleString()}</p>
              <Progress value={90} className="h-1" />
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Skip Rate</span>
                {consensus.skipRate > 5 ? (
                  <AlertTriangle className="w-4 h-4 text-yellow-500" />
                ) : (
                  <CheckCircle className="w-4 h-4 text-green-500" />
                )}
              </div>
              <p className="text-2xl font-bold">{consensus.skipRate.toFixed(2)}%</p>
              <Progress 
                value={Math.min(consensus.skipRate * 10, 100)} 
                className="h-1"
              />
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Blocks Produced</span>
                <Award className="w-4 h-4 text-muted-foreground" />
              </div>
              <p className="text-2xl font-bold">
                {consensus.blocksProduced}/{consensus.leaderSlots}
              </p>
              <Progress 
                value={votingEfficiency} 
                className="h-1"
              />
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Slot Progression Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Slot Progression</CardTitle>
            <CardDescription>Real-time slot advancement tracking</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={slotProgressData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="slot"
                  stackId="1"
                  stroke="#8884d8"
                  fill="#8884d8"
                  fillOpacity={0.6}
                />
                <Area
                  type="monotone"
                  dataKey="root"
                  stackId="2"
                  stroke="#82ca9d"
                  fill="#82ca9d"
                  fillOpacity={0.6}
                />
                <Area
                  type="monotone"
                  dataKey="optimistic"
                  stackId="3"
                  stroke="#ffc658"
                  fill="#ffc658"
                  fillOpacity={0.6}
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        
        {/* Skip Rate Trend */}
        <Card>
          <CardHeader>
            <CardTitle>Skip Rate Trend</CardTitle>
            <CardDescription>Block production skip rate over time</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={skipRateData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="skipRate"
                  stroke="#ff7300"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  activeDot={{ r: 6 }}
                />
                <Line
                  type="monotone"
                  dataKey="target"
                  stroke="#82ca9d"
                  strokeDasharray="5 5"
                  strokeWidth={1}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
      
      {/* Performance Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Radial Performance Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Performance Metrics</CardTitle>
            <CardDescription>Key consensus performance indicators</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <RadialBarChart cx="50%" cy="50%" innerRadius="10%" outerRadius="90%" data={performanceMetrics}>
                <RadialBar dataKey="value" cornerRadius={10} fill="#8884d8" />
                <Legend />
                <Tooltip />
              </RadialBarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        
        {/* Stake Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Stake Distribution</CardTitle>
            <CardDescription>Active vs Delinquent stake</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={stakeDistribution}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={(entry) => `${entry.name}: ${entry.percentage.toFixed(1)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {stakeDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        
        {/* Tower State */}
        <Card>
          <CardHeader>
            <CardTitle>Tower State</CardTitle>
            <CardDescription>Consensus tower synchronization</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Tower Height</span>
                <span className="font-bold">{consensus.towerHeight}</span>
              </div>
              <Progress value={75} className="h-2" />
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Tower Root</span>
                <span className="font-bold">{consensus.towerRoot}</span>
              </div>
              <Progress value={towerSync} className="h-2" />
            </div>
            
            <Separator />
            
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Optimistic Slot</span>
                <span className="font-bold">{consensus.optimisticSlot.toLocaleString()}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Confirmation Time</span>
                <span className="font-bold">{consensus.optimisticConfirmationTime}ms</span>
              </div>
            </div>
            
            <div className="flex items-center justify-between p-2 bg-muted rounded-lg">
              <span className="text-sm">Tower Sync</span>
              <Badge variant={towerSync > 90 ? 'default' : 'secondary'}>
                {towerSync.toFixed(1)}%
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* Credits and Commission */}
      <Card>
        <CardHeader>
          <CardTitle>Credits & Commission</CardTitle>
          <CardDescription>Epoch credits and commission settings</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Current Credits</p>
              <p className="text-xl font-bold">{consensus.credits.toLocaleString()}</p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Epoch Credits</p>
              <p className="text-xl font-bold">{consensus.epochCredits.toLocaleString()}</p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Commission</p>
              <p className="text-xl font-bold">{consensus.commission}%</p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Credit Efficiency</p>
              <div className="flex items-center gap-2">
                <p className="text-xl font-bold">{creditEfficiency.toFixed(1)}%</p>
                {creditEfficiency > 95 ? (
                  <TrendingUp className="w-4 h-4 text-green-500" />
                ) : (
                  <TrendingDown className="w-4 h-4 text-red-500" />
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
});

ConsensusPanel.displayName = 'ConsensusPanel';