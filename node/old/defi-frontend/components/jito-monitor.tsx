'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { useNodeStore } from '@/lib/store';
import { formatNumber, formatPercentage, formatLatency } from '@/lib/utils';
import { Zap, TrendingUp, Clock, DollarSign, Activity, CheckCircle, XCircle } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';

export function JitoMonitor() {
  const jitoMetrics = useNodeStore((state) => state.jitoMetrics);

  if (!jitoMetrics) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Jito MEV Monitor</CardTitle>
          <CardDescription>Waiting for Jito metrics...</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const bundleData = [
    { name: 'Landed', value: jitoMetrics.bundlesLanded, color: '#14F195' },
    { name: 'Failed', value: jitoMetrics.bundlesReceived - jitoMetrics.bundlesLanded, color: '#ef4444' },
  ];

  const tipData = jitoMetrics.tipDistribution?.map((tip, index) => ({
    range: `${index * 0.01}-${(index + 1) * 0.01} SOL`,
    count: tip,
  })) || [];

  return (
    <div className="space-y-6">
      {/* Connection Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Block Engine Connection</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-2">
              {jitoMetrics.blockEngineConnected ? (
                <>
                  <CheckCircle className="h-5 w-5 text-green-500" />
                  <span className="text-green-500 font-medium">Connected</span>
                </>
              ) : (
                <>
                  <XCircle className="h-5 w-5 text-red-500" />
                  <span className="text-red-500 font-medium">Disconnected</span>
                </>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Relayer Connection</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-2">
              {jitoMetrics.relayerConnected ? (
                <>
                  <CheckCircle className="h-5 w-5 text-green-500" />
                  <span className="text-green-500 font-medium">Connected</span>
                </>
              ) : (
                <>
                  <XCircle className="h-5 w-5 text-red-500" />
                  <span className="text-red-500 font-medium">Disconnected</span>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Bundle Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Bundles Received</CardTitle>
              <Zap className="h-4 w-4 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatNumber(jitoMetrics.bundlesReceived)}</div>
            <p className="text-xs text-muted-foreground">Total bundles</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Bundles Landed</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-500">
              {formatNumber(jitoMetrics.bundlesLanded)}
            </div>
            <p className="text-xs text-muted-foreground">Successfully included</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Landing Rate</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatPercentage(jitoMetrics.bundleRate)}</div>
            <Progress value={jitoMetrics.bundleRate} className="h-2 mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Avg Latency</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatLatency(jitoMetrics.avgBundleLatency)}</div>
            <p className="text-xs text-muted-foreground">Bundle processing time</p>
          </CardContent>
        </Card>
      </div>

      {/* MEV Rewards */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>MEV Rewards</CardTitle>
              <CardDescription>Total MEV earnings from Jito bundles</CardDescription>
            </div>
            <DollarSign className="h-5 w-5 text-solana-green" />
          </div>
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-bold text-solana-green">
            {jitoMetrics.mevRewards.toFixed(4)} SOL
          </div>
          <p className="text-sm text-muted-foreground mt-2">
            Approximately ${(jitoMetrics.mevRewards * 150).toFixed(2)} USD
          </p>
        </CardContent>
      </Card>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Bundle Success Rate Pie Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Bundle Success Distribution</CardTitle>
            <CardDescription>Landed vs Failed bundles</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={bundleData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {bundleData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Tip Distribution Bar Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Tip Distribution</CardTitle>
            <CardDescription>Bundle tips by amount range</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={tipData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis
                  dataKey="range"
                  className="text-xs"
                  tick={{ fill: 'hsl(var(--muted-foreground))' }}
                />
                <YAxis
                  className="text-xs"
                  tick={{ fill: 'hsl(var(--muted-foreground))' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--background))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '6px',
                  }}
                />
                <Bar dataKey="count" fill="#9945FF" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}