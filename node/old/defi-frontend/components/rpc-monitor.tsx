'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { useNodeStore } from '@/lib/store';
import { formatNumber, formatLatency, formatPercentage } from '@/lib/utils';
import { Server, Activity, AlertCircle, Users, Globe, Gauge } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';

export function RPCMonitor() {
  const rpcMetrics = useNodeStore((state) => state.rpcMetrics);

  if (!rpcMetrics) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>RPC Monitor</CardTitle>
          <CardDescription>Waiting for RPC metrics...</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  // Convert methods object to array for chart
  const methodsData = rpcMetrics.methods
    ? Object.entries(rpcMetrics.methods)
        .map(([method, stats]) => ({
          method,
          count: stats.count,
          avgTime: stats.avgTime,
          errors: stats.errors,
        }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 10)
    : [];

  return (
    <div className="space-y-6">
      {/* RPC Performance Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Requests/sec</CardTitle>
              <Server className="h-4 w-4 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatNumber(rpcMetrics.requestsPerSecond)}</div>
            <p className="text-xs text-muted-foreground">Current throughput</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Avg Response Time</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatLatency(rpcMetrics.avgResponseTime)}</div>
            <p className="text-xs text-muted-foreground">Average latency</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Error Rate</CardTitle>
              <AlertCircle className="h-4 w-4 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-500">
              {formatPercentage(rpcMetrics.errorRate)}
            </div>
            <Progress value={rpcMetrics.errorRate} className="h-2 mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Active Connections</CardTitle>
              <Users className="h-4 w-4 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatNumber(rpcMetrics.activeConnections)}</div>
            <div className="text-xs text-muted-foreground">
              WS: {rpcMetrics.wsConnections} | HTTP: {rpcMetrics.httpConnections}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Latency Percentiles */}
      <Card>
        <CardHeader>
          <CardTitle>Response Time Percentiles</CardTitle>
          <CardDescription>Latency distribution across requests</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">P50 (Median)</span>
                <Gauge className="h-4 w-4 text-blue-500" />
              </div>
              <div className="text-2xl font-bold text-blue-500">
                {formatLatency(rpcMetrics.avgResponseTime)}
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">P95</span>
                <Gauge className="h-4 w-4 text-yellow-500" />
              </div>
              <div className="text-2xl font-bold text-yellow-500">
                {formatLatency(rpcMetrics.p95ResponseTime)}
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">P99</span>
                <Gauge className="h-4 w-4 text-red-500" />
              </div>
              <div className="text-2xl font-bold text-red-500">
                {formatLatency(rpcMetrics.p99ResponseTime)}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Top RPC Methods */}
      <Card>
        <CardHeader>
          <CardTitle>Top RPC Methods</CardTitle>
          <CardDescription>Most frequently called RPC methods</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={methodsData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis
                dataKey="method"
                className="text-xs"
                tick={{ fill: 'hsl(var(--muted-foreground))' }}
                angle={-45}
                textAnchor="end"
                height={80}
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

      {/* Method Performance Table */}
      <Card>
        <CardHeader>
          <CardTitle>Method Performance Details</CardTitle>
          <CardDescription>Detailed statistics for each RPC method</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="relative overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-xs uppercase bg-muted">
                <tr>
                  <th className="px-4 py-2 text-left">Method</th>
                  <th className="px-4 py-2 text-right">Calls</th>
                  <th className="px-4 py-2 text-right">Avg Time</th>
                  <th className="px-4 py-2 text-right">Errors</th>
                  <th className="px-4 py-2 text-right">Error Rate</th>
                </tr>
              </thead>
              <tbody>
                {methodsData.map((method) => (
                  <tr key={method.method} className="border-b">
                    <td className="px-4 py-2 font-mono text-xs">{method.method}</td>
                    <td className="px-4 py-2 text-right">{formatNumber(method.count)}</td>
                    <td className="px-4 py-2 text-right">{formatLatency(method.avgTime)}</td>
                    <td className="px-4 py-2 text-right text-red-500">{method.errors}</td>
                    <td className="px-4 py-2 text-right">
                      {formatPercentage((method.errors / method.count) * 100)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}