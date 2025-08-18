'use client';

import { useMemo, memo } from 'react';
import { useMonitoringStore } from '@/lib/monitoring-store';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Progress } from '@/components/ui/Progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import {
  RadialBarChart,
  RadialBar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from 'recharts';
import {
  Heart,
  Activity,
  CheckCircle,
  AlertTriangle,
  XCircle,
  TrendingUp,
  TrendingDown,
  Zap,
  Shield,
  Server,
  Cpu,
  Network,
  Database,
  Layers,
  AlertCircle,
} from 'lucide-react';

export const HealthOverview = memo(() => {
  const health = useMonitoringStore((state) => state.health);
  const consensus = useMonitoringStore((state) => state.consensus);
  const performance = useMonitoringStore((state) => state.performance);
  const os = useMonitoringStore((state) => state.os);
  const activeAlerts = useMonitoringStore((state) => state.activeAlerts);
  
  // Calculate health metrics for radar chart
  const healthMetrics = useMemo(() => {
    if (!health) return [];
    
    return [
      { metric: 'Consensus', value: health.components.consensus, fullMark: 100 },
      { metric: 'Performance', value: health.components.performance, fullMark: 100 },
      { metric: 'RPC', value: health.components.rpc, fullMark: 100 },
      { metric: 'Network', value: health.components.network, fullMark: 100 },
      { metric: 'System', value: health.components.system, fullMark: 100 },
      { metric: 'Jito', value: health.components.jito, fullMark: 100 },
      { metric: 'Geyser', value: health.components.geyser, fullMark: 100 },
      { metric: 'Security', value: health.components.security, fullMark: 100 },
    ];
  }, [health]);
  
  // Overall health for radial chart
  const overallHealthData = useMemo(() => {
    if (!health) return [];
    
    return [{
      name: 'Health',
      value: health.overall,
      fill: health.overall >= 90 ? '#10b981' : health.overall >= 70 ? '#f59e0b' : '#ef4444',
    }];
  }, [health]);
  
  // Critical metrics
  const criticalMetrics = useMemo(() => {
    const metrics: any[] = [];
    
    if (consensus?.skipRate > 5) {
      metrics.push({
        type: 'warning',
        label: 'High Skip Rate',
        value: `${consensus.skipRate.toFixed(2)}%`,
        icon: AlertTriangle,
      });
    }
    
    if (consensus?.votingState === 'delinquent') {
      metrics.push({
        type: 'critical',
        label: 'Validator Delinquent',
        value: 'Action Required',
        icon: XCircle,
      });
    }
    
    if (performance && performance.currentTPS < performance.averageTPS * 0.5) {
      metrics.push({
        type: 'warning',
        label: 'Low TPS',
        value: `${performance.currentTPS} TPS`,
        icon: TrendingDown,
      });
    }
    
    if (os) {
      const memUsage = (os.memoryDetails.used / (os.memoryDetails.used + os.memoryDetails.free)) * 100;
      if (memUsage > 90) {
        metrics.push({
          type: 'critical',
          label: 'High Memory Usage',
          value: `${memUsage.toFixed(1)}%`,
          icon: AlertCircle,
        });
      }
    }
    
    return metrics;
  }, [consensus, performance, os]);
  
  // System status summary
  const systemStatus = useMemo(() => {
    if (!health) return 'unknown';
    if (health.overall >= 90) return 'healthy';
    if (health.overall >= 70) return 'degraded';
    if (health.overall >= 50) return 'warning';
    return 'critical';
  }, [health]);
  
  const getStatusConfig = (status: string) => {
    switch (status) {
      case 'healthy':
        return { color: 'text-green-500 bg-green-50', icon: CheckCircle, text: 'All Systems Operational' };
      case 'degraded':
        return { color: 'text-yellow-500 bg-yellow-50', icon: AlertTriangle, text: 'Performance Degraded' };
      case 'warning':
        return { color: 'text-orange-500 bg-orange-50', icon: AlertCircle, text: 'Issues Detected' };
      case 'critical':
        return { color: 'text-red-500 bg-red-50', icon: XCircle, text: 'Critical Issues' };
      default:
        return { color: 'text-gray-500 bg-gray-50', icon: Activity, text: 'Status Unknown' };
    }
  };
  
  const statusConfig = getStatusConfig(systemStatus);
  const StatusIcon = statusConfig.icon;
  
  if (!health) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-96">
          <p className="text-muted-foreground">Initializing health monitoring...</p>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <div className="space-y-4">
      {/* System Status Alert */}
      {systemStatus !== 'healthy' && (
        <Alert className={statusConfig.color}>
          <StatusIcon className="h-4 w-4" />
          <AlertTitle>{statusConfig.text}</AlertTitle>
          <AlertDescription>
            {health.issues.length > 0 && (
              <ul className="mt-2 space-y-1">
                {health.issues.slice(0, 3).map((issue, index) => (
                  <li key={index} className="text-sm">• {issue}</li>
                ))}
              </ul>
            )}
          </AlertDescription>
        </Alert>
      )}
      
      {/* Health Score Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Overall Health */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Heart className="w-5 h-5" />
              Overall Health
            </CardTitle>
            <CardDescription>System health score</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <RadialBarChart cx="50%" cy="50%" innerRadius="60%" outerRadius="90%" data={overallHealthData}>
                <RadialBar dataKey="value" cornerRadius={10} fill="#8884d8" />
                <PolarGrid />
                <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" className="fill-foreground">
                  <tspan fontSize="36" fontWeight="bold">{health.overall}</tspan>
                  <tspan fontSize="20" x="50%" dy="30">%</tspan>
                </text>
              </RadialBarChart>
            </ResponsiveContainer>
            
            <div className="mt-4 flex items-center justify-center">
              <Badge variant={health.overall >= 80 ? 'default' : health.overall >= 60 ? 'secondary' : 'destructive'}>
                {statusConfig.text}
              </Badge>
            </div>
          </CardContent>
        </Card>
        
        {/* Component Health Radar */}
        <Card>
          <CardHeader>
            <CardTitle>Component Health</CardTitle>
            <CardDescription>Individual component status</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <RadarChart data={healthMetrics}>
                <PolarGrid />
                <PolarAngleAxis dataKey="metric" />
                <PolarRadiusAxis angle={90} domain={[0, 100]} />
                <Radar name="Health" dataKey="value" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        
        {/* Critical Metrics */}
        <Card>
          <CardHeader>
            <CardTitle>Critical Metrics</CardTitle>
            <CardDescription>Metrics requiring attention</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {criticalMetrics.length === 0 ? (
              <div className="flex items-center justify-center h-[250px]">
                <div className="text-center">
                  <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-2" />
                  <p className="text-sm text-muted-foreground">No critical issues</p>
                </div>
              </div>
            ) : (
              criticalMetrics.map((metric, index) => {
                const Icon = metric.icon;
                return (
                  <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-muted">
                    <div className="flex items-center gap-3">
                      <Icon className={`w-5 h-5 ${metric.type === 'critical' ? 'text-red-500' : 'text-yellow-500'}`} />
                      <div>
                        <p className="text-sm font-medium">{metric.label}</p>
                        <p className="text-xs text-muted-foreground">{metric.value}</p>
                      </div>
                    </div>
                    <Badge variant={metric.type === 'critical' ? 'destructive' : 'secondary'}>
                      {metric.type}
                    </Badge>
                  </div>
                );
              })
            )}
          </CardContent>
        </Card>
      </div>
      
      {/* Health Components Grid */}
      <Card>
        <CardHeader>
          <CardTitle>Health Breakdown</CardTitle>
          <CardDescription>Detailed health scores for each component</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(health.components).map(([component, score]) => {
              const icons: Record<string, any> = {
                consensus: Layers,
                performance: Zap,
                rpc: Server,
                network: Network,
                system: Cpu,
                jito: Activity,
                geyser: Database,
                security: Shield,
              };
              const Icon = icons[component] || Activity;
              const scoreColor = score >= 80 ? 'text-green-500' : score >= 60 ? 'text-yellow-500' : 'text-red-500';
              
              return (
                <div key={component} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Icon className="w-4 h-4 text-muted-foreground" />
                      <span className="text-sm font-medium capitalize">{component}</span>
                    </div>
                    <span className={`text-sm font-bold ${scoreColor}`}>{score}%</span>
                  </div>
                  <Progress value={score} className="h-2" />
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
      
      {/* Recommendations */}
      {health.recommendations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              Recommendations
            </CardTitle>
            <CardDescription>Suggested optimizations for better performance</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {health.recommendations.map((recommendation, index) => (
                <div key={index} className="flex items-start gap-3 p-3 rounded-lg bg-muted">
                  <Badge variant="outline" className="mt-0.5">{index + 1}</Badge>
                  <p className="text-sm">{recommendation}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
      
      {/* Active Alerts Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5" />
              Alert Summary
            </span>
            <div className="flex gap-2">
              {['critical', 'high', 'medium', 'low'].map((severity) => {
                const count = activeAlerts.filter(a => a.severity === severity && !a.acknowledged).length;
                if (count === 0) return null;
                
                return (
                  <Badge
                    key={severity}
                    variant={severity === 'critical' ? 'destructive' : severity === 'high' ? 'secondary' : 'outline'}
                  >
                    {count} {severity}
                  </Badge>
                );
              })}
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <p className="text-2xl font-bold text-red-500">
                {activeAlerts.filter(a => a.severity === 'critical').length}
              </p>
              <p className="text-xs text-muted-foreground">Critical</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-orange-500">
                {activeAlerts.filter(a => a.severity === 'high').length}
              </p>
              <p className="text-xs text-muted-foreground">High</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-yellow-500">
                {activeAlerts.filter(a => a.severity === 'medium').length}
              </p>
              <p className="text-xs text-muted-foreground">Medium</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-blue-500">
                {activeAlerts.filter(a => a.severity === 'low').length}
              </p>
              <p className="text-xs text-muted-foreground">Low</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
});

HealthOverview.displayName = 'HealthOverview';