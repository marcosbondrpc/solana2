'use client';

import { useEffect, useState, useMemo, useCallback, memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useMonitoringStore, selectHealthScore, selectCriticalAlerts } from '@/lib/monitoring-store';
import { getWebSocketService } from '@/services/websocket-service';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  Cpu,
  Database,
  HardDrive,
  Heart,
  Network,
  Server,
  Shield,
  Zap,
  TrendingUp,
  TrendingDown,
  Clock,
  Layers,
  Package,
  Lock,
  Bell,
  Settings,
  RefreshCw,
  Download,
  Upload,
  AlertCircle,
  XCircle,
} from 'lucide-react';

// Import sub-components
import { ConsensusPanel } from './panels/consensus-panel';
import { PerformancePanel } from './panels/performance-panel';
import { RPCPanel } from './panels/rpc-panel';
import { NetworkPanel } from './panels/network-panel';
import { SystemPanel } from './panels/system-panel';
import { JitoPanel } from './panels/jito-panel';
import { GeyserPanel } from './panels/geyser-panel';
import { SecurityPanel } from './panels/security-panel';
import { ControlPanel } from './panels/control-panel';
import { AlertsPanel } from './panels/alerts-panel';
import { HealthOverview } from './panels/health-overview';

const MonitoringDashboard = memo(() => {
  const [activeTab, setActiveTab] = useState('overview');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  
  // Store subscriptions
  const connectionStatus = useMonitoringStore((state) => state.connectionStatus);
  const healthScore = useMonitoringStore(selectHealthScore);
  const criticalAlerts = useMonitoringStore(selectCriticalAlerts);
  const consensus = useMonitoringStore((state) => state.consensus);
  const performance = useMonitoringStore((state) => state.performance);
  const config = useMonitoringStore((state) => state.config);
  
  // Initialize WebSocket connection
  useEffect(() => {
    const ws = getWebSocketService();
    ws.connect();
    
    return () => {
      if (!autoRefresh) {
        ws.disconnect();
      }
    };
  }, [autoRefresh]);
  
  // Request notification permissions
  useEffect(() => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, []);
  
  // Fullscreen toggle
  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);
  
  // Connection status indicator
  const connectionIndicator = useMemo(() => {
    const statusConfig = {
      connected: { color: 'bg-green-500', text: 'Connected', icon: CheckCircle },
      connecting: { color: 'bg-yellow-500', text: 'Connecting', icon: RefreshCw },
      disconnected: { color: 'bg-gray-500', text: 'Disconnected', icon: XCircle },
      error: { color: 'bg-red-500', text: 'Error', icon: AlertCircle },
    };
    
    const config = statusConfig[connectionStatus];
    const Icon = config.icon;
    
    return (
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${config.color} animate-pulse`} />
        <Icon className="w-4 h-4" />
        <span className="text-sm font-medium">{config.text}</span>
      </div>
    );
  }, [connectionStatus]);
  
  // Health status badge
  const healthBadge = useMemo(() => {
    const getHealthColor = (score: number) => {
      if (score >= 90) return 'bg-green-500';
      if (score >= 70) return 'bg-yellow-500';
      if (score >= 50) return 'bg-orange-500';
      return 'bg-red-500';
    };
    
    return (
      <div className="flex items-center gap-2">
        <Heart className="w-4 h-4" />
        <Progress value={healthScore} className="w-20 h-2" />
        <span className="text-sm font-bold">{healthScore}%</span>
        <div className={`w-2 h-2 rounded-full ${getHealthColor(healthScore)}`} />
      </div>
    );
  }, [healthScore]);
  
  // Critical alerts banner
  const alertsBanner = useMemo(() => {
    if (criticalAlerts.length === 0) return null;
    
    return (
      <Alert variant="destructive" className="mb-4">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Critical Alerts</AlertTitle>
        <AlertDescription>
          {criticalAlerts.length} critical issue(s) require immediate attention.
          <Button
            variant="link"
            size="sm"
            className="ml-2"
            onClick={() => setActiveTab('alerts')}
          >
            View Details
          </Button>
        </AlertDescription>
      </Alert>
    );
  }, [criticalAlerts]);
  
  // Tab configuration
  const tabs = [
    { id: 'overview', label: 'Overview', icon: Activity },
    { id: 'consensus', label: 'Consensus', icon: Layers },
    { id: 'performance', label: 'Performance', icon: TrendingUp },
    { id: 'rpc', label: 'RPC Layer', icon: Server },
    { id: 'network', label: 'Network', icon: Network },
    { id: 'system', label: 'System', icon: Cpu },
    { id: 'jito', label: 'Jito MEV', icon: Zap },
    { id: 'geyser', label: 'Geyser', icon: Database },
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'control', label: 'Control', icon: Settings },
    { id: 'alerts', label: 'Alerts', icon: Bell, badge: criticalAlerts.length },
  ];
  
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold">Solana Node Monitor</h1>
            {connectionIndicator}
            {healthBadge}
          </div>
          
          <div className="flex items-center gap-2">
            <Button
              variant={autoRefresh ? 'default' : 'outline'}
              size="sm"
              onClick={() => setAutoRefresh(!autoRefresh)}
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
              Auto Refresh
            </Button>
            
            <Button
              variant="outline"
              size="sm"
              onClick={() => useMonitoringStore.getState().createSnapshot()}
            >
              <Download className="w-4 h-4 mr-2" />
              Snapshot
            </Button>
            
            <Button
              variant="outline"
              size="sm"
              onClick={toggleFullscreen}
            >
              {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
            </Button>
          </div>
        </div>
      </div>
      
      {/* Alerts Banner */}
      <div className="container mt-4">
        {alertsBanner}
      </div>
      
      {/* Main Content */}
      <div className="container py-4">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          {/* Tab Navigation */}
          <TabsList className="grid grid-cols-11 gap-2 h-auto p-1">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <TabsTrigger
                  key={tab.id}
                  value={tab.id}
                  className="relative flex flex-col items-center gap-1 py-2"
                >
                  <Icon className="w-4 h-4" />
                  <span className="text-xs">{tab.label}</span>
                  {tab.badge && tab.badge > 0 && (
                    <Badge
                      variant="destructive"
                      className="absolute -top-1 -right-1 h-4 w-4 p-0 text-[10px]"
                    >
                      {tab.badge}
                    </Badge>
                  )}
                </TabsTrigger>
              );
            })}
          </TabsList>
          
          {/* Tab Content with Animation */}
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
            >
              <TabsContent value="overview" className="space-y-4">
                <HealthOverview />
                <QuickStats />
                <RecentActivity />
              </TabsContent>
              
              <TabsContent value="consensus">
                <ConsensusPanel />
              </TabsContent>
              
              <TabsContent value="performance">
                <PerformancePanel />
              </TabsContent>
              
              <TabsContent value="rpc">
                <RPCPanel />
              </TabsContent>
              
              <TabsContent value="network">
                <NetworkPanel />
              </TabsContent>
              
              <TabsContent value="system">
                <SystemPanel />
              </TabsContent>
              
              <TabsContent value="jito">
                <JitoPanel />
              </TabsContent>
              
              <TabsContent value="geyser">
                <GeyserPanel />
              </TabsContent>
              
              <TabsContent value="security">
                <SecurityPanel />
              </TabsContent>
              
              <TabsContent value="control">
                <ControlPanel />
              </TabsContent>
              
              <TabsContent value="alerts">
                <AlertsPanel />
              </TabsContent>
            </motion.div>
          </AnimatePresence>
        </Tabs>
      </div>
    </div>
  );
});

// Quick Stats Component
const QuickStats = memo(() => {
  const consensus = useMonitoringStore((state) => state.consensus);
  const performance = useMonitoringStore((state) => state.performance);
  const jito = useMonitoringStore((state) => state.jito);
  const os = useMonitoringStore((state) => state.os);
  
  const stats = [
    {
      label: 'Current Slot',
      value: consensus?.slot.toLocaleString() ?? '-',
      icon: Clock,
      trend: 'up',
    },
    {
      label: 'TPS',
      value: performance?.currentTPS.toLocaleString() ?? '-',
      icon: TrendingUp,
      trend: performance?.currentTPS > performance?.averageTPS ? 'up' : 'down',
    },
    {
      label: 'Skip Rate',
      value: `${consensus?.skipRate.toFixed(2) ?? '-'}%`,
      icon: AlertTriangle,
      trend: consensus?.skipRate > 5 ? 'down' : 'up',
    },
    {
      label: 'MEV Rewards',
      value: `◎${jito?.profitability.dailyEarnings.toFixed(2) ?? '-'}`,
      icon: Zap,
      trend: 'up',
    },
    {
      label: 'CPU Usage',
      value: `${((os?.memoryDetails.used ?? 0) / (os?.memoryDetails.used + os?.memoryDetails.free ?? 1) * 100).toFixed(1)}%`,
      icon: Cpu,
      trend: 'stable',
    },
    {
      label: 'Network I/O',
      value: `${((os?.networkInterfaces.eth0?.rxBytes ?? 0) / 1e9).toFixed(2)} GB`,
      icon: Network,
      trend: 'up',
    },
  ];
  
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      {stats.map((stat, index) => {
        const Icon = stat.icon;
        return (
          <Card key={index}>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <Icon className="w-4 h-4 text-muted-foreground" />
                {stat.trend === 'up' && <TrendingUp className="w-3 h-3 text-green-500" />}
                {stat.trend === 'down' && <TrendingDown className="w-3 h-3 text-red-500" />}
              </div>
              <div className="mt-2">
                <p className="text-2xl font-bold">{stat.value}</p>
                <p className="text-xs text-muted-foreground">{stat.label}</p>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
});

// Recent Activity Component
const RecentActivity = memo(() => {
  const logs = useMonitoringStore((state) => state.security?.auditLog.slice(0, 5) ?? []);
  
  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Activity</CardTitle>
        <CardDescription>Latest system events and actions</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {logs.map((log, index) => (
            <div key={index} className="flex items-center justify-between p-2 rounded-lg hover:bg-muted">
              <div className="flex items-center gap-3">
                {log.result === 'success' ? (
                  <CheckCircle className="w-4 h-4 text-green-500" />
                ) : (
                  <XCircle className="w-4 h-4 text-red-500" />
                )}
                <div>
                  <p className="text-sm font-medium">{log.action}</p>
                  <p className="text-xs text-muted-foreground">
                    {log.user} • {new Date(log.timestamp).toLocaleTimeString()}
                  </p>
                </div>
              </div>
              <Badge variant={log.result === 'success' ? 'default' : 'destructive'}>
                {log.result}
              </Badge>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
});

MonitoringDashboard.displayName = 'MonitoringDashboard';
QuickStats.displayName = 'QuickStats';
RecentActivity.displayName = 'RecentActivity';

export default MonitoringDashboard;