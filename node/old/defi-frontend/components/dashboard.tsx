'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { MetricsGrid } from '@/components/metrics-grid';
import { SystemMonitor } from '@/components/system-monitor';
import { JitoMonitor } from '@/components/jito-monitor';
import { RPCMonitor } from '@/components/rpc-monitor';
import { LogViewer } from '@/components/log-viewer';
import { CommandPanel } from '@/components/command-panel';
import { NetworkStatus } from '@/components/network-status';
import { ChartContainer } from '@/components/chart-container';
import { useNodeStore } from '@/lib/store';
import {
  Activity,
  Cpu,
  HardDrive,
  Network,
  Zap,
  Server,
  FileText,
  Terminal,
  RefreshCw,
} from 'lucide-react';

export function Dashboard() {
  const [activeSection, setActiveSection] = useState('overview');
  const nodeMetrics = useNodeStore((state) => state.nodeMetrics);
  const systemMetrics = useNodeStore((state) => state.systemMetrics);
  const jitoMetrics = useNodeStore((state) => state.jitoMetrics);
  const rpcMetrics = useNodeStore((state) => state.rpcMetrics);

  const sections = [
    { id: 'overview', label: 'Overview', icon: Activity },
    { id: 'system', label: 'System', icon: Cpu },
    { id: 'network', label: 'Network', icon: Network },
    { id: 'jito', label: 'Jito MEV', icon: Zap },
    { id: 'rpc', label: 'RPC', icon: Server },
    { id: 'logs', label: 'Logs', icon: FileText },
    { id: 'commands', label: 'Commands', icon: Terminal },
  ];

  return (
    <div className="space-y-6">
      {/* Section Tabs */}
      <div className="flex space-x-2 overflow-x-auto pb-2">
        {sections.map((section) => {
          const Icon = section.icon;
          return (
            <Button
              key={section.id}
              variant={activeSection === section.id ? 'default' : 'outline'}
              onClick={() => setActiveSection(section.id)}
              className="flex items-center space-x-2"
            >
              <Icon className="h-4 w-4" />
              <span>{section.label}</span>
            </Button>
          );
        })}
      </div>

      {/* Overview Section */}
      {activeSection === 'overview' && (
        <div className="space-y-6">
          <MetricsGrid />
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ChartContainer
              title="Slot Progression"
              description="Real-time slot tracking"
              dataKey="slotHistory"
              color="hsl(var(--primary))"
            />
            <ChartContainer
              title="TPS History"
              description="Transactions per second"
              dataKey="tpsHistory"
              color="hsl(var(--accent))"
            />
          </div>
        </div>
      )}

      {/* System Section */}
      {activeSection === 'system' && <SystemMonitor />}

      {/* Network Section */}
      {activeSection === 'network' && <NetworkStatus />}

      {/* Jito Section */}
      {activeSection === 'jito' && <JitoMonitor />}

      {/* RPC Section */}
      {activeSection === 'rpc' && <RPCMonitor />}

      {/* Logs Section */}
      {activeSection === 'logs' && <LogViewer />}

      {/* Commands Section */}
      {activeSection === 'commands' && <CommandPanel />}
    </div>
  );
}