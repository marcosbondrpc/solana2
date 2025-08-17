'use client';

import { useState } from 'react';
import { cn } from '@/lib/utils';
import {
  Activity,
  AlertCircle,
  BarChart3,
  Blocks,
  Cpu,
  Database,
  FileText,
  Gauge,
  HardDrive,
  Home,
  Layers,
  Network,
  Play,
  RefreshCw,
  Server,
  Settings,
  Shield,
  Terminal,
  Zap,
} from 'lucide-react';
import { Button } from '@/components/ui/button';

interface SidebarProps {
  open: boolean;
}

const menuItems = [
  { icon: Home, label: 'Overview', href: '#overview' },
  { icon: Activity, label: 'Node Status', href: '#status' },
  { icon: BarChart3, label: 'Metrics', href: '#metrics' },
  { icon: Cpu, label: 'System Resources', href: '#resources' },
  { icon: Network, label: 'Network', href: '#network' },
  { icon: Zap, label: 'Jito MEV', href: '#jito' },
  { icon: Server, label: 'RPC Monitor', href: '#rpc' },
  { icon: Blocks, label: 'Block Production', href: '#blocks' },
  { icon: FileText, label: 'Logs', href: '#logs' },
  { icon: Terminal, label: 'Commands', href: '#commands' },
  { icon: Shield, label: 'Security', href: '#security' },
  { icon: Settings, label: 'Settings', href: '#settings' },
];

export function Sidebar({ open }: SidebarProps) {
  const [activeItem, setActiveItem] = useState('Overview');

  return (
    <aside
      className={cn(
        'fixed left-0 top-16 h-[calc(100vh-4rem)] bg-background border-r transition-all duration-300 z-40',
        open ? 'w-64' : 'w-16'
      )}
    >
      <div className="flex flex-col h-full">
        <nav className="flex-1 space-y-1 p-2">
          {menuItems.map((item) => {
            const Icon = item.icon;
            return (
              <Button
                key={item.label}
                variant={activeItem === item.label ? 'secondary' : 'ghost'}
                className={cn(
                  'w-full justify-start',
                  !open && 'justify-center px-2'
                )}
                onClick={() => setActiveItem(item.label)}
              >
                <Icon className={cn('h-5 w-5', open && 'mr-3')} />
                {open && <span>{item.label}</span>}
              </Button>
            );
          })}
        </nav>

        <div className="border-t p-2">
          <Button
            variant="outline"
            className={cn(
              'w-full justify-start',
              !open && 'justify-center px-2'
            )}
          >
            <RefreshCw className={cn('h-5 w-5', open && 'mr-3')} />
            {open && <span>Refresh</span>}
          </Button>
        </div>
      </div>
    </aside>
  );
}