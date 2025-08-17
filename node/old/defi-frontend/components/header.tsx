'use client';

import { useState } from 'react';
import { useTheme } from 'next-themes';
import {
  Menu,
  Moon,
  Sun,
  Wifi,
  WifiOff,
  Activity,
  Settings,
  Terminal,
  Zap,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useNodeStore } from '@/lib/store';
import { cn } from '@/lib/utils';

interface HeaderProps {
  onToggleSidebar: () => void;
}

export function Header({ onToggleSidebar }: HeaderProps) {
  const { theme, setTheme } = useTheme();
  const isConnected = useNodeStore((state) => state.isConnected);
  const network = useNodeStore((state) => state.network);
  const nodeMetrics = useNodeStore((state) => state.nodeMetrics);

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex h-16 items-center px-4">
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleSidebar}
          className="mr-4"
        >
          <Menu className="h-5 w-5" />
        </Button>

        <div className="flex items-center space-x-4 flex-1">
          <div className="flex items-center space-x-2">
            <div className="relative">
              <Zap className="h-8 w-8 text-solana-purple" />
              <div className="absolute -bottom-1 -right-1 h-3 w-3 rounded-full bg-solana-green animate-pulse" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-solana-purple to-solana-green bg-clip-text text-transparent">
                Solana Node Dashboard
              </h1>
              <p className="text-xs text-muted-foreground">
                Ultra Performance Monitor
              </p>
            </div>
          </div>

          <div className="flex-1" />

          {/* Network Status */}
          <div className="flex items-center space-x-2 px-3 py-1 rounded-lg bg-muted">
            <div
              className={cn(
                'h-2 w-2 rounded-full',
                isConnected ? 'bg-green-500' : 'bg-red-500'
              )}
            />
            <span className="text-sm font-medium capitalize">{network}</span>
          </div>

          {/* Node Status */}
          {nodeMetrics && (
            <div className="flex items-center space-x-4 px-3 py-1 rounded-lg bg-muted">
              <div className="flex items-center space-x-2">
                <Activity className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm">
                  Slot: <span className="font-mono font-medium">{nodeMetrics.slot.toLocaleString()}</span>
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <div
                  className={cn(
                    'h-2 w-2 rounded-full',
                    nodeMetrics.health === 'healthy' ? 'bg-green-500' :
                    nodeMetrics.health === 'warning' ? 'bg-yellow-500' :
                    'bg-red-500'
                  )}
                />
                <span className="text-sm capitalize">{nodeMetrics.health}</span>
              </div>
            </div>
          )}

          {/* Connection Status */}
          <div className="flex items-center space-x-2">
            {isConnected ? (
              <Wifi className="h-5 w-5 text-green-500" />
            ) : (
              <WifiOff className="h-5 w-5 text-red-500" />
            )}
          </div>

          {/* Terminal Button */}
          <Button variant="outline" size="icon">
            <Terminal className="h-5 w-5" />
          </Button>

          {/* Settings Button */}
          <Button variant="outline" size="icon">
            <Settings className="h-5 w-5" />
          </Button>

          {/* Theme Toggle */}
          <Button
            variant="outline"
            size="icon"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          >
            <Sun className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
            <Moon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
            <span className="sr-only">Toggle theme</span>
          </Button>
        </div>
      </div>
    </header>
  );
}