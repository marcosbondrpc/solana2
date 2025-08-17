'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import { executeCommand } from '@/lib/api';
import {
  Play,
  RefreshCw,
  Settings,
  Shield,
  Trash2,
  Upload,
  Download,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Loader2,
} from 'lucide-react';
import type { Command } from '@/types';

const commands: Command[] = [
  // Deploy commands
  {
    id: 'start-validator',
    name: 'Start Validator',
    description: 'Start the Solana validator with optimized settings',
    category: 'deploy',
    script: '/home/kidgordones/0solana/node/scripts/deploy/start-validator-main.sh',
  },
  {
    id: 'start-rpc',
    name: 'Start RPC Node',
    description: 'Start minimal RPC node configuration',
    category: 'deploy',
    script: '/home/kidgordones/0solana/node/scripts/deploy/start-rpc-minimal.sh',
  },
  {
    id: 'restart-optimized',
    name: 'Restart Optimized',
    description: 'Restart node with performance optimizations',
    category: 'deploy',
    script: '/home/kidgordones/0solana/node/scripts/deploy/restart-node-optimized.sh',
    requiresConfirmation: true,
  },
  
  // Monitor commands
  {
    id: 'monitor-basic',
    name: 'Basic Monitor',
    description: 'Show basic node monitoring stats',
    category: 'monitor',
    script: '/home/kidgordones/0solana/node/scripts/monitor/monitor-basic.sh',
  },
  {
    id: 'monitor-performance',
    name: 'Performance Monitor',
    description: 'Show detailed performance metrics',
    category: 'monitor',
    script: '/home/kidgordones/0solana/node/scripts/monitor/monitor-performance.sh',
  },
  
  // Maintain commands
  {
    id: 'clean-logs',
    name: 'Clean Logs',
    description: 'Clean old log files to free disk space',
    category: 'maintain',
    script: '/home/kidgordones/0solana/node/scripts/maintain/clean-logs.sh',
  },
  {
    id: 'update-node',
    name: 'Update Node',
    description: 'Update Solana node to latest version',
    category: 'maintain',
    script: '/home/kidgordones/0solana/node/scripts/maintain/update-node.sh',
    requiresConfirmation: true,
    dangerous: true,
  },
  
  // Optimize commands
  {
    id: 'optimize-system',
    name: 'Optimize System',
    description: 'Apply ultra system optimizations',
    category: 'optimize',
    script: '/home/kidgordones/0solana/node/scripts/optimize/optimize-system-ultra.sh',
    requiresConfirmation: true,
  },
  {
    id: 'optimize-validator',
    name: 'Optimize Validator',
    description: 'Apply top 1 validator optimizations',
    category: 'optimize',
    script: '/home/kidgordones/0solana/node/scripts/optimize/optimize-validator-top1.sh',
    requiresConfirmation: true,
  },
  
  // Utils commands
  {
    id: 'backup-keys',
    name: 'Backup Keys',
    description: 'Backup validator keys securely',
    category: 'utils',
    script: '/home/kidgordones/0solana/node/scripts/utils/backup-keys.sh',
  },
  {
    id: 'check-health',
    name: 'Check Health',
    description: 'Comprehensive health check',
    category: 'utils',
    script: '/home/kidgordones/0solana/node/scripts/utils/check-health.sh',
  },
];

export function CommandPanel() {
  const { toast } = useToast();
  const [executing, setExecuting] = useState<string | null>(null);
  const [results, setResults] = useState<Record<string, any>>({});

  const handleExecute = async (command: Command) => {
    if (command.requiresConfirmation) {
      const confirmed = window.confirm(
        `Are you sure you want to execute "${command.name}"? ${
          command.dangerous ? 'This action is potentially dangerous!' : ''
        }`
      );
      if (!confirmed) return;
    }

    setExecuting(command.id);
    
    try {
      const result = await executeCommand(command.script);
      
      setResults((prev) => ({
        ...prev,
        [command.id]: result,
      }));
      
      toast({
        title: result.success ? 'Command Executed' : 'Command Failed',
        description: `${command.name} ${result.success ? 'completed successfully' : 'failed'}`,
        variant: result.success ? 'default' : 'destructive',
      });
    } catch (error) {
      toast({
        title: 'Execution Error',
        description: 'Failed to execute command',
        variant: 'destructive',
      });
    } finally {
      setExecuting(null);
    }
  };

  const categories = [...new Set(commands.map((c) => c.category))];

  return (
    <div className="space-y-6">
      {categories.map((category) => (
        <Card key={category}>
          <CardHeader>
            <CardTitle className="capitalize">{category} Commands</CardTitle>
            <CardDescription>
              Execute {category} operations on your Solana node
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {commands
                .filter((cmd) => cmd.category === category)
                .map((command) => {
                  const result = results[command.id];
                  const isExecuting = executing === command.id;
                  
                  return (
                    <Card key={command.id} className="relative">
                      <CardHeader className="pb-3">
                        <div className="flex items-start justify-between">
                          <div className="space-y-1">
                            <CardTitle className="text-sm font-medium">
                              {command.name}
                            </CardTitle>
                            <CardDescription className="text-xs">
                              {command.description}
                            </CardDescription>
                          </div>
                          {command.dangerous && (
                            <AlertTriangle className="h-4 w-4 text-yellow-500" />
                          )}
                        </div>
                      </CardHeader>
                      <CardContent className="pb-3">
                        <Button
                          onClick={() => handleExecute(command)}
                          disabled={isExecuting}
                          variant={command.dangerous ? 'destructive' : 'default'}
                          size="sm"
                          className="w-full"
                        >
                          {isExecuting ? (
                            <>
                              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                              Executing...
                            </>
                          ) : (
                            <>
                              <Play className="mr-2 h-4 w-4" />
                              Execute
                            </>
                          )}
                        </Button>
                        
                        {result && (
                          <div className="mt-2 p-2 rounded bg-muted">
                            <div className="flex items-center space-x-2">
                              {result.success ? (
                                <CheckCircle className="h-4 w-4 text-green-500" />
                              ) : (
                                <XCircle className="h-4 w-4 text-red-500" />
                              )}
                              <span className="text-xs">
                                {result.success ? 'Success' : 'Failed'}
                              </span>
                            </div>
                            {result.duration && (
                              <div className="text-xs text-muted-foreground mt-1">
                                Duration: {result.duration}ms
                              </div>
                            )}
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  );
                })}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}