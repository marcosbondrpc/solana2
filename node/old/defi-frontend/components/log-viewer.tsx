'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useNodeStore } from '@/lib/store';
import { cn } from '@/lib/utils';
import { format } from 'date-fns';
import {
  AlertCircle,
  AlertTriangle,
  Info,
  Bug,
  Trash2,
  Download,
  Filter,
  ChevronDown,
} from 'lucide-react';

export function LogViewer() {
  const logs = useNodeStore((state) => state.logs);
  const clearLogs = useNodeStore((state) => state.clearLogs);
  const [filter, setFilter] = useState<'all' | 'info' | 'warn' | 'error' | 'debug'>('all');
  const [autoScroll, setAutoScroll] = useState(true);
  const logContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  const filteredLogs = filter === 'all' 
    ? logs 
    : logs.filter(log => log.level === filter);

  const getLogIcon = (level: string) => {
    switch (level) {
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'warn':
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'info':
        return <Info className="h-4 w-4 text-blue-500" />;
      case 'debug':
        return <Bug className="h-4 w-4 text-gray-500" />;
      default:
        return null;
    }
  };

  const getLogColor = (level: string) => {
    switch (level) {
      case 'error':
        return 'text-red-500 bg-red-500/10 border-red-500/20';
      case 'warn':
        return 'text-yellow-500 bg-yellow-500/10 border-yellow-500/20';
      case 'info':
        return 'text-blue-500 bg-blue-500/10 border-blue-500/20';
      case 'debug':
        return 'text-gray-500 bg-gray-500/10 border-gray-500/20';
      default:
        return '';
    }
  };

  const downloadLogs = () => {
    const logText = filteredLogs
      .map((log) => `[${format(log.timestamp, 'yyyy-MM-dd HH:mm:ss')}] [${log.level.toUpperCase()}] [${log.source}] ${log.message}`)
      .join('\n');
    
    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `solana-logs-${format(new Date(), 'yyyy-MM-dd-HHmmss')}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <Card className="h-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Log Viewer</CardTitle>
            <CardDescription>
              Real-time node logs ({filteredLogs.length} entries)
            </CardDescription>
          </div>
          <div className="flex items-center space-x-2">
            {/* Filter Buttons */}
            <div className="flex items-center space-x-1 bg-muted rounded-lg p-1">
              {(['all', 'info', 'warn', 'error', 'debug'] as const).map((level) => (
                <Button
                  key={level}
                  variant={filter === level ? 'secondary' : 'ghost'}
                  size="sm"
                  onClick={() => setFilter(level)}
                  className="capitalize"
                >
                  {level === 'all' ? 'All' : level}
                </Button>
              ))}
            </div>

            {/* Auto Scroll Toggle */}
            <Button
              variant={autoScroll ? 'secondary' : 'outline'}
              size="sm"
              onClick={() => setAutoScroll(!autoScroll)}
            >
              <ChevronDown className="h-4 w-4 mr-1" />
              Auto Scroll
            </Button>

            {/* Download Logs */}
            <Button
              variant="outline"
              size="sm"
              onClick={downloadLogs}
              disabled={filteredLogs.length === 0}
            >
              <Download className="h-4 w-4 mr-1" />
              Export
            </Button>

            {/* Clear Logs */}
            <Button
              variant="outline"
              size="sm"
              onClick={clearLogs}
              disabled={logs.length === 0}
            >
              <Trash2 className="h-4 w-4 mr-1" />
              Clear
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div
          ref={logContainerRef}
          className="h-[600px] overflow-y-auto font-mono text-sm space-y-1 p-4 bg-muted/30 rounded-lg"
        >
          {filteredLogs.length === 0 ? (
            <div className="text-center text-muted-foreground py-8">
              No logs to display
            </div>
          ) : (
            filteredLogs.map((log, index) => (
              <div
                key={index}
                className={cn(
                  'flex items-start space-x-2 p-2 rounded border',
                  getLogColor(log.level)
                )}
              >
                <div className="flex-shrink-0 mt-0.5">
                  {getLogIcon(log.level)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2 text-xs text-muted-foreground mb-1">
                    <span>{format(log.timestamp, 'HH:mm:ss.SSS')}</span>
                    <span className="uppercase font-semibold">{log.level}</span>
                    <span className="text-xs bg-background/50 px-1 rounded">
                      {log.source}
                    </span>
                  </div>
                  <div className="break-all whitespace-pre-wrap">
                    {log.message}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
}