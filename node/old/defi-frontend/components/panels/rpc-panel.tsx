'use client';

import { memo } from 'react';
import { useMonitoringStore } from '@/lib/monitoring-store';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export const RPCPanel = memo(() => {
  const rpc = useMonitoringStore((state) => state.rpcLayer);
  
  if (!rpc) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-96">
          <p className="text-muted-foreground">Waiting for RPC data...</p>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>RPC Layer Monitoring</CardTitle>
          <CardDescription>Real-time RPC endpoint performance and health</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">RPC monitoring panel implementation</p>
        </CardContent>
      </Card>
    </div>
  );
});

RPCPanel.displayName = 'RPCPanel';