'use client';

import { memo } from 'react';
import { useMonitoringStore } from '@/lib/monitoring-store';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card';

export const NetworkPanel = memo(() => {
  const network = useMonitoringStore((state) => state.network);
  
  if (!network) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-96">
          <p className="text-muted-foreground">Waiting for network data...</p>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Network Monitoring</CardTitle>
          <CardDescription>Gossip, TPU, TVU, and repair connections</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">Network monitoring panel implementation</p>
        </CardContent>
      </Card>
    </div>
  );
});

NetworkPanel.displayName = 'NetworkPanel';