'use client';

import { memo } from 'react';
import { useMonitoringStore } from '@/lib/monitoring-store';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export const GeyserPanel = memo(() => {
  const geyser = useMonitoringStore((state) => state.geyser);
  
  if (!geyser) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-96">
          <p className="text-muted-foreground">Waiting for Geyser plugin data...</p>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Geyser Plugin Monitoring</CardTitle>
          <CardDescription>Plugin status and streaming metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">Geyser plugin monitoring panel implementation</p>
        </CardContent>
      </Card>
    </div>
  );
});

GeyserPanel.displayName = 'GeyserPanel';