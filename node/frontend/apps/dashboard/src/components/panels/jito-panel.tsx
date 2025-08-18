'use client';

import { memo } from 'react';
import { useMonitoringStore } from '@/lib/monitoring-store';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card';

export const JitoPanel = memo(() => {
  const jito = useMonitoringStore((state) => state.jito);
  
  if (!jito) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-96">
          <p className="text-muted-foreground">Waiting for Jito MEV data...</p>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Jito MEV Monitoring</CardTitle>
          <CardDescription>Bundle statistics and MEV rewards</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">Jito MEV monitoring panel implementation</p>
        </CardContent>
      </Card>
    </div>
  );
});

JitoPanel.displayName = 'JitoPanel';