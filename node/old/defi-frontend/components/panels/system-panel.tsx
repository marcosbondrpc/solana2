'use client';

import { memo } from 'react';
import { useMonitoringStore } from '@/lib/monitoring-store';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export const SystemPanel = memo(() => {
  const os = useMonitoringStore((state) => state.os);
  
  if (!os) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-96">
          <p className="text-muted-foreground">Waiting for system data...</p>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>System Monitoring</CardTitle>
          <CardDescription>CPU, memory, disk, and OS metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">System monitoring panel implementation</p>
        </CardContent>
      </Card>
    </div>
  );
});

SystemPanel.displayName = 'SystemPanel';