'use client';

import { memo } from 'react';
import { useMonitoringStore } from '@/lib/monitoring-store';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card';

export const AlertsPanel = memo(() => {
  const alerts = useMonitoringStore((state) => state.activeAlerts);
  
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Alert Management</CardTitle>
          <CardDescription>Active alerts and notification settings</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            {alerts.length} active alert(s) - Alert panel implementation
          </p>
        </CardContent>
      </Card>
    </div>
  );
});

AlertsPanel.displayName = 'AlertsPanel';