'use client';

import { memo } from 'react';
import { useMonitoringStore } from '@/lib/monitoring-store';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export const SecurityPanel = memo(() => {
  const security = useMonitoringStore((state) => state.security);
  
  if (!security) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-96">
          <p className="text-muted-foreground">Waiting for security data...</p>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Security Monitoring</CardTitle>
          <CardDescription>Key management, firewall, and audit logs</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">Security monitoring panel implementation</p>
        </CardContent>
      </Card>
    </div>
  );
});

SecurityPanel.displayName = 'SecurityPanel';