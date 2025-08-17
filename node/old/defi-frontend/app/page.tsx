'use client';

import { useEffect } from 'react';
import dynamic from 'next/dynamic';

// Dynamically import the monitoring dashboard for better code splitting
const MonitoringDashboard = dynamic(
  () => import('@/components/monitoring-dashboard'),
  {
    loading: () => (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading Solana Node Monitor...</p>
        </div>
      </div>
    ),
    ssr: false, // Disable SSR for WebSocket components
  }
);

export default function Home() {
  useEffect(() => {
    // Request notification permission on load
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }

    // Set up performance monitoring
    if ('performance' in window && 'measure' in window.performance) {
      performance.mark('dashboard-init');
    }
  }, []);

  return <MonitoringDashboard />;
}