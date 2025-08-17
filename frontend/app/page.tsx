'use client';

import dynamic from 'next/dynamic';
import { Suspense } from 'react';

// Dynamically import the ultra-performance dashboard
const UltraPerformanceDashboard = dynamic(
  () => import('../components/dashboards/UltraPerformanceDashboard'),
  {
    loading: () => (
      <div className="h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          <div className="mt-4 text-gray-400">Loading Ultra Performance Dashboard...</div>
        </div>
      </div>
    ),
    ssr: false
  }
);

export default function Page() {
  return (
    <Suspense fallback={
      <div className="h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-white">Initializing MEV Infrastructure...</div>
      </div>
    }>
      <UltraPerformanceDashboard />
    </Suspense>
  );
}
