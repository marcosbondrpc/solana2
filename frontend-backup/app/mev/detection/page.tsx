'use client';

import { Suspense } from 'react';
import { SandwichDetectionFeed } from '@/components/mev/SandwichDetectionFeed';
import { ArchetypeRadar } from '@/components/mev/ArchetypeRadar';
import { EconomicImpactGauge } from '@/components/mev/EconomicImpactGauge';
import { LatencyHistogram } from '@/components/mev/LatencyHistogram';
import { MetricsOverview } from '@/components/mev/MetricsOverview';
import { PerformanceMonitor } from '@/components/mev/PerformanceMonitor';
import { TimeRangeSelector } from '@/components/ui/TimeRangeSelector';
import { AlertBanner } from '@/components/ui/AlertBanner';
import { LoadingState } from '@/components/ui/LoadingState';

export default function DetectionPage() {
  return (
    <>
      <div className="p-6 space-y-6">
        {/* Security Alert Banner */}
        <AlertBanner 
          type="info"
          message="DEFENSIVE MONITORING ONLY - This dashboard is for security research and MEV detection. No execution capabilities."
        />
        
        {/* Page Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold">MEV Detection Overview</h1>
            <p className="text-muted-foreground mt-1">
              Real-time sandwich attack detection and entity behavior analysis
            </p>
          </div>
          <TimeRangeSelector />
        </div>
      
      {/* Metrics Overview Cards */}
      <Suspense fallback={<LoadingState />}>
        <MetricsOverview />
      </Suspense>
      
      {/* Main Grid Layout */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left Column - Detection Feed */}
        <div className="xl:col-span-2 space-y-6">
          <div className="glass rounded-lg p-4">
            <h2 className="text-xl font-semibold mb-4">Live Detection Feed</h2>
            <Suspense fallback={<LoadingState />}>
              <SandwichDetectionFeed />
            </Suspense>
          </div>
          
          {/* Latency Analysis */}
          <div className="glass rounded-lg p-4">
            <h2 className="text-xl font-semibold mb-4">Latency Distribution</h2>
            <Suspense fallback={<LoadingState />}>
              <LatencyHistogram />
            </Suspense>
          </div>
        </div>
        
        {/* Right Column - Analytics */}
        <div className="space-y-6">
          {/* Archetype Classification */}
          <div className="glass rounded-lg p-4">
            <h2 className="text-xl font-semibold mb-4">Entity Archetypes</h2>
            <Suspense fallback={<LoadingState />}>
              <ArchetypeRadar />
            </Suspense>
          </div>
          
          {/* Economic Impact */}
          <div className="glass rounded-lg p-4">
            <h2 className="text-xl font-semibold mb-4">Economic Impact</h2>
            <Suspense fallback={<LoadingState />}>
              <EconomicImpactGauge />
            </Suspense>
          </div>
        </div>
      </div>
      </div>
      
      {/* Performance Monitor - Always visible */}
      <Suspense fallback={null}>
        <PerformanceMonitor />
      </Suspense>
    </>
  );
}