'use client';

import { Suspense } from 'react';
import { LatencyDistributionChart } from '@/components/mev/latency-distribution-chart';
import { BundleAdjacencyRate } from '@/components/mev/bundle-adjacency-rate';
import { TipIntensityHeatmap } from '@/components/mev/tip-intensity-heatmap';
import { VenueAgilityTimeline } from '@/components/mev/venue-agility-timeline';
import { PercentileCards } from '@/components/mev/percentile-cards';
import { LoadingState } from '@/components/ui/loading-state';
import { TimeRangeSelector } from '@/components/ui/time-range-selector';

export default function LatencyPage() {
  return (
    <div className="p-6 space-y-6">
      {/* Page Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Latency Analysis</h1>
          <p className="text-muted-foreground mt-1">
            Deep dive into MEV operator performance characteristics
          </p>
        </div>
        <TimeRangeSelector />
      </div>
      
      {/* Percentile Overview */}
      <Suspense fallback={<LoadingState />}>
        <PercentileCards />
      </Suspense>
      
      {/* Main Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Latency Distribution */}
        <div className="glass rounded-lg p-4">
          <h2 className="text-xl font-semibold mb-4">Latency Distribution</h2>
          <Suspense fallback={<LoadingState />}>
            <LatencyDistributionChart />
          </Suspense>
        </div>
        
        {/* Bundle Adjacency */}
        <div className="glass rounded-lg p-4">
          <h2 className="text-xl font-semibold mb-4">Bundle Adjacency Rate</h2>
          <Suspense fallback={<LoadingState />}>
            <BundleAdjacencyRate />
          </Suspense>
        </div>
        
        {/* Tip Intensity Heatmap */}
        <div className="glass rounded-lg p-4 xl:col-span-2">
          <h2 className="text-xl font-semibold mb-4">Tip Intensity Analysis</h2>
          <Suspense fallback={<LoadingState />}>
            <TipIntensityHeatmap />
          </Suspense>
        </div>
        
        {/* Venue Agility */}
        <div className="glass rounded-lg p-4 xl:col-span-2">
          <h2 className="text-xl font-semibold mb-4">Venue Agility Timeline</h2>
          <Suspense fallback={<LoadingState />}>
            <VenueAgilityTimeline />
          </Suspense>
        </div>
      </div>
    </div>
  );
}