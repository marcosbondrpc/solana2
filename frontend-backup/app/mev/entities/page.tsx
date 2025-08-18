'use client';

import { useState, Suspense } from 'react';
import { EntityProfileCard } from '@/components/mev/entity-profile-card';
import { FleetNetworkGraph } from '@/components/mev/fleet-network-graph';
import { WalletRotationTimeline } from '@/components/mev/wallet-rotation-timeline';
import { BehavioralAnalysis } from '@/components/mev/behavioral-analysis';
import { SearchBar } from '@/components/ui/search-bar';
import { LoadingState } from '@/components/ui/loading-state';
import { Badge } from '@/components/ui/badge';

export default function EntitiesPage() {
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  
  return (
    <div className="p-6 space-y-6">
      {/* Page Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Entity Profiles</h1>
          <p className="text-muted-foreground mt-1">
            Behavioral analysis and wallet clustering for MEV operators
          </p>
        </div>
        <div className="flex gap-4">
          <SearchBar 
            placeholder="Search by address or entity..."
            value={searchQuery}
            onChange={setSearchQuery}
          />
        </div>
      </div>
      
      {/* Entity Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Total Entities</div>
          <div className="text-2xl font-bold">1,247</div>
          <Badge variant="secondary" className="mt-2">+23 today</Badge>
        </div>
        <div className="glass rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Empire Class</div>
          <div className="text-2xl font-bold text-mev-empire">12</div>
          <div className="text-xs text-muted-foreground mt-1">High sophistication</div>
        </div>
        <div className="glass rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Warlord Class</div>
          <div className="text-2xl font-bold text-mev-warlord">89</div>
          <div className="text-xs text-muted-foreground mt-1">Regional dominance</div>
        </div>
        <div className="glass rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Guerrilla Class</div>
          <div className="text-2xl font-bold text-mev-guerrilla">1,146</div>
          <div className="text-xs text-muted-foreground mt-1">Opportunistic</div>
        </div>
      </div>
      
      {/* Main Content Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left Column - Entity List */}
        <div className="space-y-4">
          <div className="glass rounded-lg p-4">
            <h2 className="text-xl font-semibold mb-4">Top Entities</h2>
            <Suspense fallback={<LoadingState />}>
              <div className="space-y-2">
                {/* Entity list will be populated from API */}
                <EntityProfileCard 
                  entityId="0x1234...5678"
                  archetype="empire"
                  profitability={2847392}
                  successRate={0.73}
                  onClick={() => setSelectedEntity('0x1234...5678')}
                  isSelected={selectedEntity === '0x1234...5678'}
                />
                <EntityProfileCard 
                  entityId="0xabcd...ef01"
                  archetype="warlord"
                  profitability={982374}
                  successRate={0.61}
                  onClick={() => setSelectedEntity('0xabcd...ef01')}
                  isSelected={selectedEntity === '0xabcd...ef01'}
                />
              </div>
            </Suspense>
          </div>
        </div>
        
        {/* Middle Column - Network Graph */}
        <div className="xl:col-span-2 space-y-6">
          <div className="glass rounded-lg p-4 h-[500px]">
            <h2 className="text-xl font-semibold mb-4">Fleet Network Visualization</h2>
            <Suspense fallback={<LoadingState />}>
              <FleetNetworkGraph selectedEntity={selectedEntity} />
            </Suspense>
          </div>
          
          {/* Bottom Row - Analysis */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="glass rounded-lg p-4">
              <h2 className="text-xl font-semibold mb-4">Behavioral Patterns</h2>
              <Suspense fallback={<LoadingState />}>
                <BehavioralAnalysis entityId={selectedEntity} />
              </Suspense>
            </div>
            
            <div className="glass rounded-lg p-4">
              <h2 className="text-xl font-semibold mb-4">Wallet Rotation</h2>
              <Suspense fallback={<LoadingState />}>
                <WalletRotationTimeline entityId={selectedEntity} />
              </Suspense>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}