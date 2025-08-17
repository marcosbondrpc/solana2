/**
 * Ultra-Optimized Detection Store
 * Manages MEV detection state with circular buffers and memory efficiency
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

interface SandwichAttack {
  id: string;
  timestamp: number;
  victimTx: string;
  frontrunTx: string;
  backrunTx: string;
  confidence: number;
  profitUSD: number;
  victimLoss: number;
  attackerAddress: string;
  dexPair: string;
  blockNumber: number;
  gasUsed: number;
  status: 'detected' | 'confirmed' | 'failed';
}

interface EntityArchetype {
  address: string;
  type: 'Empire' | 'Warlord' | 'Guerrilla' | 'Phantom';
  metrics: {
    speed: number;
    volume: number;
    sophistication: number;
    aggression: number;
    efficiency: number;
    stealth: number;
  };
  lastSeen: number;
  totalProfit: number;
  attackCount: number;
}

interface LatencyMetrics {
  timestamp: number;
  p50: number;
  p95: number;
  p99: number;
  max: number;
  samples: number;
}

interface EconomicImpact {
  totalExtracted: number;
  victimCount: number;
  avgLossPerVictim: number;
  networkImpactPercent: number;
  gasOverhead: number;
  failureRate: number;
}

// Circular buffer implementation for memory efficiency
class CircularBuffer<T> {
  private buffer: T[];
  private pointer: number;
  private size: number;
  private count: number;
  
  constructor(size: number) {
    this.size = size;
    this.buffer = new Array(size);
    this.pointer = 0;
    this.count = 0;
  }
  
  push(item: T) {
    this.buffer[this.pointer] = item;
    this.pointer = (this.pointer + 1) % this.size;
    if (this.count < this.size) this.count++;
  }
  
  getAll(): T[] {
    if (this.count < this.size) {
      return this.buffer.slice(0, this.count);
    }
    
    // Return items in correct order
    return [
      ...this.buffer.slice(this.pointer),
      ...this.buffer.slice(0, this.pointer)
    ].filter(item => item !== undefined);
  }
  
  getLast(n: number): T[] {
    const all = this.getAll();
    return all.slice(-n);
  }
  
  clear() {
    this.buffer = new Array(this.size);
    this.pointer = 0;
    this.count = 0;
  }
  
  get length() {
    return this.count;
  }
}

interface DetectionState {
  // Sandwich attacks with circular buffer
  sandwichBuffer: CircularBuffer<SandwichAttack>;
  recentSandwiches: SandwichAttack[];
  
  // Entity tracking
  entities: Map<string, EntityArchetype>;
  entityStats: {
    empireCount: number;
    warlordCount: number;
    guerrillaCount: number;
    phantomCount: number;
  };
  
  // Latency tracking
  latencyBuffer: CircularBuffer<LatencyMetrics>;
  currentLatency: LatencyMetrics | null;
  
  // Economic impact
  economicImpact: EconomicImpact;
  impactHistory: CircularBuffer<EconomicImpact>;
  
  // Performance metrics
  performanceMetrics: {
    fps: number;
    memoryUsage: number;
    cpuUsage: number;
    renderTime: number;
  };
  
  // Settings
  settings: {
    maxSandwiches: number;
    maxEntities: number;
    updateInterval: number;
    autoScroll: boolean;
    alertsEnabled: boolean;
  };
}

interface DetectionActions {
  // Sandwich detection
  addSandwichAttack: (attack: SandwichAttack) => void;
  batchAddSandwiches: (attacks: SandwichAttack[]) => void;
  confirmSandwich: (id: string) => void;
  
  // Entity management
  updateEntity: (entity: EntityArchetype) => void;
  removeEntity: (address: string) => void;
  
  // Latency updates
  updateLatency: (metrics: LatencyMetrics) => void;
  
  // Economic impact
  updateEconomicImpact: (impact: Partial<EconomicImpact>) => void;
  
  // Performance
  updatePerformanceMetrics: (metrics: Partial<DetectionState['performanceMetrics']>) => void;
  
  // Settings
  updateSettings: (settings: Partial<DetectionState['settings']>) => void;
  
  // Utilities
  clearOldData: () => void;
  reset: () => void;
}

export const useDetectionStore = create<DetectionState & DetectionActions>()(
  subscribeWithSelector(
    immer((set, get) => ({
      // Initial state
      sandwichBuffer: new CircularBuffer(10000),
      recentSandwiches: [],
      
      entities: new Map(),
      entityStats: {
        empireCount: 0,
        warlordCount: 0,
        guerrillaCount: 0,
        phantomCount: 0
      },
      
      latencyBuffer: new CircularBuffer(1000),
      currentLatency: null,
      
      economicImpact: {
        totalExtracted: 0,
        victimCount: 0,
        avgLossPerVictim: 0,
        networkImpactPercent: 0,
        gasOverhead: 0,
        failureRate: 0
      },
      
      impactHistory: new CircularBuffer(100),
      
      performanceMetrics: {
        fps: 60,
        memoryUsage: 0,
        cpuUsage: 0,
        renderTime: 0
      },
      
      settings: {
        maxSandwiches: 10000,
        maxEntities: 5000,
        updateInterval: 100,
        autoScroll: true,
        alertsEnabled: true
      },
      
      // Actions
      addSandwichAttack: (attack) => set((state) => {
        state.sandwichBuffer.push(attack);
        state.recentSandwiches = state.sandwichBuffer.getLast(100);
        
        // Update economic impact
        state.economicImpact.totalExtracted += attack.profitUSD;
        state.economicImpact.victimCount++;
        state.economicImpact.avgLossPerVictim = 
          (state.economicImpact.avgLossPerVictim * (state.economicImpact.victimCount - 1) + attack.victimLoss) / 
          state.economicImpact.victimCount;
        
        // Update entity if exists
        if (state.entities.has(attack.attackerAddress)) {
          const entity = state.entities.get(attack.attackerAddress)!;
          entity.totalProfit += attack.profitUSD;
          entity.attackCount++;
          entity.lastSeen = Date.now();
        }
      }),
      
      batchAddSandwiches: (attacks) => set((state) => {
        attacks.forEach(attack => {
          state.sandwichBuffer.push(attack);
        });
        state.recentSandwiches = state.sandwichBuffer.getLast(100);
        
        // Batch update economic impact
        const totalProfit = attacks.reduce((sum, a) => sum + a.profitUSD, 0);
        const totalLoss = attacks.reduce((sum, a) => sum + a.victimLoss, 0);
        
        state.economicImpact.totalExtracted += totalProfit;
        state.economicImpact.victimCount += attacks.length;
        state.economicImpact.avgLossPerVictim = 
          (state.economicImpact.avgLossPerVictim * (state.economicImpact.victimCount - attacks.length) + totalLoss) / 
          state.economicImpact.victimCount;
      }),
      
      confirmSandwich: (id) => set((state) => {
        const sandwiches = state.sandwichBuffer.getAll();
        const sandwich = sandwiches.find(s => s.id === id);
        if (sandwich) {
          sandwich.status = 'confirmed';
        }
      }),
      
      updateEntity: (entity) => set((state) => {
        const existing = state.entities.get(entity.address);
        
        if (!existing || state.entities.size < state.settings.maxEntities) {
          state.entities.set(entity.address, entity);
          
          // Update stats
          const counts = { Empire: 0, Warlord: 0, Guerrilla: 0, Phantom: 0 };
          state.entities.forEach(e => {
            counts[e.type]++;
          });
          
          state.entityStats = {
            empireCount: counts.Empire,
            warlordCount: counts.Warlord,
            guerrillaCount: counts.Guerrilla,
            phantomCount: counts.Phantom
          };
        }
      }),
      
      removeEntity: (address) => set((state) => {
        state.entities.delete(address);
      }),
      
      updateLatency: (metrics) => set((state) => {
        state.latencyBuffer.push(metrics);
        state.currentLatency = metrics;
      }),
      
      updateEconomicImpact: (impact) => set((state) => {
        Object.assign(state.economicImpact, impact);
        state.impactHistory.push({ ...state.economicImpact });
      }),
      
      updatePerformanceMetrics: (metrics) => set((state) => {
        Object.assign(state.performanceMetrics, metrics);
      }),
      
      updateSettings: (settings) => set((state) => {
        Object.assign(state.settings, settings);
      }),
      
      clearOldData: () => set((state) => {
        // Remove old entities
        const cutoff = Date.now() - 3600000; // 1 hour
        state.entities.forEach((entity, address) => {
          if (entity.lastSeen < cutoff) {
            state.entities.delete(address);
          }
        });
      }),
      
      reset: () => set((state) => {
        state.sandwichBuffer.clear();
        state.recentSandwiches = [];
        state.entities.clear();
        state.latencyBuffer.clear();
        state.impactHistory.clear();
        state.currentLatency = null;
        state.economicImpact = {
          totalExtracted: 0,
          victimCount: 0,
          avgLossPerVictim: 0,
          networkImpactPercent: 0,
          gasOverhead: 0,
          failureRate: 0
        };
      })
    }))
  )
);

// Selectors for optimized access
export const useSandwiches = () => useDetectionStore(state => state.recentSandwiches);
export const useEntities = () => useDetectionStore(state => Array.from(state.entities.values()));
export const useLatency = () => useDetectionStore(state => state.currentLatency);
export const useEconomicImpact = () => useDetectionStore(state => state.economicImpact);
export const usePerformanceMetrics = () => useDetectionStore(state => state.performanceMetrics);

// Performance monitor
if (typeof window !== 'undefined') {
  let frameCount = 0;
  let lastTime = performance.now();
  
  const measureFPS = () => {
    frameCount++;
    const currentTime = performance.now();
    
    if (currentTime >= lastTime + 1000) {
      const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
      
      useDetectionStore.getState().updatePerformanceMetrics({ fps });
      
      frameCount = 0;
      lastTime = currentTime;
    }
    
    requestAnimationFrame(measureFPS);
  };
  
  measureFPS();
  
  // Memory monitoring
  if ('memory' in performance) {
    setInterval(() => {
      const memory = (performance as any).memory;
      const memoryUsage = memory.usedJSHeapSize / memory.jsHeapSizeLimit * 100;
      
      useDetectionStore.getState().updatePerformanceMetrics({ memoryUsage });
    }, 5000);
  }
}