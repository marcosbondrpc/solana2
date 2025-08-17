/**
 * Bandit Store for Thompson Sampling MEV optimization
 * Tracks arm performance, budget allocation, and canary results
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { devtools } from 'zustand/middleware';

export interface BanditArm {
  id: string;
  route: 'direct' | 'jito' | 'hedged';
  alpha: number; // Success count + 1
  beta: number;  // Failure count + 1
  sampledProbability: number;
  expectedValue: number;
  totalPulls: number;
  successfulPulls: number;
  totalProfit: number;
  averageLatency: number;
  lastUpdate: number;
  weekOverWeekGrowth: number;
  confidence: number;
  budget: number;
  budgetUtilization: number;
}

export interface CanaryTransaction {
  id: string;
  timestamp: number;
  armId: string;
  route: string;
  amount: number;
  success: boolean;
  latency: number;
  landingSlot?: number;
  profit?: number;
  gasUsed?: number;
  errorReason?: string;
}

export interface DecisionDNA {
  id: string;
  timestamp: number;
  opportunityId: string;
  fingerprint: string; // 256-bit hex string
  features: {
    tokenPair: string;
    volumeClass: 'micro' | 'small' | 'medium' | 'large' | 'whale';
    marketCondition: 'stable' | 'volatile' | 'trending';
    competitionLevel: 'low' | 'medium' | 'high';
    timeOfDay: number;
    networkCongestion: number;
    gasPrice: number;
    profitMargin: number;
    slippage: number;
    routeComplexity: number;
  };
  selectedArm: string;
  outcome: 'success' | 'failure' | 'pending';
}

export interface BudgetAllocation {
  timestamp: number;
  totalBudget: number;
  allocations: Map<string, {
    armId: string;
    allocated: number;
    spent: number;
    remaining: number;
    efficiency: number;
  }>;
  rebalanceFrequency: number;
  lastRebalance: number;
}

export interface ThompsonSamplingState {
  epoch: number;
  temperature: number; // Exploration vs exploitation
  learningRate: number;
  decayRate: number;
  minSamples: number;
  confidenceThreshold: number;
  explorationBonus: number;
}

interface BanditState {
  // Core data
  arms: Map<string, BanditArm>;
  canaryTransactions: CanaryTransaction[];
  decisionDNAs: Map<string, DecisionDNA>;
  budgetAllocation: BudgetAllocation;
  samplingState: ThompsonSamplingState;
  
  // Performance metrics
  totalDecisions: number;
  successfulDecisions: number;
  averageRegret: number;
  cumulativeReward: number;
  explorationRate: number;
  
  // Actions
  updateArm: (armId: string, update: Partial<BanditArm>) => void;
  recordCanary: (canary: CanaryTransaction) => void;
  recordDecision: (dna: DecisionDNA) => void;
  sampleArm: () => BanditArm | null;
  updateBudget: (allocation: Partial<BudgetAllocation>) => void;
  updateSamplingState: (state: Partial<ThompsonSamplingState>) => void;
  calculateExpectedValues: () => void;
  rebalanceBudget: () => void;
  getArmStats: (armId: string) => any;
  getBestArm: () => BanditArm | null;
  reset: () => void;
}

const initialState = {
  arms: new Map([
    ['direct', {
      id: 'direct',
      route: 'direct' as const,
      alpha: 1,
      beta: 1,
      sampledProbability: 0.5,
      expectedValue: 0,
      totalPulls: 0,
      successfulPulls: 0,
      totalProfit: 0,
      averageLatency: 0,
      lastUpdate: Date.now(),
      weekOverWeekGrowth: 0,
      confidence: 0,
      budget: 0,
      budgetUtilization: 0
    }],
    ['jito', {
      id: 'jito',
      route: 'jito' as const,
      alpha: 1,
      beta: 1,
      sampledProbability: 0.5,
      expectedValue: 0,
      totalPulls: 0,
      successfulPulls: 0,
      totalProfit: 0,
      averageLatency: 0,
      lastUpdate: Date.now(),
      weekOverWeekGrowth: 0,
      confidence: 0,
      budget: 0,
      budgetUtilization: 0
    }],
    ['hedged', {
      id: 'hedged',
      route: 'hedged' as const,
      alpha: 1,
      beta: 1,
      sampledProbability: 0.5,
      expectedValue: 0,
      totalPulls: 0,
      successfulPulls: 0,
      totalProfit: 0,
      averageLatency: 0,
      lastUpdate: Date.now(),
      weekOverWeekGrowth: 0,
      confidence: 0,
      budget: 0,
      budgetUtilization: 0
    }]
  ]),
  canaryTransactions: [],
  decisionDNAs: new Map(),
  budgetAllocation: {
    timestamp: Date.now(),
    totalBudget: 1000,
    allocations: new Map(),
    rebalanceFrequency: 3600000, // 1 hour
    lastRebalance: Date.now()
  },
  samplingState: {
    epoch: 0,
    temperature: 1.0,
    learningRate: 0.1,
    decayRate: 0.99,
    minSamples: 10,
    confidenceThreshold: 0.95,
    explorationBonus: 0.1
  },
  totalDecisions: 0,
  successfulDecisions: 0,
  averageRegret: 0,
  cumulativeReward: 0,
  explorationRate: 0.3
};

// Beta distribution sampling for Thompson Sampling
function sampleBeta(alpha: number, beta: number): number {
  // Using Kumaraswamy approximation for performance
  const x = Math.random();
  const a = alpha;
  const b = beta;
  return Math.pow(x, 1/a) * Math.pow(1 - Math.pow(x, 1/a), 1/b - 1);
}

export const useBanditStore = create<BanditState>()(
  devtools(
    subscribeWithSelector(
      immer((set, get) => ({
        ...initialState,
        
        updateArm: (armId, update) => {
          set((state) => {
            const arm = state.arms.get(armId);
            if (arm) {
              Object.assign(arm, update);
              arm.lastUpdate = Date.now();
              
              // Update expected value
              arm.expectedValue = arm.alpha / (arm.alpha + arm.beta);
              
              // Update confidence (using Wilson score interval)
              const n = arm.totalPulls;
              const p = arm.successfulPulls / Math.max(1, n);
              const z = 1.96; // 95% confidence
              arm.confidence = n > 0 
                ? (p + z*z/(2*n) - z * Math.sqrt((p*(1-p) + z*z/(4*n))/n)) / (1 + z*z/n)
                : 0;
            }
          });
        },
        
        recordCanary: (canary) => {
          set((state) => {
            state.canaryTransactions.push(canary);
            
            // Keep only last 1000 canaries
            if (state.canaryTransactions.length > 1000) {
              state.canaryTransactions.shift();
            }
            
            // Update arm statistics
            const arm = state.arms.get(canary.armId);
            if (arm) {
              arm.totalPulls++;
              if (canary.success) {
                arm.successfulPulls++;
                arm.alpha++;
              } else {
                arm.beta++;
              }
              
              // Update profit and latency
              if (canary.profit) {
                arm.totalProfit += canary.profit;
              }
              
              // Update average latency (exponential moving average)
              arm.averageLatency = arm.averageLatency * 0.9 + canary.latency * 0.1;
              
              // Resample probability
              arm.sampledProbability = sampleBeta(arm.alpha, arm.beta);
            }
          });
        },
        
        recordDecision: (dna) => {
          set((state) => {
            state.decisionDNAs.set(dna.id, dna);
            state.totalDecisions++;
            
            if (dna.outcome === 'success') {
              state.successfulDecisions++;
            }
            
            // Keep only last 10000 DNAs
            if (state.decisionDNAs.size > 10000) {
              const firstKey = state.decisionDNAs.keys().next().value;
              state.decisionDNAs.delete(firstKey);
            }
          });
        },
        
        sampleArm: () => {
          const state = get();
          const arms = Array.from(state.arms.values());
          
          // Thompson Sampling: sample from each arm's Beta distribution
          let bestArm: BanditArm | null = null;
          let bestSample = -1;
          
          for (const arm of arms) {
            // Apply exploration bonus for under-sampled arms
            const explorationBonus = arm.totalPulls < state.samplingState.minSamples 
              ? state.samplingState.explorationBonus 
              : 0;
            
            const sample = sampleBeta(arm.alpha, arm.beta) + explorationBonus;
            
            if (sample > bestSample) {
              bestSample = sample;
              bestArm = arm;
            }
          }
          
          return bestArm;
        },
        
        updateBudget: (allocation) => {
          set((state) => {
            Object.assign(state.budgetAllocation, allocation);
            state.budgetAllocation.timestamp = Date.now();
          });
        },
        
        updateSamplingState: (samplingState) => {
          set((state) => {
            Object.assign(state.samplingState, samplingState);
            
            // Decay temperature over time
            state.samplingState.temperature *= state.samplingState.decayRate;
            state.samplingState.epoch++;
          });
        },
        
        calculateExpectedValues: () => {
          set((state) => {
            for (const [id, arm] of state.arms) {
              // Calculate expected value considering multiple factors
              const successRate = arm.alpha / (arm.alpha + arm.beta);
              const profitPerPull = arm.totalProfit / Math.max(1, arm.totalPulls);
              const latencyPenalty = Math.max(0, 1 - arm.averageLatency / 100); // Penalty for latency > 100ms
              
              arm.expectedValue = successRate * profitPerPull * latencyPenalty;
            }
          });
        },
        
        rebalanceBudget: () => {
          set((state) => {
            const totalBudget = state.budgetAllocation.totalBudget;
            const arms = Array.from(state.arms.values());
            
            // Calculate weights based on expected values
            const totalWeight = arms.reduce((sum, arm) => sum + Math.max(0.1, arm.expectedValue), 0);
            
            state.budgetAllocation.allocations.clear();
            
            for (const arm of arms) {
              const weight = Math.max(0.1, arm.expectedValue) / totalWeight;
              const allocated = totalBudget * weight;
              
              state.budgetAllocation.allocations.set(arm.id, {
                armId: arm.id,
                allocated,
                spent: 0,
                remaining: allocated,
                efficiency: arm.totalProfit / Math.max(1, allocated)
              });
              
              arm.budget = allocated;
            }
            
            state.budgetAllocation.lastRebalance = Date.now();
          });
        },
        
        getArmStats: (armId) => {
          const state = get();
          const arm = state.arms.get(armId);
          if (!arm) return null;
          
          const recentCanaries = state.canaryTransactions
            .filter(c => c.armId === armId)
            .slice(-100);
          
          const recentSuccess = recentCanaries.filter(c => c.success).length;
          const recentTotal = recentCanaries.length;
          
          return {
            ...arm,
            recentSuccessRate: recentTotal > 0 ? recentSuccess / recentTotal : 0,
            recentCanaries,
            allocation: state.budgetAllocation.allocations.get(armId)
          };
        },
        
        getBestArm: () => {
          const state = get();
          let bestArm: BanditArm | null = null;
          let bestValue = -1;
          
          for (const arm of state.arms.values()) {
            if (arm.expectedValue > bestValue) {
              bestValue = arm.expectedValue;
              bestArm = arm;
            }
          }
          
          return bestArm;
        },
        
        reset: () => {
          set(initialState);
        }
      }))
    ),
    {
      name: 'bandit-store'
    }
  )
);

// Selectors
export const selectArmPerformance = (state: BanditState) => 
  Array.from(state.arms.values()).map(arm => ({
    id: arm.id,
    route: arm.route,
    successRate: arm.alpha / (arm.alpha + arm.beta),
    expectedValue: arm.expectedValue,
    pulls: arm.totalPulls,
    profit: arm.totalProfit,
    latency: arm.averageLatency,
    confidence: arm.confidence,
    budget: arm.budget
  }));

export const selectRecentCanaries = (state: BanditState) =>
  state.canaryTransactions.slice(-20);

export const selectDecisionDNAPatterns = (state: BanditState) => {
  const patterns = new Map<string, { count: number; successRate: number }>();
  
  for (const dna of state.decisionDNAs.values()) {
    const key = `${dna.features.tokenPair}-${dna.features.volumeClass}-${dna.features.marketCondition}`;
    
    if (!patterns.has(key)) {
      patterns.set(key, { count: 0, successRate: 0 });
    }
    
    const pattern = patterns.get(key)!;
    pattern.count++;
    if (dna.outcome === 'success') {
      pattern.successRate = (pattern.successRate * (pattern.count - 1) + 1) / pattern.count;
    } else {
      pattern.successRate = (pattern.successRate * (pattern.count - 1)) / pattern.count;
    }
  }
  
  return Array.from(patterns.entries())
    .map(([key, value]) => ({ pattern: key, ...value }))
    .sort((a, b) => b.count - a.count);
};