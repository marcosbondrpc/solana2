/**
 * Bandit Store - Thompson Sampling & Multi-Armed Bandit Management
 * Handles adaptive optimization with microsecond precision
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { subscribeWithSelector } from 'zustand/middleware';
import { apiService } from '../lib/api-service';

export interface BanditArm {
  id: string;
  route: string;
  alpha: number;
  beta: number;
  pulls: number;
  rewards: number;
  failures: number;
  ucb_score: number;
  expected_value: number;
  confidence_interval: [number, number];
  last_pull: number;
  avg_reward: number;
  variance: number;
  thompson_sample?: number;
}

export interface BanditPerformance {
  exploration_rate: number;
  exploitation_rate: number;
  total_pulls: number;
  total_rewards: number;
  convergence_metric: number;
  regret: number;
  cumulative_regret: number;
  optimal_arm_pulls: number;
  suboptimal_pulls: number;
  switching_cost: number;
}

export interface BanditPolicy {
  type: 'thompson' | 'ucb' | 'epsilon_greedy' | 'exp3';
  epsilon?: number;
  c?: number; // UCB exploration parameter
  gamma?: number; // EXP3 parameter
  decay_rate?: number;
  min_exploration?: number;
}

export interface RoutePerformance {
  route_id: string;
  success_rate: number;
  avg_latency_ms: number;
  p50_latency_ms: number;
  p99_latency_ms: number;
  total_volume: number;
  profit_realized: number;
  gas_efficiency: number;
}

export interface AdaptiveThreshold {
  metric: string;
  current_threshold: number;
  min_threshold: number;
  max_threshold: number;
  adaptation_rate: number;
  last_updated: number;
}

interface BanditState {
  // Bandit Arms
  arms: Map<string, BanditArm>;
  activeArms: string[];
  bestArm: string | null;
  
  // Performance Metrics
  performance: BanditPerformance;
  routePerformance: Map<string, RoutePerformance>;
  
  // Policies
  currentPolicy: BanditPolicy;
  policyHistory: Array<{
    policy: BanditPolicy;
    timestamp: number;
    performance: BanditPerformance;
  }>;
  
  // Adaptive Thresholds
  adaptiveThresholds: Map<string, AdaptiveThreshold>;
  
  // Historical Data
  pullHistory: Array<{
    arm_id: string;
    timestamp: number;
    reward: number;
    context?: any;
  }>;
  rewardDistribution: Map<string, number[]>;
  
  // Real-time Metrics
  currentExplorationRate: number;
  optimalArmProbability: Map<string, number>;
  armSelectionCounts: Map<string, number>;
  
  // Convergence Tracking
  convergenceHistory: Array<{
    timestamp: number;
    metric: number;
    dominant_arm: string;
  }>;
  isConverged: boolean;
  convergenceThreshold: number;
  
  // Actions
  pullArm: (armId: string) => void;
  updateReward: (armId: string, reward: number) => void;
  selectArm: () => string;
  thompsonSample: () => string;
  calculateUCB: (armId: string) => number;
  updatePolicy: (policy: BanditPolicy) => void;
  addArm: (arm: BanditArm) => void;
  removeArm: (armId: string) => void;
  resetArm: (armId: string) => void;
  calculateRegret: () => number;
  updateAdaptiveThreshold: (metric: string, performance: number) => void;
  loadMetrics: () => Promise<void>;
  analyzeConvergence: () => void;
  getArmStatistics: (armId: string) => any;
  exportPerformanceReport: () => any;
}

// Beta distribution sampling for Thompson Sampling
function sampleBeta(alpha: number, beta: number): number {
  // Approximate Beta sampling using Gamma distribution ratio
  const sampleGamma = (shape: number): number => {
    let sum = 0;
    for (let i = 0; i < shape; i++) {
      sum -= Math.log(Math.random());
    }
    return sum;
  };
  
  const x = sampleGamma(alpha);
  const y = sampleGamma(beta);
  return x / (x + y);
}

export const useBanditStore = create<BanditState>()(
  subscribeWithSelector(
    immer((set, get) => ({
      // Initial state
      arms: new Map(),
      activeArms: [],
      bestArm: null,
      performance: {
        exploration_rate: 0.1,
        exploitation_rate: 0.9,
        total_pulls: 0,
        total_rewards: 0,
        convergence_metric: 0,
        regret: 0,
        cumulative_regret: 0,
        optimal_arm_pulls: 0,
        suboptimal_pulls: 0,
        switching_cost: 0
      },
      routePerformance: new Map(),
      currentPolicy: {
        type: 'thompson',
        epsilon: 0.1,
        c: 2,
        gamma: 0.1,
        decay_rate: 0.999,
        min_exploration: 0.01
      },
      policyHistory: [],
      adaptiveThresholds: new Map(),
      pullHistory: [],
      rewardDistribution: new Map(),
      currentExplorationRate: 0.1,
      optimalArmProbability: new Map(),
      armSelectionCounts: new Map(),
      convergenceHistory: [],
      isConverged: false,
      convergenceThreshold: 0.95,

      // Actions
      pullArm: (armId: string) => {
        set(state => {
          const arm = state.arms.get(armId);
          if (arm) {
            arm.pulls++;
            arm.last_pull = Date.now();
            
            // Update selection counts
            const count = state.armSelectionCounts.get(armId) || 0;
            state.armSelectionCounts.set(armId, count + 1);
            
            // Update performance
            state.performance.total_pulls++;
            
            // Add to history
            state.pullHistory.push({
              arm_id: armId,
              timestamp: Date.now(),
              reward: 0 // Will be updated when reward comes in
            });
            
            // Trim history if too large
            if (state.pullHistory.length > 10000) {
              state.pullHistory = state.pullHistory.slice(-5000);
            }
          }
        });
      },

      updateReward: (armId: string, reward: number) => {
        set(state => {
          const arm = state.arms.get(armId);
          if (arm) {
            // Update arm statistics
            arm.rewards += reward;
            arm.avg_reward = arm.rewards / arm.pulls;
            
            // Update Beta distribution parameters for Thompson Sampling
            if (reward > 0) {
              arm.alpha++;
            } else {
              arm.beta++;
            }
            
            // Update expected value
            arm.expected_value = arm.alpha / (arm.alpha + arm.beta);
            
            // Calculate confidence interval
            const variance = (arm.alpha * arm.beta) / 
              ((arm.alpha + arm.beta) * (arm.alpha + arm.beta) * (arm.alpha + arm.beta + 1));
            const stdDev = Math.sqrt(variance);
            arm.confidence_interval = [
              Math.max(0, arm.expected_value - 2 * stdDev),
              Math.min(1, arm.expected_value + 2 * stdDev)
            ];
            
            // Update reward distribution
            const distribution = state.rewardDistribution.get(armId) || [];
            distribution.push(reward);
            if (distribution.length > 1000) {
              distribution.shift();
            }
            state.rewardDistribution.set(armId, distribution);
            
            // Update variance
            if (distribution.length > 1) {
              const mean = distribution.reduce((a, b) => a + b, 0) / distribution.length;
              const variance = distribution.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / distribution.length;
              arm.variance = variance;
            }
            
            // Update performance metrics
            state.performance.total_rewards += reward;
            
            // Update last pull in history
            const lastPull = state.pullHistory[state.pullHistory.length - 1];
            if (lastPull && lastPull.arm_id === armId) {
              lastPull.reward = reward;
            }
          }
        });
        
        // Check for convergence
        get().analyzeConvergence();
      },

      selectArm: () => {
        const state = get();
        const policy = state.currentPolicy;
        
        switch (policy.type) {
          case 'thompson':
            return state.thompsonSample();
            
          case 'ucb':
            let bestUCB = -Infinity;
            let bestArm = '';
            state.arms.forEach((arm, id) => {
              const ucb = state.calculateUCB(id);
              if (ucb > bestUCB) {
                bestUCB = ucb;
                bestArm = id;
              }
            });
            return bestArm;
            
          case 'epsilon_greedy':
            if (Math.random() < (policy.epsilon || 0.1)) {
              // Explore: random arm
              const arms = Array.from(state.arms.keys());
              return arms[Math.floor(Math.random() * arms.length)];
            } else {
              // Exploit: best arm
              let bestValue = -Infinity;
              let bestArm = '';
              state.arms.forEach((arm, id) => {
                if (arm.expected_value > bestValue) {
                  bestValue = arm.expected_value;
                  bestArm = id;
                }
              });
              return bestArm;
            }
            
          default:
            return state.thompsonSample();
        }
      },

      thompsonSample: () => {
        const state = get();
        let bestSample = -Infinity;
        let bestArm = '';
        
        state.arms.forEach((arm, id) => {
          const sample = sampleBeta(arm.alpha + 1, arm.beta + 1);
          arm.thompson_sample = sample;
          
          if (sample > bestSample) {
            bestSample = sample;
            bestArm = id;
          }
        });
        
        return bestArm;
      },

      calculateUCB: (armId: string) => {
        const state = get();
        const arm = state.arms.get(armId);
        if (!arm || arm.pulls === 0) return Infinity;
        
        const c = state.currentPolicy.c || 2;
        const totalPulls = state.performance.total_pulls;
        
        const exploitation = arm.avg_reward;
        const exploration = Math.sqrt((c * Math.log(totalPulls)) / arm.pulls);
        
        return exploitation + exploration;
      },

      updatePolicy: (policy: BanditPolicy) => {
        const state = get();
        
        set(state => {
          // Save current policy to history
          state.policyHistory.push({
            policy: state.currentPolicy,
            timestamp: Date.now(),
            performance: { ...state.performance }
          });
          
          // Update policy
          state.currentPolicy = policy;
          
          // Adjust exploration rate based on policy
          if (policy.type === 'epsilon_greedy') {
            state.currentExplorationRate = policy.epsilon || 0.1;
          } else if (policy.type === 'thompson') {
            // Thompson sampling naturally balances exploration/exploitation
            state.currentExplorationRate = 0.1; // Approximate
          }
        });
      },

      addArm: (arm: BanditArm) => {
        set(state => {
          state.arms.set(arm.id, arm);
          state.activeArms.push(arm.id);
          state.armSelectionCounts.set(arm.id, 0);
          state.rewardDistribution.set(arm.id, []);
        });
      },

      removeArm: (armId: string) => {
        set(state => {
          state.arms.delete(armId);
          state.activeArms = state.activeArms.filter(id => id !== armId);
          state.armSelectionCounts.delete(armId);
          state.rewardDistribution.delete(armId);
          if (state.bestArm === armId) {
            state.bestArm = null;
          }
        });
      },

      resetArm: (armId: string) => {
        set(state => {
          const arm = state.arms.get(armId);
          if (arm) {
            arm.alpha = 1;
            arm.beta = 1;
            arm.pulls = 0;
            arm.rewards = 0;
            arm.failures = 0;
            arm.avg_reward = 0;
            arm.variance = 0;
            arm.expected_value = 0.5;
            arm.confidence_interval = [0, 1];
            state.rewardDistribution.set(armId, []);
            state.armSelectionCounts.set(armId, 0);
          }
        });
      },

      calculateRegret: () => {
        const state = get();
        
        // Find the true best arm (highest average reward)
        let bestAvgReward = 0;
        let bestArmId = '';
        state.arms.forEach((arm, id) => {
          if (arm.avg_reward > bestAvgReward) {
            bestAvgReward = arm.avg_reward;
            bestArmId = id;
          }
        });
        
        // Calculate regret
        let regret = 0;
        state.arms.forEach((arm, id) => {
          if (id !== bestArmId) {
            const suboptimalPulls = arm.pulls;
            const lostReward = (bestAvgReward - arm.avg_reward) * suboptimalPulls;
            regret += lostReward;
          }
        });
        
        set(state => {
          state.performance.regret = regret;
          state.performance.cumulative_regret += regret;
          
          const bestArm = state.arms.get(bestArmId);
          if (bestArm) {
            state.performance.optimal_arm_pulls = bestArm.pulls;
            state.performance.suboptimal_pulls = state.performance.total_pulls - bestArm.pulls;
          }
        });
        
        return regret;
      },

      updateAdaptiveThreshold: (metric: string, performance: number) => {
        set(state => {
          const threshold = state.adaptiveThresholds.get(metric) || {
            metric,
            current_threshold: 0.5,
            min_threshold: 0.1,
            max_threshold: 0.9,
            adaptation_rate: 0.01,
            last_updated: Date.now()
          };
          
          // Adaptive update based on performance
          if (performance > threshold.current_threshold) {
            // Increase threshold slowly
            threshold.current_threshold = Math.min(
              threshold.max_threshold,
              threshold.current_threshold * (1 + threshold.adaptation_rate)
            );
          } else {
            // Decrease threshold faster
            threshold.current_threshold = Math.max(
              threshold.min_threshold,
              threshold.current_threshold * (1 - threshold.adaptation_rate * 2)
            );
          }
          
          threshold.last_updated = Date.now();
          state.adaptiveThresholds.set(metric, threshold);
        });
      },

      loadMetrics: async () => {
        try {
          const metrics = await apiService.getThompsonSamplingMetrics();
          
          set(state => {
            // Clear existing arms
            state.arms.clear();
            state.activeArms = [];
            
            // Load arms from API
            for (const arm of metrics.arms) {
              const banditArm: BanditArm = {
                id: arm.id,
                route: arm.route,
                alpha: arm.alpha,
                beta: arm.beta,
                pulls: arm.pulls,
                rewards: arm.rewards,
                failures: arm.pulls - arm.rewards,
                ucb_score: arm.ucb_score,
                expected_value: arm.expected_value || arm.alpha / (arm.alpha + arm.beta),
                confidence_interval: [0, 1],
                last_pull: Date.now(),
                avg_reward: arm.rewards / Math.max(1, arm.pulls),
                variance: 0,
                thompson_sample: 0
              };
              
              state.arms.set(arm.id, banditArm);
              state.activeArms.push(arm.id);
            }
            
            // Update performance metrics
            state.performance.exploration_rate = metrics.exploration_rate;
            state.performance.exploitation_rate = metrics.exploitation_rate;
            state.performance.total_pulls = metrics.total_pulls;
            state.performance.convergence_metric = metrics.convergence_metric;
            state.performance.regret = metrics.regret || 0;
          });
        } catch (error) {
          console.error('Failed to load bandit metrics:', error);
        }
      },

      analyzeConvergence: () => {
        const state = get();
        
        // Find dominant arm (highest expected value)
        let dominantArm = '';
        let maxExpectedValue = 0;
        let secondMaxValue = 0;
        
        state.arms.forEach((arm, id) => {
          if (arm.expected_value > maxExpectedValue) {
            secondMaxValue = maxExpectedValue;
            maxExpectedValue = arm.expected_value;
            dominantArm = id;
          } else if (arm.expected_value > secondMaxValue) {
            secondMaxValue = arm.expected_value;
          }
        });
        
        // Calculate convergence metric (gap between best and second best)
        const convergenceMetric = maxExpectedValue - secondMaxValue;
        
        set(state => {
          state.performance.convergence_metric = convergenceMetric;
          
          // Add to history
          state.convergenceHistory.push({
            timestamp: Date.now(),
            metric: convergenceMetric,
            dominant_arm: dominantArm
          });
          
          // Trim history
          if (state.convergenceHistory.length > 1000) {
            state.convergenceHistory = state.convergenceHistory.slice(-500);
          }
          
          // Check if converged
          state.isConverged = convergenceMetric > state.convergenceThreshold;
          state.bestArm = dominantArm;
          
          // Update probabilities
          state.arms.forEach((arm, id) => {
            state.optimalArmProbability.set(id, arm.expected_value);
          });
        });
      },

      getArmStatistics: (armId: string) => {
        const state = get();
        const arm = state.arms.get(armId);
        if (!arm) return null;
        
        const distribution = state.rewardDistribution.get(armId) || [];
        
        return {
          arm,
          distribution,
          percentiles: {
            p25: distribution.sort((a, b) => a - b)[Math.floor(distribution.length * 0.25)] || 0,
            p50: distribution.sort((a, b) => a - b)[Math.floor(distribution.length * 0.50)] || 0,
            p75: distribution.sort((a, b) => a - b)[Math.floor(distribution.length * 0.75)] || 0,
            p95: distribution.sort((a, b) => a - b)[Math.floor(distribution.length * 0.95)] || 0,
            p99: distribution.sort((a, b) => a - b)[Math.floor(distribution.length * 0.99)] || 0
          },
          selectionRate: (state.armSelectionCounts.get(armId) || 0) / Math.max(1, state.performance.total_pulls)
        };
      },

      exportPerformanceReport: () => {
        const state = get();
        
        return {
          timestamp: Date.now(),
          performance: state.performance,
          arms: Array.from(state.arms.entries()).map(([id, arm]) => ({
            id,
            ...arm,
            statistics: state.getArmStatistics(id)
          })),
          policy: state.currentPolicy,
          convergence: {
            isConverged: state.isConverged,
            metric: state.performance.convergence_metric,
            history: state.convergenceHistory.slice(-100)
          },
          adaptiveThresholds: Array.from(state.adaptiveThresholds.entries())
        };
      }
    }))
  )
);

// Auto-refresh metrics every 2 seconds
setInterval(() => {
  useBanditStore.getState().loadMetrics();
}, 2000);

// Calculate regret every 5 seconds
setInterval(() => {
  useBanditStore.getState().calculateRegret();
}, 5000);