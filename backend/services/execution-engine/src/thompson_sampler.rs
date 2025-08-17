use anyhow::Result;
use dashmap::DashMap;
use parking_lot::RwLock;
use rand::prelude::*;
use rand_distr::{Beta, Distribution};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tracing::{debug, info};

use crate::{ExecutionRequest, ExecutionResult, Route, RouteType};

// Thompson Sampling for adaptive route selection
// This implements a multi-armed bandit with Beta-Bernoulli conjugate priors

const INITIAL_ALPHA: f64 = 1.0; // Prior successes
const INITIAL_BETA: f64 = 1.0;  // Prior failures
const DECAY_FACTOR: f64 = 0.99; // Exponential decay for old observations
const MIN_OBSERVATIONS: u32 = 10; // Minimum observations before trusting estimate
const EXPLORATION_BONUS: f64 = 0.1; // Bonus for under-explored routes

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteStats {
    pub route_type: RouteType,
    pub successes: f64,
    pub failures: f64,
    pub total_profit: f64,
    pub avg_latency_ms: f64,
    pub last_update: Instant,
    pub observations: u32,
}

pub struct ThompsonSampler {
    route_stats: Arc<DashMap<RouteType, RouteStats>>,
    total_rounds: AtomicU64,
    exploration_rate: Arc<RwLock<f64>>,
    available_routes: Arc<RwLock<Vec<RouteType>>>,
}

impl ThompsonSampler {
    pub fn new() -> Self {
        let mut available_routes = vec![
            RouteType::LeaderTPU,
            RouteType::NextLeaderTPU,
            RouteType::JitoBlockEngine,
            RouteType::JitoRelayer,
            RouteType::QuicTPU,
            RouteType::UdpTPU,
        ];
        
        let route_stats = Arc::new(DashMap::new());
        
        // Initialize stats for each route
        for route_type in &available_routes {
            route_stats.insert(
                *route_type,
                RouteStats {
                    route_type: *route_type,
                    successes: INITIAL_ALPHA,
                    failures: INITIAL_BETA,
                    total_profit: 0.0,
                    avg_latency_ms: 0.0,
                    last_update: Instant::now(),
                    observations: 0,
                },
            );
        }
        
        Self {
            route_stats,
            total_rounds: AtomicU64::new(0),
            exploration_rate: Arc::new(RwLock::new(0.1)),
            available_routes: Arc::new(RwLock::new(available_routes)),
        }
    }
    
    pub async fn select_route(&self, request: &ExecutionRequest) -> Result<Route> {
        self.total_rounds.fetch_add(1, Ordering::Relaxed);
        
        // Apply decay to old observations
        self.apply_decay();
        
        // Thompson Sampling: sample from posterior for each route
        let mut samples = Vec::new();
        
        for entry in self.route_stats.iter() {
            let stats = entry.value();
            
            // Create Beta distribution with current parameters
            let alpha = stats.successes.max(0.1);
            let beta = stats.failures.max(0.1);
            
            // Sample from posterior
            let beta_dist = Beta::new(alpha, beta)?;
            let mut rng = thread_rng();
            let sample = beta_dist.sample(&mut rng);
            
            // Adjust sample based on context
            let adjusted_sample = self.adjust_sample_for_context(
                sample,
                &stats,
                request.expected_profit,
            );
            
            samples.push((stats.route_type, adjusted_sample, stats.clone()));
        }
        
        // Select route with highest adjusted sample
        samples.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let selected_route_type = samples[0].0;
        let selected_stats = &samples[0].2;
        
        debug!(
            "Thompson Sampler selected {:?} with score {:.4} (α={:.2}, β={:.2}, obs={})",
            selected_route_type,
            samples[0].1,
            selected_stats.successes,
            selected_stats.failures,
            selected_stats.observations
        );
        
        // Convert to actual Route
        self.create_route(selected_route_type, request).await
    }
    
    pub async fn update_with_outcome(&self, outcome: ExecutionResult) {
        let route_type = outcome.route_used.route_type;
        
        if let Some(mut stats) = self.route_stats.get_mut(&route_type) {
            // Update success/failure counts
            if outcome.status == crate::ExecutionStatus::Confirmed {
                stats.successes += 1.0;
            } else {
                stats.failures += 1.0;
            }
            
            // Update profit tracking
            if let Some(profit_str) = outcome.metadata.get("profit") {
                if let Ok(profit) = profit_str.parse::<f64>() {
                    stats.total_profit += profit;
                }
            }
            
            // Update latency with exponential moving average
            let alpha = 0.1; // EMA factor
            stats.avg_latency_ms = (1.0 - alpha) * stats.avg_latency_ms + 
                                   alpha * outcome.latency_ms as f64;
            
            stats.observations += 1;
            stats.last_update = Instant::now();
            
            debug!(
                "Updated {:?}: successes={:.1}, failures={:.1}, avg_latency={:.1}ms",
                route_type,
                stats.successes,
                stats.failures,
                stats.avg_latency_ms
            );
        }
    }
    
    fn adjust_sample_for_context(
        &self,
        base_sample: f64,
        stats: &RouteStats,
        expected_profit: u64,
    ) -> f64 {
        let mut adjusted = base_sample;
        
        // 1. Exploration bonus for under-explored routes
        if stats.observations < MIN_OBSERVATIONS {
            adjusted += EXPLORATION_BONUS * (1.0 - stats.observations as f64 / MIN_OBSERVATIONS as f64);
        }
        
        // 2. Latency penalty for high-value opportunities
        if expected_profit > 1_000_000_000 { // > 1 SOL
            let latency_penalty = (stats.avg_latency_ms / 10.0).min(0.2);
            adjusted -= latency_penalty;
        }
        
        // 3. Profit-weighted adjustment
        if stats.observations > 0 {
            let avg_profit = stats.total_profit / stats.observations as f64;
            let profit_bonus = (avg_profit / 1_000_000_000.0).min(0.1); // Normalize to SOL
            adjusted += profit_bonus;
        }
        
        // 4. Recency bonus - prefer recently successful routes
        let time_since_update = stats.last_update.elapsed().as_secs() as f64;
        if time_since_update < 60.0 {
            adjusted += 0.05 * (1.0 - time_since_update / 60.0);
        }
        
        adjusted.max(0.0).min(1.0)
    }
    
    fn apply_decay(&self) {
        // Apply exponential decay to old observations
        for mut entry in self.route_stats.iter_mut() {
            let time_since_update = entry.last_update.elapsed().as_secs() as f64;
            
            // Decay more aggressively for older observations
            if time_since_update > 300.0 { // 5 minutes
                let decay = DECAY_FACTOR.powf(time_since_update / 300.0);
                entry.successes = (entry.successes - INITIAL_ALPHA) * decay + INITIAL_ALPHA;
                entry.failures = (entry.failures - INITIAL_BETA) * decay + INITIAL_BETA;
            }
        }
    }
    
    async fn create_route(&self, route_type: RouteType, request: &ExecutionRequest) -> Result<Route> {
        // Create actual route based on type and request context
        let endpoint = match route_type {
            RouteType::LeaderTPU => "leader.tpu.solana.com:8003".to_string(),
            RouteType::NextLeaderTPU => "next-leader.tpu.solana.com:8003".to_string(),
            RouteType::JitoBlockEngine => "mainnet.block-engine.jito.wtf".to_string(),
            RouteType::JitoRelayer => "amsterdam.mainnet.relayer.jito.wtf".to_string(),
            RouteType::QuicTPU => "quic.tpu.solana.com:8003".to_string(),
            RouteType::UdpTPU => "udp.tpu.solana.com:8003".to_string(),
        };
        
        // Dynamic priority fee based on route and expected profit
        let priority_fee = self.calculate_priority_fee(route_type, request.expected_profit);
        
        // Tip amount for Jito routes
        let tip_amount = if matches!(route_type, RouteType::JitoBlockEngine | RouteType::JitoRelayer) {
            Some(self.calculate_jito_tip(request.expected_profit))
        } else {
            None
        };
        
        Ok(Route {
            route_type,
            endpoint,
            priority_fee,
            tip_amount,
        })
    }
    
    fn calculate_priority_fee(&self, route_type: RouteType, expected_profit: u64) -> u64 {
        // Base fee depends on route
        let base_fee = match route_type {
            RouteType::LeaderTPU | RouteType::NextLeaderTPU => 50_000,
            RouteType::JitoBlockEngine | RouteType::JitoRelayer => 100_000,
            RouteType::QuicTPU => 75_000,
            RouteType::UdpTPU => 25_000,
        };
        
        // Scale with expected profit
        let profit_multiplier = if expected_profit > 10_000_000_000 { // > 10 SOL
            3.0
        } else if expected_profit > 1_000_000_000 { // > 1 SOL
            2.0
        } else {
            1.0
        };
        
        (base_fee as f64 * profit_multiplier) as u64
    }
    
    fn calculate_jito_tip(&self, expected_profit: u64) -> u64 {
        // Jito tip is typically 50-80% of expected profit for high-value
        // Adjust based on competition and land rate
        let land_rate = self.get_current_land_rate();
        
        let tip_percentage = if land_rate < 0.5 {
            0.8 // Low land rate, need higher tip
        } else if land_rate < 0.7 {
            0.6 // Medium land rate
        } else {
            0.5 // High land rate, can use lower tip
        };
        
        ((expected_profit as f64) * tip_percentage) as u64
    }
    
    fn get_current_land_rate(&self) -> f64 {
        // Get current bundle land rate from stats
        if let Some(stats) = self.route_stats.get(&RouteType::JitoBlockEngine) {
            if stats.observations > 0 {
                return stats.successes / (stats.successes + stats.failures);
            }
        }
        0.65 // Default assumption
    }
    
    pub fn get_statistics(&self) -> HashMap<RouteType, RouteStatistics> {
        let mut stats = HashMap::new();
        
        for entry in self.route_stats.iter() {
            let route_stats = entry.value();
            let win_rate = if route_stats.observations > 0 {
                route_stats.successes / (route_stats.successes + route_stats.failures)
            } else {
                0.5
            };
            
            stats.insert(
                route_stats.route_type,
                RouteStatistics {
                    win_rate,
                    observations: route_stats.observations,
                    avg_latency_ms: route_stats.avg_latency_ms,
                    total_profit: route_stats.total_profit,
                    confidence_interval: self.calculate_confidence_interval(&route_stats),
                },
            );
        }
        
        stats
    }
    
    fn calculate_confidence_interval(&self, stats: &RouteStats) -> (f64, f64) {
        // Wilson score interval for binomial proportion
        let n = (stats.successes + stats.failures) as f64;
        if n == 0.0 {
            return (0.0, 1.0);
        }
        
        let p = stats.successes / n;
        let z = 1.96; // 95% confidence
        
        let denominator = 1.0 + z * z / n;
        let center = (p + z * z / (2.0 * n)) / denominator;
        let spread = (z / denominator) * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt();
        
        ((center - spread).max(0.0), (center + spread).min(1.0))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteStatistics {
    pub win_rate: f64,
    pub observations: u32,
    pub avg_latency_ms: f64,
    pub total_profit: f64,
    pub confidence_interval: (f64, f64),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_thompson_sampling() {
        let sampler = ThompsonSampler::new();
        
        // Create test request
        let request = ExecutionRequest {
            id: "test".to_string(),
            transaction: Default::default(),
            strategy: crate::ExecutionStrategy::AdaptiveBandit,
            priority: crate::ExecutionPriority::High,
            max_retries: 3,
            timeout_ms: 1000,
            expected_profit: 1_000_000_000,
            metadata: HashMap::new(),
        };
        
        // Select route multiple times
        for _ in 0..100 {
            let route = sampler.select_route(&request).await.unwrap();
            assert!(!route.endpoint.is_empty());
            assert!(route.priority_fee > 0);
        }
        
        // Check that all routes have been explored
        let stats = sampler.get_statistics();
        for (_, route_stats) in stats {
            assert!(route_stats.observations > 0);
        }
    }
}