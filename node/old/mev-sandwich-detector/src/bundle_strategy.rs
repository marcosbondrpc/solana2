use std::sync::Arc;
use parking_lot::RwLock;
use anyhow::Result;
use tracing::{info, debug};

pub struct TipLadderStrategy {
    buckets: Vec<TipBucket>,
    network_load: Arc<RwLock<NetworkLoad>>,
    historical_success: Arc<RwLock<HistoricalData>>,
}

#[derive(Clone)]
struct TipBucket {
    profit_range: (u64, u64),
    base_tip_bps: u16,
    congestion_multiplier: f32,
    success_threshold: f32,
}

struct NetworkLoad {
    current_load: f32,
    avg_gas_price: u64,
    mempool_depth: usize,
    competitor_activity: f32,
}

struct HistoricalData {
    success_by_tip: Vec<(u64, f32)>,
    profit_by_tip: Vec<(u64, i64)>,
}

impl TipLadderStrategy {
    pub fn new() -> Self {
        info!("Initializing Tip Ladder Strategy");
        
        let buckets = vec![
            TipBucket {
                profit_range: (0, 50_000_000),           // < 0.05 SOL
                base_tip_bps: 300,                       // 3%
                congestion_multiplier: 1.2,
                success_threshold: 0.5,
            },
            TipBucket {
                profit_range: (50_000_000, 100_000_000), // 0.05-0.1 SOL
                base_tip_bps: 500,                       // 5%
                congestion_multiplier: 1.5,
                success_threshold: 0.6,
            },
            TipBucket {
                profit_range: (100_000_000, 250_000_000), // 0.1-0.25 SOL
                base_tip_bps: 750,                        // 7.5%
                congestion_multiplier: 1.8,
                success_threshold: 0.65,
            },
            TipBucket {
                profit_range: (250_000_000, 500_000_000), // 0.25-0.5 SOL
                base_tip_bps: 1000,                       // 10%
                congestion_multiplier: 2.0,
                success_threshold: 0.7,
            },
            TipBucket {
                profit_range: (500_000_000, 1_000_000_000), // 0.5-1 SOL
                base_tip_bps: 1250,                         // 12.5%
                congestion_multiplier: 2.2,
                success_threshold: 0.75,
            },
            TipBucket {
                profit_range: (1_000_000_000, u64::MAX),    // > 1 SOL
                base_tip_bps: 1500,                         // 15%
                congestion_multiplier: 2.5,
                success_threshold: 0.8,
            },
        ];
        
        Self {
            buckets,
            network_load: Arc::new(RwLock::new(NetworkLoad::default())),
            historical_success: Arc::new(RwLock::new(HistoricalData::default())),
        }
    }
    
    pub fn calculate_tip(&self, expected_profit: u64, confidence: f32) -> u64 {
        // Find appropriate bucket
        let bucket = self.buckets.iter()
            .find(|b| expected_profit >= b.profit_range.0 && expected_profit < b.profit_range.1)
            .unwrap_or(&self.buckets.last().unwrap());
        
        // Get network conditions
        let network = self.network_load.read();
        
        // Calculate base tip
        let base_tip = (expected_profit * bucket.base_tip_bps as u64) / 10000;
        
        // Apply dynamic adjustments
        let confidence_factor = Self::confidence_to_multiplier(confidence);
        let network_factor = Self::network_load_to_multiplier(network.current_load, bucket.congestion_multiplier);
        let competition_factor = Self::competition_to_multiplier(network.competitor_activity);
        
        // Calculate adaptive tip based on historical success
        let historical_factor = self.calculate_historical_factor(expected_profit);
        
        // Combine all factors
        let final_tip = (base_tip as f64 
            * confidence_factor 
            * network_factor 
            * competition_factor
            * historical_factor) as u64;
        
        // Apply bounds
        let min_tip = expected_profit / 100;  // Min 1%
        let max_tip = expected_profit / 2;    // Max 50%
        
        let bounded_tip = final_tip.max(min_tip).min(max_tip);
        
        debug!(
            "Tip calculation: profit={}, base={}, final={}, factors: conf={:.2}, net={:.2}, comp={:.2}, hist={:.2}",
            expected_profit, base_tip, bounded_tip,
            confidence_factor, network_factor, competition_factor, historical_factor
        );
        
        bounded_tip
    }
    
    fn confidence_to_multiplier(confidence: f32) -> f64 {
        // Higher confidence = higher tip (willing to pay more for certainty)
        0.5 + (confidence as f64 * 0.75)  // 0.5x to 1.25x
    }
    
    fn network_load_to_multiplier(load: f32, base_multiplier: f32) -> f64 {
        // Higher load = higher tip needed
        if load < 0.3 {
            0.8  // Low load, can tip less
        } else if load < 0.6 {
            1.0  // Normal load
        } else if load < 0.8 {
            base_multiplier as f64  // High load
        } else {
            (base_multiplier * 1.5) as f64  // Very high load
        }
    }
    
    fn competition_to_multiplier(competition: f32) -> f64 {
        // More competition = higher tip
        1.0 + (competition as f64 * 0.5)  // 1.0x to 1.5x
    }
    
    fn calculate_historical_factor(&self, profit: u64) -> f64 {
        let history = self.historical_success.read();
        
        // Find similar profit trades
        let similar: Vec<_> = history.success_by_tip.iter()
            .filter(|(tip, _)| {
                let tip_ratio = *tip as f64 / profit as f64;
                tip_ratio > 0.01 && tip_ratio < 0.5
            })
            .collect();
        
        if similar.is_empty() {
            return 1.0;  // No history, use default
        }
        
        // Calculate optimal tip ratio from history
        let mut best_ratio = 0.1;
        let mut best_success = 0.0;
        
        for (tip, success) in similar {
            let ratio = *tip as f64 / profit as f64;
            if *success > best_success {
                best_success = *success;
                best_ratio = ratio;
            }
        }
        
        // Adjust towards historical best
        let current_ratio = 0.1;  // Default 10%
        let adjustment = (best_ratio - current_ratio) * 0.5;  // Move 50% towards best
        
        1.0 + adjustment
    }
    
    pub fn update_network_load(&self, load: f32, gas_price: u64, mempool_depth: usize) {
        let mut network = self.network_load.write();
        network.current_load = load;
        network.avg_gas_price = gas_price;
        network.mempool_depth = mempool_depth;
        
        // Estimate competitor activity
        network.competitor_activity = (mempool_depth as f32 / 1000.0).min(1.0);
    }
    
    pub fn record_outcome(&self, tip: u64, profit: u64, success: bool) {
        let mut history = self.historical_success.write();
        
        // Update success rate for this tip level
        let success_rate = if success { 1.0 } else { 0.0 };
        history.success_by_tip.push((tip, success_rate));
        
        // Keep only recent history
        if history.success_by_tip.len() > 10000 {
            history.success_by_tip.remove(0);
        }
        
        // Record profit
        let net_profit = if success {
            profit as i64 - tip as i64
        } else {
            -(tip as i64)
        };
        history.profit_by_tip.push((tip, net_profit));
        
        if history.profit_by_tip.len() > 10000 {
            history.profit_by_tip.remove(0);
        }
    }
    
    pub fn get_optimal_tip_stats(&self) -> TipStats {
        let history = self.historical_success.read();
        
        // Calculate statistics
        let total_trades = history.success_by_tip.len();
        let successful = history.success_by_tip.iter().filter(|(_, s)| *s > 0.0).count();
        let total_profit: i64 = history.profit_by_tip.iter().map(|(_, p)| p).sum();
        
        let avg_tip = if !history.success_by_tip.is_empty() {
            history.success_by_tip.iter().map(|(t, _)| *t).sum::<u64>() / total_trades as u64
        } else {
            0
        };
        
        TipStats {
            total_trades,
            successful_trades: successful,
            success_rate: if total_trades > 0 {
                successful as f64 / total_trades as f64
            } else {
                0.0
            },
            total_profit,
            average_tip: avg_tip,
        }
    }
}

pub struct TipStats {
    pub total_trades: usize,
    pub successful_trades: usize,
    pub success_rate: f64,
    pub total_profit: i64,
    pub average_tip: u64,
}

impl Default for NetworkLoad {
    fn default() -> Self {
        Self {
            current_load: 0.5,
            avg_gas_price: 50_000,
            mempool_depth: 100,
            competitor_activity: 0.3,
        }
    }
}

impl Default for HistoricalData {
    fn default() -> Self {
        Self {
            success_by_tip: Vec::new(),
            profit_by_tip: Vec::new(),
        }
    }
}