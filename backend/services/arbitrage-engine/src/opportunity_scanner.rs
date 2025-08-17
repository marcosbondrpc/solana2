use ahash::{AHashMap, AHashSet};
use anyhow::Result;
use arc_swap::ArcSwap;
use dashmap::DashMap;
use dex_registry::{Pool, PricingGraph, SwapPath, DexType};
use ethnum::U256;
use parking_lot::RwLock;
use petgraph::graph::{DiGraph, NodeIndex};
use priority_queue::PriorityQueue;
use rayon::prelude::*;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use solana_sdk::pubkey::Pubkey;
use std::{
    cmp::Reverse,
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tracing::{debug, error, info, warn};

// Constants for performance
const SCAN_INTERVAL_MS: u64 = 10;
const MIN_PROFIT_THRESHOLD: u64 = 100_000; // 0.1 SOL minimum profit
const MAX_CYCLES_PER_SCAN: usize = 1000;
const PARALLEL_WORKERS: usize = 8;
const CACHE_TTL_MS: u64 = 100;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageOpportunity {
    pub id: String,
    pub opportunity_type: OpportunityType,
    pub route: Vec<SwapLeg>,
    pub input_amount: u64,
    pub expected_output: u64,
    pub profit: i64,
    pub profit_percentage: f64,
    pub price_impact: f64,
    pub gas_estimate: u64,
    pub net_profit: i64,
    pub confidence: f64,
    pub discovery_time: Instant,
    pub valid_until: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpportunityType {
    TwoLegArbitrage,      // A -> B -> A
    TriangularArbitrage,  // A -> B -> C -> A
    MultiHopArbitrage,    // k-cycle with k > 3
    CrossDexArbitrage,    // Same pair across different DEXes
    StableSwapArbitrage,  // Stable coin arbitrage
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapLeg {
    pub pool_address: Pubkey,
    pub dex_type: DexType,
    pub token_in: Pubkey,
    pub token_out: Pubkey,
    pub amount_in: u64,
    pub expected_out: u64,
    pub slippage_tolerance: u16,
}

pub struct OpportunityScanner {
    pricing_graph: Arc<PricingGraph>,
    opportunity_cache: Arc<DashMap<String, ArbitrageOpportunity>>,
    token_pairs: Arc<RwLock<HashMap<(Pubkey, Pubkey), Vec<Pubkey>>>>,
    stable_coins: Arc<HashSet<Pubkey>>,
    metrics: Arc<ScannerMetrics>,
    shutdown: Arc<AtomicBool>,
    min_profit_threshold: u64,
    gas_price_estimator: Arc<GasPriceEstimator>,
}

struct ScannerMetrics {
    opportunities_found: AtomicU64,
    profitable_opportunities: AtomicU64,
    cycles_scanned: AtomicU64,
    avg_scan_time_ms: AtomicU64,
    total_profit_simulated: AtomicU64,
}

struct GasPriceEstimator {
    base_fee: AtomicU64,
    priority_fee: AtomicU64,
    compute_units_per_hop: u64,
}

impl OpportunityScanner {
    pub fn new(pricing_graph: Arc<PricingGraph>) -> Self {
        // Initialize with common stable coins
        let mut stable_coins = HashSet::new();
        stable_coins.insert(Pubkey::new_from_array([1; 32])); // USDC
        stable_coins.insert(Pubkey::new_from_array([2; 32])); // USDT
        stable_coins.insert(Pubkey::new_from_array([3; 32])); // BUSD
        stable_coins.insert(Pubkey::new_from_array([4; 32])); // DAI
        
        Self {
            pricing_graph,
            opportunity_cache: Arc::new(DashMap::new()),
            token_pairs: Arc::new(RwLock::new(HashMap::new())),
            stable_coins: Arc::new(stable_coins),
            metrics: Arc::new(ScannerMetrics {
                opportunities_found: AtomicU64::new(0),
                profitable_opportunities: AtomicU64::new(0),
                cycles_scanned: AtomicU64::new(0),
                avg_scan_time_ms: AtomicU64::new(0),
                total_profit_simulated: AtomicU64::new(0),
            }),
            shutdown: Arc::new(AtomicBool::new(false)),
            min_profit_threshold: MIN_PROFIT_THRESHOLD,
            gas_price_estimator: Arc::new(GasPriceEstimator {
                base_fee: AtomicU64::new(5000),
                priority_fee: AtomicU64::new(50000),
                compute_units_per_hop: 200_000,
            }),
        }
    }
    
    pub async fn scan_opportunities(&self) -> Result<Vec<ArbitrageOpportunity>> {
        let start = Instant::now();
        let mut all_opportunities = Vec::new();
        
        // Parallel scanning of different opportunity types
        let (two_leg, triangular, cross_dex, stable) = rayon::join(
            || self.scan_two_leg_arbitrage(),
            || self.scan_triangular_arbitrage(),
            || self.scan_cross_dex_arbitrage(),
            || self.scan_stable_arbitrage(),
        );
        
        // Collect all opportunities
        all_opportunities.extend(two_leg?);
        all_opportunities.extend(triangular?);
        all_opportunities.extend(cross_dex?);
        all_opportunities.extend(stable?);
        
        // Filter and rank opportunities
        let mut profitable: Vec<_> = all_opportunities
            .into_iter()
            .filter(|opp| opp.net_profit > self.min_profit_threshold as i64)
            .collect();
        
        // Sort by net profit (descending)
        profitable.sort_by(|a, b| b.net_profit.cmp(&a.net_profit));
        
        // Update metrics
        let scan_time = start.elapsed().as_millis() as u64;
        self.metrics.avg_scan_time_ms.store(scan_time, Ordering::Relaxed);
        self.metrics.profitable_opportunities.store(profitable.len() as u64, Ordering::Relaxed);
        
        // Cache opportunities
        for opp in &profitable {
            self.opportunity_cache.insert(opp.id.clone(), opp.clone());
        }
        
        // Clean expired opportunities
        self.clean_expired_opportunities();
        
        Ok(profitable)
    }
    
    fn scan_two_leg_arbitrage(&self) -> Result<Vec<ArbitrageOpportunity>> {
        let mut opportunities = Vec::new();
        
        // Get all token pairs with direct pools
        let pairs = self.get_all_token_pairs();
        
        // Check each pair for arbitrage
        pairs.par_iter().for_each(|(token_a, token_b)| {
            // Find all pools connecting these tokens
            let pools = self.find_pools_for_pair(*token_a, *token_b);
            
            if pools.len() < 2 {
                return; // Need at least 2 pools for arbitrage
            }
            
            // Test different input amounts
            let test_amounts = vec![
                1_000_000_000,    // 1 SOL
                5_000_000_000,    // 5 SOL
                10_000_000_000,   // 10 SOL
                50_000_000_000,   // 50 SOL
            ];
            
            for amount in test_amounts {
                // Try arbitrage: Pool1(A->B) -> Pool2(B->A)
                for pool1 in &pools {
                    for pool2 in &pools {
                        if pool1.address() == pool2.address() {
                            continue;
                        }
                        
                        if let Some(opp) = self.simulate_two_leg(
                            pool1.as_ref(),
                            pool2.as_ref(),
                            *token_a,
                            *token_b,
                            amount,
                        ) {
                            if opp.net_profit > 0 {
                                // Safe to collect as we're in a parallel context
                                // Would need proper synchronization in real impl
                            }
                        }
                    }
                }
            }
        });
        
        Ok(opportunities)
    }
    
    fn scan_triangular_arbitrage(&self) -> Result<Vec<ArbitrageOpportunity>> {
        let mut opportunities = Vec::new();
        
        // Find all triangular cycles in the graph
        let cycles = self.find_triangular_cycles()?;
        
        // Test each cycle for profitability
        for cycle in cycles.iter().take(MAX_CYCLES_PER_SCAN) {
            if cycle.len() != 3 {
                continue;
            }
            
            // Test different input amounts
            let test_amounts = vec![1_000_000_000, 5_000_000_000, 10_000_000_000];
            
            for amount in test_amounts {
                if let Some(opp) = self.simulate_triangular(
                    &cycle[0],
                    &cycle[1],
                    &cycle[2],
                    amount,
                ) {
                    if opp.net_profit > self.min_profit_threshold as i64 {
                        opportunities.push(opp);
                    }
                }
            }
        }
        
        self.metrics.cycles_scanned.fetch_add(cycles.len() as u64, Ordering::Relaxed);
        
        Ok(opportunities)
    }
    
    fn scan_cross_dex_arbitrage(&self) -> Result<Vec<ArbitrageOpportunity>> {
        let mut opportunities = Vec::new();
        
        // Group pools by token pair
        let pools_by_pair = self.group_pools_by_pair();
        
        // Check each pair across different DEXes
        for ((token_a, token_b), pools) in pools_by_pair.iter() {
            if pools.len() < 2 {
                continue;
            }
            
            // Group by DEX type
            let mut by_dex: HashMap<DexType, Vec<_>> = HashMap::new();
            for pool in pools {
                by_dex.entry(pool.dex_type()).or_default().push(pool.clone());
            }
            
            if by_dex.len() < 2 {
                continue; // Need at least 2 different DEXes
            }
            
            // Compare prices across DEXes
            let test_amounts = vec![1_000_000_000, 10_000_000_000];
            
            for amount in test_amounts {
                let mut prices: Vec<(DexType, Arc<dyn Pool>, f64)> = Vec::new();
                
                for (dex_type, dex_pools) in &by_dex {
                    for pool in dex_pools {
                        if let Ok(output) = pool.get_amount_out(amount, *token_a) {
                            let price = output as f64 / amount as f64;
                            prices.push((*dex_type, pool.clone(), price));
                        }
                    }
                }
                
                // Find best arbitrage between DEXes
                prices.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
                
                if prices.len() >= 2 {
                    let price_diff = (prices.last().unwrap().2 - prices[0].2) / prices[0].2;
                    
                    if price_diff > 0.001 { // 0.1% minimum difference
                        // Create cross-DEX opportunity
                        let buy_pool = &prices[0].1;
                        let sell_pool = &prices.last().unwrap().1;
                        
                        if let Some(opp) = self.simulate_cross_dex(
                            buy_pool.as_ref(),
                            sell_pool.as_ref(),
                            *token_a,
                            *token_b,
                            amount,
                        ) {
                            opportunities.push(opp);
                        }
                    }
                }
            }
        }
        
        Ok(opportunities)
    }
    
    fn scan_stable_arbitrage(&self) -> Result<Vec<ArbitrageOpportunity>> {
        let mut opportunities = Vec::new();
        
        // Find arbitrage between stable coins
        let stable_pairs: Vec<(Pubkey, Pubkey)> = self.stable_coins
            .iter()
            .flat_map(|a| {
                self.stable_coins
                    .iter()
                    .filter(|b| a != *b)
                    .map(move |b| (*a, *b))
            })
            .collect();
        
        for (stable_a, stable_b) in stable_pairs {
            // Stable coins should trade near 1:1
            let pools = self.find_pools_for_pair(stable_a, stable_b);
            
            for pool in pools {
                let test_amount = 1_000_000_000; // 1000 USDC
                
                if let Ok(output) = pool.get_amount_out(test_amount, stable_a) {
                    let rate = output as f64 / test_amount as f64;
                    
                    // If rate deviates from 1.0, there's arbitrage
                    if (rate - 1.0).abs() > 0.001 {
                        // Find reverse path
                        if let Ok(path) = self.pricing_graph.find_best_path(
                            stable_b,
                            stable_a,
                            output,
                            2,
                        ) {
                            let final_amount = path.expected_out;
                            let profit = final_amount as i64 - test_amount as i64;
                            
                            if profit > self.min_profit_threshold as i64 {
                                let opp = ArbitrageOpportunity {
                                    id: format!("stable_{}_{}", stable_a, stable_b),
                                    opportunity_type: OpportunityType::StableSwapArbitrage,
                                    route: vec![], // Would be filled with actual route
                                    input_amount: test_amount,
                                    expected_output: final_amount,
                                    profit,
                                    profit_percentage: (profit as f64 / test_amount as f64) * 100.0,
                                    price_impact: pool.calculate_price_impact(test_amount, stable_a),
                                    gas_estimate: self.estimate_gas(2),
                                    net_profit: profit - self.estimate_gas(2) as i64,
                                    confidence: 0.95, // High confidence for stable swaps
                                    discovery_time: Instant::now(),
                                    valid_until: Instant::now() + Duration::from_millis(100),
                                };
                                
                                opportunities.push(opp);
                            }
                        }
                    }
                }
            }
        }
        
        Ok(opportunities)
    }
    
    fn simulate_two_leg(
        &self,
        pool1: &dyn Pool,
        pool2: &dyn Pool,
        token_a: Pubkey,
        token_b: Pubkey,
        amount_in: u64,
    ) -> Option<ArbitrageOpportunity> {
        // First leg: A -> B through pool1
        let amount_b = pool1.get_amount_out(amount_in, token_a).ok()?;
        
        // Second leg: B -> A through pool2
        let amount_a_out = pool2.get_amount_out(amount_b, token_b).ok()?;
        
        // Calculate profit
        let profit = amount_a_out as i64 - amount_in as i64;
        
        if profit <= 0 {
            return None;
        }
        
        // Calculate total price impact
        let impact1 = pool1.calculate_price_impact(amount_in, token_a);
        let impact2 = pool2.calculate_price_impact(amount_b, token_b);
        let total_impact = impact1 + impact2;
        
        // Estimate gas
        let gas = self.estimate_gas(2);
        let net_profit = profit - gas as i64;
        
        if net_profit <= self.min_profit_threshold as i64 {
            return None;
        }
        
        Some(ArbitrageOpportunity {
            id: format!("2leg_{}_{}_{}", pool1.address(), pool2.address(), amount_in),
            opportunity_type: OpportunityType::TwoLegArbitrage,
            route: vec![
                SwapLeg {
                    pool_address: pool1.address(),
                    dex_type: pool1.dex_type(),
                    token_in: token_a,
                    token_out: token_b,
                    amount_in,
                    expected_out: amount_b,
                    slippage_tolerance: 50, // 0.5%
                },
                SwapLeg {
                    pool_address: pool2.address(),
                    dex_type: pool2.dex_type(),
                    token_in: token_b,
                    token_out: token_a,
                    amount_in: amount_b,
                    expected_out: amount_a_out,
                    slippage_tolerance: 50,
                },
            ],
            input_amount: amount_in,
            expected_output: amount_a_out,
            profit,
            profit_percentage: (profit as f64 / amount_in as f64) * 100.0,
            price_impact: total_impact,
            gas_estimate: gas,
            net_profit,
            confidence: self.calculate_confidence(total_impact, profit as u64),
            discovery_time: Instant::now(),
            valid_until: Instant::now() + Duration::from_millis(50),
        })
    }
    
    fn simulate_triangular(
        &self,
        token_a: &Pubkey,
        token_b: &Pubkey,
        token_c: &Pubkey,
        amount_in: u64,
    ) -> Option<ArbitrageOpportunity> {
        // Find best path for each leg
        let path_ab = self.pricing_graph.find_best_path(*token_a, *token_b, amount_in, 1).ok()?;
        let path_bc = self.pricing_graph.find_best_path(*token_b, *token_c, path_ab.expected_out, 1).ok()?;
        let path_ca = self.pricing_graph.find_best_path(*token_c, *token_a, path_bc.expected_out, 1).ok()?;
        
        let final_amount = path_ca.expected_out;
        let profit = final_amount as i64 - amount_in as i64;
        
        if profit <= 0 {
            return None;
        }
        
        let total_impact = path_ab.price_impact + path_bc.price_impact + path_ca.price_impact;
        let gas = self.estimate_gas(3);
        let net_profit = profit - gas as i64;
        
        if net_profit <= self.min_profit_threshold as i64 {
            return None;
        }
        
        Some(ArbitrageOpportunity {
            id: format!("tri_{}_{}_{}_{}", token_a, token_b, token_c, amount_in),
            opportunity_type: OpportunityType::TriangularArbitrage,
            route: vec![], // Would combine routes from all paths
            input_amount: amount_in,
            expected_output: final_amount,
            profit,
            profit_percentage: (profit as f64 / amount_in as f64) * 100.0,
            price_impact: total_impact,
            gas_estimate: gas,
            net_profit,
            confidence: self.calculate_confidence(total_impact, profit as u64),
            discovery_time: Instant::now(),
            valid_until: Instant::now() + Duration::from_millis(40),
        })
    }
    
    fn simulate_cross_dex(
        &self,
        buy_pool: &dyn Pool,
        sell_pool: &dyn Pool,
        token_a: Pubkey,
        token_b: Pubkey,
        amount_in: u64,
    ) -> Option<ArbitrageOpportunity> {
        // Buy token_b with token_a on buy_pool
        let amount_b = buy_pool.get_amount_out(amount_in, token_a).ok()?;
        
        // Sell token_b for token_a on sell_pool
        let amount_a_out = sell_pool.get_amount_out(amount_b, token_b).ok()?;
        
        let profit = amount_a_out as i64 - amount_in as i64;
        
        if profit <= 0 {
            return None;
        }
        
        let impact1 = buy_pool.calculate_price_impact(amount_in, token_a);
        let impact2 = sell_pool.calculate_price_impact(amount_b, token_b);
        let total_impact = impact1 + impact2;
        
        let gas = self.estimate_gas(2);
        let net_profit = profit - gas as i64;
        
        if net_profit <= self.min_profit_threshold as i64 {
            return None;
        }
        
        Some(ArbitrageOpportunity {
            id: format!("xdex_{}_{}_{}", buy_pool.address(), sell_pool.address(), amount_in),
            opportunity_type: OpportunityType::CrossDexArbitrage,
            route: vec![
                SwapLeg {
                    pool_address: buy_pool.address(),
                    dex_type: buy_pool.dex_type(),
                    token_in: token_a,
                    token_out: token_b,
                    amount_in,
                    expected_out: amount_b,
                    slippage_tolerance: 50,
                },
                SwapLeg {
                    pool_address: sell_pool.address(),
                    dex_type: sell_pool.dex_type(),
                    token_in: token_b,
                    token_out: token_a,
                    amount_in: amount_b,
                    expected_out: amount_a_out,
                    slippage_tolerance: 50,
                },
            ],
            input_amount: amount_in,
            expected_output: amount_a_out,
            profit,
            profit_percentage: (profit as f64 / amount_in as f64) * 100.0,
            price_impact: total_impact,
            gas_estimate: gas,
            net_profit,
            confidence: self.calculate_confidence(total_impact, profit as u64),
            discovery_time: Instant::now(),
            valid_until: Instant::now() + Duration::from_millis(30),
        })
    }
    
    fn estimate_gas(&self, num_hops: usize) -> u64 {
        let base = self.gas_price_estimator.base_fee.load(Ordering::Relaxed);
        let priority = self.gas_price_estimator.priority_fee.load(Ordering::Relaxed);
        let compute = self.gas_price_estimator.compute_units_per_hop * num_hops as u64;
        
        base + priority + (compute * 100) // Simplified gas calculation
    }
    
    fn calculate_confidence(&self, price_impact: f64, profit: u64) -> f64 {
        // Higher profit and lower impact = higher confidence
        let profit_score = (profit as f64 / 1_000_000_000.0).min(1.0); // Normalize to 1 SOL
        let impact_score = 1.0 - (price_impact * 10.0).min(1.0); // Penalize high impact
        
        (profit_score * 0.6 + impact_score * 0.4).max(0.0).min(1.0)
    }
    
    fn find_triangular_cycles(&self) -> Result<Vec<Vec<Pubkey>>> {
        // This would use graph algorithms to find cycles
        // Simplified implementation
        Ok(vec![])
    }
    
    fn get_all_token_pairs(&self) -> Vec<(Pubkey, Pubkey)> {
        // Get all unique token pairs from registered pools
        vec![]
    }
    
    fn find_pools_for_pair(&self, token_a: Pubkey, token_b: Pubkey) -> Vec<Arc<dyn Pool>> {
        // Find all pools connecting these tokens
        vec![]
    }
    
    fn group_pools_by_pair(&self) -> HashMap<(Pubkey, Pubkey), Vec<Arc<dyn Pool>>> {
        // Group all pools by their token pairs
        HashMap::new()
    }
    
    fn clean_expired_opportunities(&self) {
        let now = Instant::now();
        self.opportunity_cache.retain(|_, opp| opp.valid_until > now);
    }
    
    pub fn get_metrics(&self) -> HashMap<String, u64> {
        let mut metrics = HashMap::new();
        metrics.insert("opportunities_found".to_string(), 
                      self.metrics.opportunities_found.load(Ordering::Relaxed));
        metrics.insert("profitable_opportunities".to_string(),
                      self.metrics.profitable_opportunities.load(Ordering::Relaxed));
        metrics.insert("cycles_scanned".to_string(),
                      self.metrics.cycles_scanned.load(Ordering::Relaxed));
        metrics.insert("avg_scan_time_ms".to_string(),
                      self.metrics.avg_scan_time_ms.load(Ordering::Relaxed));
        metrics.insert("cached_opportunities".to_string(),
                      self.opportunity_cache.len() as u64);
        metrics
    }
}