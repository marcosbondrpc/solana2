use ahash::AHashMap;
use anyhow::Result;
use arc_swap::ArcSwap;
use bytes::Bytes;
use dashmap::DashMap;
use ethnum::U256;
use indexmap::IndexMap;
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use petgraph::{
    algo::{bellman_ford, dijkstra},
    graph::{DiGraph, NodeIndex},
    visit::EdgeRef,
};
use priority_queue::PriorityQueue;
use rayon::prelude::*;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use solana_sdk::pubkey::Pubkey;
use std::{
    cmp::Reverse,
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tracing::{debug, error, info, warn};

// Constants for optimization
const MAX_HOPS: usize = 3;
const GRAPH_UPDATE_INTERVAL_MS: u64 = 100;
const CACHE_TTL_MS: u64 = 50;
const MIN_LIQUIDITY_THRESHOLD: u64 = 1_000_000; // $1 minimum
const MAX_PRICE_IMPACT: f64 = 0.05; // 5% max price impact
const SIMD_BATCH_SIZE: usize = 16;

// Unified Pool trait for all DEXes
pub trait Pool: Send + Sync {
    fn address(&self) -> Pubkey;
    fn token_a(&self) -> Pubkey;
    fn token_b(&self) -> Pubkey;
    fn reserve_a(&self) -> u64;
    fn reserve_b(&self) -> u64;
    fn fee_bps(&self) -> u16;
    fn dex_type(&self) -> DexType;
    
    // Core pricing functions
    fn get_amount_out(&self, amount_in: u64, token_in: Pubkey) -> Result<u64>;
    fn get_amount_in(&self, amount_out: u64, token_out: Pubkey) -> Result<u64>;
    fn calculate_price_impact(&self, amount_in: u64, token_in: Pubkey) -> f64;
    
    // Advanced features
    fn supports_flash_loan(&self) -> bool { false }
    fn concentrated_liquidity_range(&self) -> Option<(Decimal, Decimal)> { None }
    fn virtual_reserves(&self) -> Option<(u64, u64)> { None }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DexType {
    Raydium,
    Orca,
    OrcaWhirlpool,
    Phoenix,
    Meteora,
    Lifinity,
    Saber,
    Mercurial,
    Crema,
    Aldrin,
    Serum,
    OpenBook,
}

// Raydium AMM V4 Pool Implementation
#[derive(Debug, Clone)]
pub struct RaydiumPool {
    pub address: Pubkey,
    pub amm_id: Pubkey,
    pub token_a: Pubkey,
    pub token_b: Pubkey,
    pub reserve_a: AtomicU64,
    pub reserve_b: AtomicU64,
    pub fee_numerator: u64,
    pub fee_denominator: u64,
}

impl Pool for RaydiumPool {
    fn address(&self) -> Pubkey { self.address }
    fn token_a(&self) -> Pubkey { self.token_a }
    fn token_b(&self) -> Pubkey { self.token_b }
    fn reserve_a(&self) -> u64 { self.reserve_a.load(Ordering::Relaxed) }
    fn reserve_b(&self) -> u64 { self.reserve_b.load(Ordering::Relaxed) }
    fn fee_bps(&self) -> u16 { 
        ((self.fee_numerator * 10000) / self.fee_denominator) as u16 
    }
    fn dex_type(&self) -> DexType { DexType::Raydium }
    
    fn get_amount_out(&self, amount_in: u64, token_in: Pubkey) -> Result<u64> {
        let (reserve_in, reserve_out) = if token_in == self.token_a {
            (self.reserve_a(), self.reserve_b())
        } else {
            (self.reserve_b(), self.reserve_a())
        };
        
        if reserve_in == 0 || reserve_out == 0 {
            return Ok(0);
        }
        
        // Apply fee
        let amount_in_with_fee = (amount_in as u128) * (self.fee_denominator - self.fee_numerator) as u128;
        let numerator = amount_in_with_fee * reserve_out as u128;
        let denominator = (reserve_in as u128 * self.fee_denominator as u128) + amount_in_with_fee;
        
        Ok((numerator / denominator) as u64)
    }
    
    fn get_amount_in(&self, amount_out: u64, token_out: Pubkey) -> Result<u64> {
        let (reserve_in, reserve_out) = if token_out == self.token_b {
            (self.reserve_a(), self.reserve_b())
        } else {
            (self.reserve_b(), self.reserve_a())
        };
        
        if reserve_in == 0 || reserve_out == 0 || amount_out >= reserve_out {
            return Ok(u64::MAX);
        }
        
        let numerator = (reserve_in as u128) * (amount_out as u128) * (self.fee_denominator as u128);
        let denominator = (reserve_out as u128 - amount_out as u128) * (self.fee_denominator - self.fee_numerator) as u128;
        
        Ok(((numerator / denominator) + 1) as u64)
    }
    
    fn calculate_price_impact(&self, amount_in: u64, token_in: Pubkey) -> f64 {
        let (reserve_in, reserve_out) = if token_in == self.token_a {
            (self.reserve_a(), self.reserve_b())
        } else {
            (self.reserve_b(), self.reserve_a())
        };
        
        if reserve_in == 0 || reserve_out == 0 {
            return 1.0;
        }
        
        let ideal_price = reserve_out as f64 / reserve_in as f64;
        let amount_out = self.get_amount_out(amount_in, token_in).unwrap_or(0);
        let actual_price = amount_out as f64 / amount_in as f64;
        
        (ideal_price - actual_price).abs() / ideal_price
    }
}

// Orca Whirlpool Implementation (Concentrated Liquidity)
#[derive(Debug, Clone)]
pub struct OrcaWhirlpool {
    pub address: Pubkey,
    pub token_a: Pubkey,
    pub token_b: Pubkey,
    pub sqrt_price: u128,
    pub liquidity: u128,
    pub fee_rate: u16,
    pub tick_spacing: u16,
    pub current_tick: i32,
    pub tick_array: Vec<TickInfo>,
}

#[derive(Debug, Clone)]
pub struct TickInfo {
    pub index: i32,
    pub liquidity_net: i128,
    pub liquidity_gross: u128,
    pub fee_growth_outside_a: u128,
    pub fee_growth_outside_b: u128,
}

impl Pool for OrcaWhirlpool {
    fn address(&self) -> Pubkey { self.address }
    fn token_a(&self) -> Pubkey { self.token_a }
    fn token_b(&self) -> Pubkey { self.token_b }
    
    fn reserve_a(&self) -> u64 {
        // Calculate virtual reserves from sqrt price and liquidity
        let virtual_reserve = (self.liquidity as u128 * self.sqrt_price) >> 64;
        virtual_reserve.min(u64::MAX as u128) as u64
    }
    
    fn reserve_b(&self) -> u64 {
        let virtual_reserve = (self.liquidity as u128 << 64) / self.sqrt_price;
        virtual_reserve.min(u64::MAX as u128) as u64
    }
    
    fn fee_bps(&self) -> u16 { self.fee_rate }
    fn dex_type(&self) -> DexType { DexType::OrcaWhirlpool }
    
    fn get_amount_out(&self, amount_in: u64, token_in: Pubkey) -> Result<u64> {
        // Concentrated liquidity swap simulation
        let zero_for_one = token_in == self.token_a;
        let mut amount_remaining = amount_in as i128;
        let mut amount_calculated = 0i128;
        let mut sqrt_price_current = self.sqrt_price;
        let mut liquidity = self.liquidity;
        let mut tick = self.current_tick;
        
        // Simulate swap through ticks
        while amount_remaining > 0 {
            // Find next initialized tick
            let next_tick = self.get_next_initialized_tick(tick, zero_for_one);
            let sqrt_price_next = Self::sqrt_price_from_tick(next_tick);
            
            // Compute swap step
            let (amount_in_step, amount_out_step, sqrt_price_new) = Self::compute_swap_step(
                sqrt_price_current,
                sqrt_price_next,
                liquidity,
                amount_remaining,
                self.fee_rate,
                zero_for_one,
            );
            
            amount_remaining -= amount_in_step;
            amount_calculated += amount_out_step;
            sqrt_price_current = sqrt_price_new;
            
            // Cross tick if necessary
            if sqrt_price_current == sqrt_price_next {
                tick = next_tick;
                // Update liquidity when crossing tick
                if let Some(tick_info) = self.tick_array.iter().find(|t| t.index == tick) {
                    if zero_for_one {
                        liquidity = liquidity.saturating_sub(tick_info.liquidity_net.abs() as u128);
                    } else {
                        liquidity = liquidity.saturating_add(tick_info.liquidity_net.abs() as u128);
                    }
                }
            }
            
            // Break if we've swapped enough or run out of liquidity
            if liquidity == 0 {
                break;
            }
        }
        
        Ok(amount_calculated.abs() as u64)
    }
    
    fn get_amount_in(&self, amount_out: u64, token_out: Pubkey) -> Result<u64> {
        // Inverse calculation for concentrated liquidity
        // This is complex and would require tick traversal in reverse
        // Simplified implementation
        let fee_adjusted = (amount_out as u128 * 10000) / (10000 - self.fee_rate as u128);
        Ok(fee_adjusted as u64)
    }
    
    fn calculate_price_impact(&self, amount_in: u64, token_in: Pubkey) -> f64 {
        let amount_out = self.get_amount_out(amount_in, token_in).unwrap_or(0);
        let spot_price = self.get_spot_price(token_in == self.token_a);
        let execution_price = amount_out as f64 / amount_in as f64;
        (spot_price - execution_price).abs() / spot_price
    }
    
    fn concentrated_liquidity_range(&self) -> Option<(Decimal, Decimal)> {
        let min_price = Self::price_from_tick(self.current_tick - 100);
        let max_price = Self::price_from_tick(self.current_tick + 100);
        Some((
            Decimal::from_f64_retain(min_price).unwrap_or_default(),
            Decimal::from_f64_retain(max_price).unwrap_or_default(),
        ))
    }
}

impl OrcaWhirlpool {
    fn get_next_initialized_tick(&self, current: i32, zero_for_one: bool) -> i32 {
        // Find next initialized tick in direction
        let mut next = if zero_for_one {
            current - self.tick_spacing as i32
        } else {
            current + self.tick_spacing as i32
        };
        
        while !self.tick_array.iter().any(|t| t.index == next) {
            next = if zero_for_one {
                next - self.tick_spacing as i32
            } else {
                next + self.tick_spacing as i32
            };
            
            // Bounds check
            if next < -443636 || next > 443636 {
                break;
            }
        }
        
        next
    }
    
    fn sqrt_price_from_tick(tick: i32) -> u128 {
        // sqrt(1.0001^tick) * 2^64
        let base: f64 = 1.0001_f64.powi(tick);
        let sqrt = base.sqrt();
        ((sqrt * (1u128 << 64) as f64) as u128).max(1)
    }
    
    fn price_from_tick(tick: i32) -> f64 {
        1.0001_f64.powi(tick)
    }
    
    fn compute_swap_step(
        sqrt_price_current: u128,
        sqrt_price_target: u128,
        liquidity: u128,
        amount_remaining: i128,
        fee_rate: u16,
        zero_for_one: bool,
    ) -> (i128, i128, u128) {
        // Concentrated liquidity math for single step
        // This is simplified - real implementation requires exact math
        let sqrt_price_new = if zero_for_one {
            sqrt_price_current.min(sqrt_price_target)
        } else {
            sqrt_price_current.max(sqrt_price_target)
        };
        
        let amount_in = amount_remaining.min(liquidity as i128 / 1000);
        let amount_out = (amount_in * (10000 - fee_rate as i128)) / 10000;
        
        (amount_in, amount_out, sqrt_price_new)
    }
    
    fn get_spot_price(&self, zero_for_one: bool) -> f64 {
        let price = (self.sqrt_price as f64 / (1u64 << 64) as f64).powi(2);
        if zero_for_one {
            price
        } else {
            1.0 / price
        }
    }
}

// Pricing Graph for multi-hop routing
pub struct PricingGraph {
    graph: Arc<ArcSwap<DiGraph<TokenNode, PoolEdge>>>,
    token_to_node: Arc<DashMap<Pubkey, NodeIndex>>,
    pool_registry: Arc<DashMap<Pubkey, Arc<dyn Pool>>>,
    path_cache: Arc<DashMap<(Pubkey, Pubkey), Vec<SwapPath>>>,
    metrics: Arc<GraphMetrics>,
    last_update: Arc<RwLock<Instant>>,
}

#[derive(Debug, Clone)]
struct TokenNode {
    mint: Pubkey,
    symbol: String,
    decimals: u8,
}

#[derive(Debug, Clone)]
struct PoolEdge {
    pool_address: Pubkey,
    dex_type: DexType,
    fee_bps: u16,
    liquidity: u64,
    last_update: Instant,
}

#[derive(Debug, Clone)]
pub struct SwapPath {
    pub route: Vec<SwapLeg>,
    pub expected_out: u64,
    pub price_impact: f64,
    pub total_fee_bps: u16,
}

#[derive(Debug, Clone)]
pub struct SwapLeg {
    pub pool_address: Pubkey,
    pub dex_type: DexType,
    pub token_in: Pubkey,
    pub token_out: Pubkey,
    pub amount_in: u64,
    pub amount_out: u64,
}

struct GraphMetrics {
    nodes: AtomicU64,
    edges: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    path_calculations: AtomicU64,
    avg_path_time_us: AtomicU64,
}

impl PricingGraph {
    pub fn new() -> Self {
        Self {
            graph: Arc::new(ArcSwap::from_pointee(DiGraph::new())),
            token_to_node: Arc::new(DashMap::new()),
            pool_registry: Arc::new(DashMap::new()),
            path_cache: Arc::new(DashMap::new()),
            metrics: Arc::new(GraphMetrics {
                nodes: AtomicU64::new(0),
                edges: AtomicU64::new(0),
                cache_hits: AtomicU64::new(0),
                cache_misses: AtomicU64::new(0),
                path_calculations: AtomicU64::new(0),
                avg_path_time_us: AtomicU64::new(0),
            }),
            last_update: Arc::new(RwLock::new(Instant::now())),
        }
    }
    
    pub fn register_pool(&self, pool: Arc<dyn Pool>) {
        // Add pool to registry
        self.pool_registry.insert(pool.address(), pool.clone());
        
        // Update graph
        let mut graph = DiGraph::clone(&self.graph.load());
        
        // Ensure tokens exist as nodes
        let token_a = pool.token_a();
        let token_b = pool.token_b();
        
        let node_a = self.get_or_create_node(&mut graph, token_a);
        let node_b = self.get_or_create_node(&mut graph, token_b);
        
        // Add bidirectional edges
        let edge = PoolEdge {
            pool_address: pool.address(),
            dex_type: pool.dex_type(),
            fee_bps: pool.fee_bps(),
            liquidity: pool.reserve_a() + pool.reserve_b(),
            last_update: Instant::now(),
        };
        
        graph.add_edge(node_a, node_b, edge.clone());
        graph.add_edge(node_b, node_a, edge);
        
        // Atomic swap of graph
        self.graph.store(Arc::new(graph));
        
        // Update metrics
        self.metrics.edges.fetch_add(2, Ordering::Relaxed);
        
        // Clear affected cache entries
        self.invalidate_cache_for_tokens(&[token_a, token_b]);
    }
    
    pub fn find_best_path(
        &self,
        token_in: Pubkey,
        token_out: Pubkey,
        amount_in: u64,
        max_hops: usize,
    ) -> Result<SwapPath> {
        let start = Instant::now();
        
        // Check cache first
        let cache_key = (token_in, token_out);
        if let Some(cached_paths) = self.path_cache.get(&cache_key) {
            if cached_paths.len() > 0 && start.duration_since(*self.last_update.read()).as_millis() < CACHE_TTL_MS as u128 {
                self.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(cached_paths[0].clone());
            }
        }
        
        self.metrics.cache_misses.fetch_add(1, Ordering::Relaxed);
        
        // Find all paths up to max_hops
        let paths = self.find_all_paths(token_in, token_out, max_hops.min(MAX_HOPS))?;
        
        // Simulate and rank paths
        let mut evaluated_paths: Vec<SwapPath> = paths
            .par_iter()
            .filter_map(|path| self.simulate_path(path, amount_in).ok())
            .collect();
        
        // Sort by expected output (descending)
        evaluated_paths.sort_by(|a, b| b.expected_out.cmp(&a.expected_out));
        
        if evaluated_paths.is_empty() {
            return Err(anyhow::anyhow!("No valid paths found"));
        }
        
        // Cache top paths
        self.path_cache.insert(cache_key, evaluated_paths.clone());
        
        // Update metrics
        let elapsed = start.elapsed().as_micros() as u64;
        self.metrics.path_calculations.fetch_add(1, Ordering::Relaxed);
        self.metrics.avg_path_time_us.store(elapsed, Ordering::Relaxed);
        
        Ok(evaluated_paths[0].clone())
    }
    
    fn find_all_paths(
        &self,
        token_in: Pubkey,
        token_out: Pubkey,
        max_hops: usize,
    ) -> Result<Vec<Vec<Pubkey>>> {
        let graph = self.graph.load();
        
        let start_node = self.token_to_node.get(&token_in)
            .ok_or_else(|| anyhow::anyhow!("Token {} not in graph", token_in))?
            .clone();
        
        let end_node = self.token_to_node.get(&token_out)
            .ok_or_else(|| anyhow::anyhow!("Token {} not in graph", token_out))?
            .clone();
        
        let mut all_paths = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back((vec![start_node], HashSet::new()));
        
        while let Some((path, visited)) = queue.pop_front() {
            if path.len() > max_hops + 1 {
                continue;
            }
            
            let current = *path.last().unwrap();
            
            if current == end_node {
                // Convert node path to token path
                let token_path: Vec<Pubkey> = path
                    .iter()
                    .map(|&node| graph[node].mint)
                    .collect();
                all_paths.push(token_path);
                continue;
            }
            
            // Explore neighbors
            for edge in graph.edges(current) {
                let neighbor = edge.target();
                if !visited.contains(&neighbor) {
                    let mut new_path = path.clone();
                    new_path.push(neighbor);
                    let mut new_visited = visited.clone();
                    new_visited.insert(neighbor);
                    queue.push_back((new_path, new_visited));
                }
            }
        }
        
        Ok(all_paths)
    }
    
    fn simulate_path(&self, token_path: &[Pubkey], amount_in: u64) -> Result<SwapPath> {
        let mut route = Vec::new();
        let mut current_amount = amount_in;
        let mut total_fee_bps = 0u16;
        let mut total_price_impact = 0.0;
        
        for i in 0..token_path.len() - 1 {
            let token_in = token_path[i];
            let token_out = token_path[i + 1];
            
            // Find best pool for this hop
            let pool = self.find_best_pool(token_in, token_out, current_amount)?;
            
            let amount_out = pool.get_amount_out(current_amount, token_in)?;
            let price_impact = pool.calculate_price_impact(current_amount, token_in);
            
            route.push(SwapLeg {
                pool_address: pool.address(),
                dex_type: pool.dex_type(),
                token_in,
                token_out,
                amount_in: current_amount,
                amount_out,
            });
            
            current_amount = amount_out;
            total_fee_bps = total_fee_bps.saturating_add(pool.fee_bps());
            total_price_impact += price_impact;
            
            // Early exit if price impact too high
            if total_price_impact > MAX_PRICE_IMPACT {
                return Err(anyhow::anyhow!("Price impact too high"));
            }
        }
        
        Ok(SwapPath {
            route,
            expected_out: current_amount,
            price_impact: total_price_impact,
            total_fee_bps,
        })
    }
    
    fn find_best_pool(
        &self,
        token_a: Pubkey,
        token_b: Pubkey,
        amount_in: u64,
    ) -> Result<Arc<dyn Pool>> {
        let mut best_pool = None;
        let mut best_output = 0u64;
        
        // Check all pools that connect these tokens
        for entry in self.pool_registry.iter() {
            let pool = entry.value();
            
            // Check if pool connects the tokens
            if (pool.token_a() == token_a && pool.token_b() == token_b) ||
               (pool.token_a() == token_b && pool.token_b() == token_a) {
                
                // Check liquidity threshold
                if pool.reserve_a() + pool.reserve_b() < MIN_LIQUIDITY_THRESHOLD {
                    continue;
                }
                
                // Calculate output
                if let Ok(output) = pool.get_amount_out(amount_in, token_a) {
                    if output > best_output {
                        best_output = output;
                        best_pool = Some(pool.clone());
                    }
                }
            }
        }
        
        best_pool.ok_or_else(|| anyhow::anyhow!("No pool found for {} -> {}", token_a, token_b))
    }
    
    fn get_or_create_node(&self, graph: &mut DiGraph<TokenNode, PoolEdge>, mint: Pubkey) -> NodeIndex {
        if let Some(node) = self.token_to_node.get(&mint) {
            return *node;
        }
        
        let node = TokenNode {
            mint,
            symbol: String::new(), // Would be fetched from metadata
            decimals: 9, // Default for SPL tokens
        };
        
        let index = graph.add_node(node);
        self.token_to_node.insert(mint, index);
        self.metrics.nodes.fetch_add(1, Ordering::Relaxed);
        
        index
    }
    
    fn invalidate_cache_for_tokens(&self, tokens: &[Pubkey]) {
        // Remove cache entries involving these tokens
        self.path_cache.retain(|key, _| {
            !tokens.contains(&key.0) && !tokens.contains(&key.1)
        });
    }
    
    pub fn update_pool_liquidity(&self, pool_address: Pubkey, reserve_a: u64, reserve_b: u64) {
        if let Some(mut pool) = self.pool_registry.get_mut(&pool_address) {
            // Update reserves based on pool type
            // This would need type-specific handling
            // For now, we'll skip as it requires downcasting
        }
        
        *self.last_update.write() = Instant::now();
    }
    
    pub fn get_metrics(&self) -> HashMap<String, u64> {
        let mut metrics = HashMap::new();
        metrics.insert("nodes".to_string(), self.metrics.nodes.load(Ordering::Relaxed));
        metrics.insert("edges".to_string(), self.metrics.edges.load(Ordering::Relaxed));
        metrics.insert("cache_hits".to_string(), self.metrics.cache_hits.load(Ordering::Relaxed));
        metrics.insert("cache_misses".to_string(), self.metrics.cache_misses.load(Ordering::Relaxed));
        metrics.insert("path_calculations".to_string(), self.metrics.path_calculations.load(Ordering::Relaxed));
        metrics.insert("avg_path_time_us".to_string(), self.metrics.avg_path_time_us.load(Ordering::Relaxed));
        metrics.insert("registered_pools".to_string(), self.pool_registry.len() as u64);
        metrics
    }
}

// Benchmark target: resolve 10k pairs < 5ms median
pub fn benchmark_pricing_graph(graph: &PricingGraph, pairs: Vec<(Pubkey, Pubkey, u64)>) -> Duration {
    let start = Instant::now();
    
    pairs.par_iter().for_each(|(token_in, token_out, amount)| {
        let _ = graph.find_best_path(*token_in, *token_out, *amount, 3);
    });
    
    start.elapsed()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_raydium_pool() {
        let pool = RaydiumPool {
            address: Pubkey::new_unique(),
            amm_id: Pubkey::new_unique(),
            token_a: Pubkey::new_unique(),
            token_b: Pubkey::new_unique(),
            reserve_a: AtomicU64::new(1_000_000_000),
            reserve_b: AtomicU64::new(2_000_000_000),
            fee_numerator: 25,
            fee_denominator: 10000,
        };
        
        let amount_out = pool.get_amount_out(1_000_000, pool.token_a).unwrap();
        assert!(amount_out > 0);
        
        let price_impact = pool.calculate_price_impact(10_000_000, pool.token_a);
        assert!(price_impact < 0.01); // Less than 1% for small trade
    }
    
    #[test]
    fn test_pricing_graph() {
        let graph = PricingGraph::new();
        
        // Create test pools
        let token_a = Pubkey::new_unique();
        let token_b = Pubkey::new_unique();
        let token_c = Pubkey::new_unique();
        
        let pool1 = Arc::new(RaydiumPool {
            address: Pubkey::new_unique(),
            amm_id: Pubkey::new_unique(),
            token_a,
            token_b,
            reserve_a: AtomicU64::new(1_000_000_000),
            reserve_b: AtomicU64::new(2_000_000_000),
            fee_numerator: 25,
            fee_denominator: 10000,
        });
        
        let pool2 = Arc::new(RaydiumPool {
            address: Pubkey::new_unique(),
            amm_id: Pubkey::new_unique(),
            token_a: token_b,
            token_b: token_c,
            reserve_a: AtomicU64::new(2_000_000_000),
            reserve_b: AtomicU64::new(3_000_000_000),
            fee_numerator: 25,
            fee_denominator: 10000,
        });
        
        graph.register_pool(pool1);
        graph.register_pool(pool2);
        
        // Find path A -> C (should go through B)
        let path = graph.find_best_path(token_a, token_c, 1_000_000, 3).unwrap();
        assert_eq!(path.route.len(), 2);
        assert!(path.expected_out > 0);
    }
}