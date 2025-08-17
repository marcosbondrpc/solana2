use anyhow::Result;
use dashmap::DashMap;
use futures::stream::StreamExt;
use parking_lot::RwLock;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};
use solana_client::nonblocking::pubsub_client::PubsubClient;
use solana_client::rpc_config::{RpcTransactionLogsConfig, RpcTransactionLogsFilter};
use solana_sdk::commitment_config::CommitmentConfig;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Signature;
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, Mutex};
use tokio::time::interval;
use tracing::{error, info, warn, debug, trace};

mod dex_interfaces;
mod price_aggregator;
mod pool_registry;
mod arbitrage_engine;
mod risk_metrics;
mod websocket_server;
mod cache_manager;
mod performance_monitor;

use dex_interfaces::*;
use price_aggregator::*;
use pool_registry::*;
use arbitrage_engine::*;
use risk_metrics::*;
use websocket_server::*;
use cache_manager::*;
use performance_monitor::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageOpportunity {
    pub id: String,
    pub timestamp: i64,
    pub block_slot: u64,
    pub token_a: String,
    pub token_b: String,
    pub token_c: Option<String>, // For triangular arb
    pub dex_path: Vec<DexInfo>,
    pub price_discrepancy: Decimal,
    pub estimated_profit_usdc: Decimal,
    pub gas_cost_sol: Decimal,
    pub priority_fee_sol: Decimal,
    pub jito_tip_sol: Decimal,
    pub net_profit_usdc: Decimal,
    pub roi_percentage: Decimal,
    pub confidence_score: f64,
    pub risk_score: f64,
    pub execution_latency_ms: u64,
    pub liquidity_available: Decimal,
    pub slippage_tolerance: Decimal,
    pub mev_competition_level: String,
    pub opportunity_type: ArbitrageType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArbitrageType {
    TwoLeg,
    Triangular,
    MultiLeg(u8),
    CexDex,
    Cyclic,
    FlashLoan,
}

/// Welford's online algorithm for computing running mean and variance
/// This is critical for adaptive threshold calculation in volatile markets
#[derive(Debug, Clone)]
pub struct SpreadStat {
    count: u64,
    mean: f64,
    m2: f64,  // Sum of squares of differences from the current mean
    min_spread: f64,
    max_spread: f64,
    last_update: Instant,
}

impl SpreadStat {
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min_spread: f64::MAX,
            max_spread: f64::MIN,
            last_update: Instant::now(),
        }
    }
    
    /// Update statistics using Welford's algorithm
    /// This maintains numerical stability even with millions of samples
    pub fn update(&mut self, spread: f64) {
        self.count += 1;
        let delta = spread - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = spread - self.mean;
        self.m2 += delta * delta2;
        
        self.min_spread = self.min_spread.min(spread);
        self.max_spread = self.max_spread.max(spread);
        self.last_update = Instant::now();
    }
    
    /// Calculate standard deviation with Bessel's correction
    pub fn stddev(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        (self.m2 / (self.count - 1) as f64).sqrt()
    }
    
    /// Get variance for risk calculations
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }
    
    /// Calculate adaptive threshold using μ + κ·σ formula
    /// κ (kappa) is the sensitivity parameter (typically 1.5-3.0)
    pub fn adaptive_threshold(&self, kappa: f64) -> f64 {
        self.mean + kappa * self.stddev()
    }
    
    /// Check if we have enough samples for reliable statistics
    pub fn is_reliable(&self) -> bool {
        self.count >= 20 && self.last_update.elapsed() < Duration::from_secs(300)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DexInfo {
    pub name: String,
    pub address: String,
    pub pool_address: String,
    pub price: Decimal,
    pub liquidity: Decimal,
    pub fee_bps: u16,
}

pub struct ArbitrageDetector {
    rpc_client: Arc<solana_client::nonblocking::rpc_client::RpcClient>,
    pubsub_client: Arc<Mutex<PubsubClient>>,
    price_feeds: Arc<DashMap<String, PriceFeed>>,
    opportunity_cache: Arc<Cache<String, ArbitrageOpportunity>>,
    risk_analyzer: Arc<RiskAnalyzer>,
    performance_monitor: Arc<PerformanceMonitor>,
    ws_server: Arc<WebSocketServer>,
    opportunity_tx: broadcast::Sender<ArbitrageOpportunity>,
    config: DetectorConfig,
    /// Spread statistics per token pair for adaptive threshold calculation
    /// This is the secret sauce - adapts to market conditions in real-time
    spread_stats: Arc<DashMap<String, SpreadStat>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DetectorConfig {
    pub rpc_endpoint: String,
    pub ws_endpoint: String,
    pub clickhouse_url: String,
    pub redis_url: String,
    pub kafka_brokers: Vec<String>,
    pub min_profit_threshold_usdc: Decimal,
    pub max_slippage_bps: u16,
    pub detection_latency_target_ms: u64,
    pub cache_ttl_seconds: u64,
    pub websocket_port: u16,
}

impl ArbitrageDetector {
    pub async fn new(config: DetectorConfig) -> Result<Self> {
        info!("Initializing Arbitrage Detector with config: {:?}", config);
        
        let rpc_client = Arc::new(
            solana_client::nonblocking::rpc_client::RpcClient::new(
                config.rpc_endpoint.clone()
            )
        );
        
        let pubsub_client = Arc::new(Mutex::new(
            PubsubClient::new(&config.ws_endpoint).await?
        ));
        
        let price_feeds = Arc::new(DashMap::new());
        let opportunity_cache = Arc::new(Cache::new(config.cache_ttl_seconds));
        let risk_analyzer = Arc::new(RiskAnalyzer::new());
        let performance_monitor = Arc::new(PerformanceMonitor::new());
        let ws_server = Arc::new(WebSocketServer::new(config.websocket_port));
        let (opportunity_tx, _) = broadcast::channel(10000);
        let spread_stats = Arc::new(DashMap::new());
        
        Ok(Self {
            rpc_client,
            pubsub_client,
            price_feeds,
            opportunity_cache,
            risk_analyzer,
            performance_monitor,
            ws_server,
            opportunity_tx,
            config,
            spread_stats,
        })
    }
    
    pub async fn start(&self) -> Result<()> {
        info!("Starting Arbitrage Detection Engine");
        
        // Start WebSocket server for real-time streaming
        let ws_handle = self.start_websocket_server();
        
        // Start monitoring all DEXs
        let monitor_handles = vec![
            self.monitor_raydium(),
            self.monitor_orca(),
            self.monitor_phoenix(),
            self.monitor_meteora(),
            self.monitor_openbook(),
            self.monitor_lifinity(),
        ];
        
        // Start arbitrage detection loop
        let detection_handle = self.run_detection_loop();
        
        // Start performance monitoring
        let perf_handle = self.start_performance_monitoring();
        
        // Wait for all tasks
        tokio::try_join!(
            ws_handle,
            detection_handle,
            perf_handle,
            futures::future::try_join_all(monitor_handles),
            self.monitor_pool_accounts(),
        )?;
        
        Ok(())
    }
    
    /// Subscribe to critical pool accounts to reduce reliance on log parsing.
    async fn monitor_pool_accounts(&self) -> Result<()> {
        info!("Starting pool account monitoring");
        // Load registry of important pools/markets
        let registry = PoolRegistry::load_default();

        let client = self.pubsub_client.lock().await;
        let mut streams = Vec::new();
        for entry in &registry.entries {
            let (stream, _sub) = client.account_subscribe(entry.account).await?;
            streams.push((entry.clone(), stream));
        }
        drop(client);

        // Fan-in streams; update price feeds on changes
        loop {
            for (entry, stream) in streams.iter_mut() {
                if let Some(update) = stream.next().await {
                    // Parse account data based on DEX type
                    if let Some(feed) = entry.parse_account_update(&update) {
                        self.price_feeds.insert(feed.token_pair.clone(), feed);
                    }
                }
            }
        }
    }
    
    async fn monitor_raydium(&self) -> Result<()> {
        info!("Starting Raydium AMM monitoring");
        
        let raydium_program = Pubkey::from_str("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8")?;
        let client = self.pubsub_client.lock().await;
        
        let (mut stream, _) = client.logs_subscribe(
            RpcTransactionLogsFilter::Mentions(vec![raydium_program.to_string()]),
            RpcTransactionLogsConfig {
                commitment: Some(CommitmentConfig::confirmed()),
            },
        ).await?;
        
        drop(client); // Release lock
        
        while let Some(log) = stream.next().await {
            let start = Instant::now();
            
            if let Err(e) = self.process_raydium_log(log).await {
                error!("Error processing Raydium log: {}", e);
            }
            
            self.performance_monitor.record_latency(
                "raydium_processing",
                start.elapsed().as_millis() as f64,
            );
        }
        
        Ok(())
    }
    
    async fn process_raydium_log(&self, log: solana_client::rpc_response::RpcLogsResponse) -> Result<()> {
        let signature = log.signature.parse::<Signature>()?;
        debug!("Processing Raydium transaction: {}", signature);
        
        // Parse swap events and update price feeds
        for log_line in &log.logs {
            if log_line.contains("ray_log") {
                if let Some(swap_data) = self.parse_raydium_swap(log_line) {
                    self.update_price_feed("raydium", swap_data).await?;
                }
            }
        }
        
        // Check for arbitrage opportunities
        self.detect_opportunities().await?;
        
        Ok(())
    }
    
    async fn monitor_orca(&self) -> Result<()> {
        info!("Starting Orca Whirlpool monitoring");
        
        let whirlpool_program = Pubkey::from_str("whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc")?;
        let client = self.pubsub_client.lock().await;
        
        let (mut stream, _) = client.logs_subscribe(
            RpcTransactionLogsFilter::Mentions(vec![whirlpool_program.to_string()]),
            RpcTransactionLogsConfig {
                commitment: Some(CommitmentConfig::confirmed()),
            },
        ).await?;
        
        drop(client);
        
        while let Some(log) = stream.next().await {
            let start = Instant::now();
            
            if let Err(e) = self.process_orca_log(log).await {
                error!("Error processing Orca log: {}", e);
            }
            
            self.performance_monitor.record_latency(
                "orca_processing",
                start.elapsed().as_millis() as f64,
            );
        }
        
        Ok(())
    }
    
    /// Monitor Phoenix orderbook DEX - stub implementation
    /// Phoenix is Solana's high-performance on-chain orderbook
    async fn monitor_phoenix(&self) -> Result<()> {
        info!("Starting Phoenix DEX monitoring (stub)");
        // Phoenix program: PhoeNiXZ8ByJGLkxNfZRnkUfjvmuYqLR89jjFHGqdXY
        // TODO: Implement Phoenix-specific monitoring once we have access to IDL
        // Phoenix uses an on-chain orderbook model rather than AMM
        // Key events to monitor: NewOrder, CancelOrder, FillEvent
        debug!("Phoenix monitor stub - awaiting implementation");
        Ok(())
    }
    
    /// Monitor Meteora Dynamic AMM pools - stub implementation
    /// Meteora offers dynamic fees and concentrated liquidity
    async fn monitor_meteora(&self) -> Result<()> {
        info!("Starting Meteora AMM monitoring (stub)");
        // Meteora DLMM program: LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo
        // TODO: Implement Meteora-specific monitoring
        // Key features: Dynamic fees, concentrated liquidity ranges
        // Monitor for: Swap events, liquidity changes, fee tier updates
        debug!("Meteora monitor stub - awaiting implementation");
        Ok(())
    }
    
    /// Monitor OpenBook v2 orderbook - stub implementation
    /// OpenBook is the decentralized central limit order book
    async fn monitor_openbook(&self) -> Result<()> {
        info!("Starting OpenBook v2 monitoring (stub)");
        // OpenBook v2 program: opnbkNkqux64GppQmwFWpuG7y3cNAxHcL9mq3paNbBr
        // TODO: Implement OpenBook v2 monitoring
        // Key events: PlaceOrder, CancelOrder, OrderMatched, Settlement
        // Critical for CEX-DEX arbitrage opportunities
        debug!("OpenBook monitor stub - awaiting implementation");
        Ok(())
    }
    
    /// Monitor Lifinity protocol - stub implementation
    /// Lifinity uses proactive market making with oracle price feeds
    async fn monitor_lifinity(&self) -> Result<()> {
        info!("Starting Lifinity monitoring (stub)");
        // Lifinity program: EewxydAPCCVuNEyrVN68PuSYdQ7wKn27V9Gjeoi8dy3S
        // TODO: Implement Lifinity-specific monitoring
        // Unique feature: Uses oracle prices for reduced impermanent loss
        // Monitor for: Swap events with oracle price divergence
        debug!("Lifinity monitor stub - awaiting implementation");
        Ok(())
    }
    
    async fn detect_opportunities(&self) -> Result<()> {
        let start = Instant::now();
        
        // Get all current price feeds
        let mut prices: HashMap<String, Vec<(String, Decimal, Decimal)>> = HashMap::new();
        
        for entry in self.price_feeds.iter() {
            let token_pair = entry.key().clone();
            let feed = entry.value().clone();
            
            prices.entry(token_pair.clone())
                .or_insert_with(Vec::new)
                .push((feed.dex_name.clone(), feed.price, feed.liquidity));
        }
        
        // Micro-tracing: measure each detection phase
        let two_leg_start = Instant::now();
        
        // Check for 2-leg arbitrage with micro-tracing per token pair
        for (token_pair, dex_prices) in &prices {
            if dex_prices.len() >= 2 {
                let pair_start = Instant::now();
                
                self.check_two_leg_arbitrage(token_pair, dex_prices).await?;
                
                // Micro-trace: log if any pair takes >1ms
                let pair_latency = pair_start.elapsed();
                if pair_latency > Duration::from_millis(1) {
                    trace!(
                        "Slow 2-leg check for {}: {:?}ms",
                        token_pair,
                        pair_latency.as_micros() as f64 / 1000.0
                    );
                }
            }
        }
        
        self.performance_monitor.record_latency(
            "two_leg_scan",
            two_leg_start.elapsed().as_millis() as f64,
        );
        
        // Check for triangular arbitrage
        let triangular_start = Instant::now();
        self.check_triangular_arbitrage(&prices).await?;
        self.performance_monitor.record_latency(
            "triangular_scan",
            triangular_start.elapsed().as_millis() as f64,
        );
        
        // Check for multi-leg opportunities
        let multi_leg_start = Instant::now();
        self.check_multi_leg_arbitrage(&prices).await?;
        self.performance_monitor.record_latency(
            "multi_leg_scan",
            multi_leg_start.elapsed().as_millis() as f64,
        );
        
        self.performance_monitor.record_latency(
            "opportunity_detection",
            start.elapsed().as_millis() as f64,
        );
        
        Ok(())
    }
    
    async fn check_two_leg_arbitrage(
        &self,
        token_pair: &str,
        dex_prices: &[(String, Decimal, Decimal)],
    ) -> Result<()> {
        let mut max_price = Decimal::ZERO;
        let mut min_price = Decimal::MAX;
        let mut max_dex = "";
        let mut min_dex = "";
        let mut max_liquidity = Decimal::ZERO;
        let mut min_liquidity = Decimal::ZERO;
        
        for (dex, price, liquidity) in dex_prices {
            if *price > max_price {
                max_price = *price;
                max_dex = dex;
                max_liquidity = *liquidity;
            }
            if *price < min_price {
                min_price = *price;
                min_dex = dex;
                min_liquidity = *liquidity;
            }
        }
        
        let price_diff = max_price - min_price;
        let price_diff_percentage = (price_diff / min_price) * Decimal::from(100);
        
        // Update spread statistics for adaptive threshold
        let spread_percentage = price_diff_percentage.to_f64().unwrap_or(0.0);
        self.spread_stats
            .entry(token_pair.to_string())
            .or_insert_with(SpreadStat::new)
            .update(spread_percentage);
        
        // Calculate adaptive threshold using μ + κ·σ formula
        // κ = 2.0 gives us ~95% confidence (2 standard deviations)
        let kappa = 2.0;
        let threshold = if let Some(stats) = self.spread_stats.get(token_pair) {
            if stats.is_reliable() {
                // Use adaptive threshold based on historical spread distribution
                let adaptive_threshold = stats.adaptive_threshold(kappa);
                debug!(
                    "Token pair {}: μ={:.4}%, σ={:.4}%, threshold={:.4}%, current_spread={:.4}%",
                    token_pair,
                    stats.mean,
                    stats.stddev(),
                    adaptive_threshold,
                    spread_percentage
                );
                
                // Never go below 0.5% to avoid false positives from fees
                // This is our safety floor - MEV bots need to cover gas + priority fees
                Decimal::from_f64(adaptive_threshold.max(0.5))
                    .unwrap_or(Decimal::from_str("0.5")?)
            } else {
                // Not enough data yet, use conservative fixed threshold
                debug!(
                    "Token pair {}: Insufficient data (n={}), using fixed threshold",
                    token_pair,
                    stats.count
                );
                Decimal::from_str("0.5")?
            }
        } else {
            // Fallback to fixed threshold if no stats available
            Decimal::from_str("0.5")?
        };
        
        // Check if profitable after fees and gas using adaptive threshold
        if price_diff_percentage > threshold {
            trace!(
                "Arbitrage opportunity detected! Pair: {}, Spread: {:.4}%, Threshold: {:.4}%",
                token_pair,
                price_diff_percentage,
                threshold
            );
            
            let opportunity = self.calculate_arbitrage_profit(
                token_pair,
                min_dex,
                max_dex,
                min_price,
                max_price,
                min_liquidity.min(max_liquidity),
                ArbitrageType::TwoLeg,
            ).await?;
            
            if opportunity.net_profit_usdc > self.config.min_profit_threshold_usdc {
                self.broadcast_opportunity(opportunity).await?;
            }
        }
        
        Ok(())
    }
    
    async fn calculate_arbitrage_profit(
        &self,
        token_pair: &str,
        buy_dex: &str,
        sell_dex: &str,
        buy_price: Decimal,
        sell_price: Decimal,
        available_liquidity: Decimal,
        arb_type: ArbitrageType,
    ) -> Result<ArbitrageOpportunity> {
        let tokens: Vec<&str> = token_pair.split('/').collect();
        let token_a = tokens[0].to_string();
        let token_b = tokens[1].to_string();
        
        // Calculate optimal trade size considering slippage
        let optimal_size = self.calculate_optimal_trade_size(
            available_liquidity,
            buy_price,
            sell_price,
        ).await?;
        
        // Estimate gas costs
        let gas_cost = self.estimate_gas_cost(&arb_type).await?;
        let priority_fee = self.calculate_priority_fee().await?;
        let jito_tip = self.calculate_jito_tip(&arb_type).await?;
        
        // Calculate revenue
        let revenue = optimal_size * (sell_price - buy_price);
        
        // Calculate net profit
        let total_costs = gas_cost + priority_fee + jito_tip;
        let net_profit = revenue - total_costs;
        
        // Calculate ROI
        let investment = optimal_size * buy_price;
        let roi = (net_profit / investment) * Decimal::from(100);
        
        // Risk assessment
        let risk_score = self.risk_analyzer.calculate_risk_score(
            &token_a,
            &token_b,
            buy_dex,
            sell_dex,
        ).await?;
        
        Ok(ArbitrageOpportunity {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().timestamp(),
            block_slot: self.get_current_slot().await?,
            token_a,
            token_b,
            token_c: None,
            dex_path: vec![
                DexInfo {
                    name: buy_dex.to_string(),
                    address: self.get_dex_address(buy_dex),
                    pool_address: self.get_pool_address(token_pair, buy_dex),
                    price: buy_price,
                    liquidity: available_liquidity,
                    fee_bps: self.get_dex_fee_bps(buy_dex),
                },
                DexInfo {
                    name: sell_dex.to_string(),
                    address: self.get_dex_address(sell_dex),
                    pool_address: self.get_pool_address(token_pair, sell_dex),
                    price: sell_price,
                    liquidity: available_liquidity,
                    fee_bps: self.get_dex_fee_bps(sell_dex),
                },
            ],
            price_discrepancy: sell_price - buy_price,
            estimated_profit_usdc: revenue,
            gas_cost_sol: gas_cost,
            priority_fee_sol: priority_fee,
            jito_tip_sol: jito_tip,
            net_profit_usdc: net_profit,
            roi_percentage: roi,
            confidence_score: self.calculate_confidence_score(&arb_type, &risk_score),
            risk_score,
            execution_latency_ms: 50, // Target latency
            liquidity_available: available_liquidity,
            slippage_tolerance: Decimal::from_str("0.005")?, // 0.5%
            mev_competition_level: self.assess_mev_competition().await?,
            opportunity_type: arb_type,
        })
    }
    
    async fn broadcast_opportunity(&self, opportunity: ArbitrageOpportunity) -> Result<()> {
        info!("Broadcasting arbitrage opportunity: {:?}", opportunity);
        
        // Send to WebSocket subscribers
        self.opportunity_tx.send(opportunity.clone())?;
        
        // Store in cache
        self.opportunity_cache.insert(
            opportunity.id.clone(),
            opportunity.clone(),
        ).await;
        
        // Send to Kafka for downstream processing
        self.send_to_kafka(opportunity).await?;
        
        Ok(())
    }
    
    async fn run_detection_loop(&self) -> Result<()> {
        let mut interval = interval(Duration::from_millis(10)); // 10ms detection loop
        
        loop {
            interval.tick().await;
            
            if let Err(e) = self.detect_opportunities().await {
                error!("Error in detection loop: {}", e);
            }
        }
    }
    
    async fn start_websocket_server(&self) -> tokio::task::JoinHandle<Result<()>> {
        let ws_server = self.ws_server.clone();
        let mut opportunity_rx = self.opportunity_tx.subscribe();
        
        tokio::spawn(async move {
            ws_server.start().await?;
            
            while let Ok(opportunity) = opportunity_rx.recv().await {
                ws_server.broadcast_opportunity(opportunity).await?;
            }
            
            Ok(())
        })
    }
    
    async fn start_performance_monitoring(&self) -> tokio::task::JoinHandle<Result<()>> {
        let monitor = self.performance_monitor.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                monitor.report_metrics().await;
            }
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("arbitrage_detector=debug,info")
        .json()
        .init();
    
    let config = DetectorConfig {
        rpc_endpoint: std::env::var("SOLANA_RPC_URL")
            .unwrap_or_else(|_| "https://api.mainnet-beta.solana.com".to_string()),
        ws_endpoint: std::env::var("SOLANA_WS_URL")
            .unwrap_or_else(|_| "wss://api.mainnet-beta.solana.com".to_string()),
        clickhouse_url: std::env::var("CLICKHOUSE_URL")
            .unwrap_or_else(|_| "http://localhost:8123".to_string()),
        redis_url: std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://localhost:6390".to_string()),
        kafka_brokers: std::env::var("KAFKA_BROKERS")
            .unwrap_or_else(|_| "localhost:9092".to_string())
            .split(',')
            .map(|s| s.to_string())
            .collect(),
        min_profit_threshold_usdc: Decimal::from_str("10.0")?,
        max_slippage_bps: 50,
        detection_latency_target_ms: 50,
        cache_ttl_seconds: 300,
        websocket_port: 8080,
    };
    
    let detector = ArbitrageDetector::new(config).await?;
    detector.start().await?;
    
    Ok(())
}