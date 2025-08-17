use anyhow::Result;
use clickhouse::Client;
use futures::StreamExt;
use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};
use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::pubkey::Pubkey;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::signal;
use tokio::sync::RwLock;
use tokio::time::interval;
use tracing::{error, info, warn};
use uuid::Uuid;

mod arbitrage_engine;
mod cache_manager;
mod performance_monitor;

use arbitrage_engine::ArbitrageEngine;
use cache_manager::Cache;
use performance_monitor::PerformanceMonitor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageOpportunity {
    pub opportunity_id: String,
    pub timestamp: u64,
    pub slot: u64,
    pub source_dex: String,
    pub target_dex: String,
    pub token_a: String,
    pub token_b: String,
    pub source_price: Decimal,
    pub target_price: Decimal,
    pub price_diff_percent: f64,
    pub estimated_profit_sol: Decimal,
    pub estimated_gas_cost: Decimal,
    pub net_profit_sol: Decimal,
    pub confidence_score: f32,
}

#[derive(Clone)]
pub struct ArbitrageDetector {
    rpc_client: Arc<RpcClient>,
    redis: Arc<RwLock<ConnectionManager>>,
    clickhouse: Arc<Client>,
    kafka_producer: Arc<FutureProducer>,
    cache: Arc<Cache>,
    engine: Arc<ArbitrageEngine>,
    monitor: Arc<PerformanceMonitor>,
}

impl ArbitrageDetector {
    pub async fn new(config: Config) -> Result<Self> {
        // Initialize RPC client
        let rpc_client = Arc::new(RpcClient::new(config.solana_rpc_url.clone()));

        // Initialize Redis
        let redis_client = redis::Client::open(format!("redis://{}:{}", config.redis_host, config.redis_port))?;
        let redis_conn = redis_client.get_tokio_connection_manager().await?;
        let redis = Arc::new(RwLock::new(redis_conn));

        // Initialize ClickHouse
        let clickhouse = Arc::new(Client::default()
            .with_url(&config.clickhouse_url)
            .with_database(&config.clickhouse_database));

        // Initialize Kafka producer
        let kafka_producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", &config.kafka_brokers)
            .set("message.timeout.ms", "5000")
            .set("compression.type", "lz4")
            .create()?;
        let kafka_producer = Arc::new(kafka_producer);

        // Initialize components
        let cache = Arc::new(Cache::new());
        let engine = Arc::new(ArbitrageEngine::new(
            Decimal::from_str(&config.min_profit_threshold_sol)?
        ));
        let monitor = Arc::new(PerformanceMonitor::new()?);

        Ok(Self {
            rpc_client,
            redis,
            clickhouse,
            kafka_producer,
            cache,
            engine,
            monitor,
        })
    }

    pub async fn start(&self) -> Result<()> {
        info!("Starting Arbitrage Detector Service");

        // Start metrics server
        let monitor = self.monitor.clone();
        tokio::spawn(async move {
            if let Err(e) = monitor.start_metrics_server(9092).await {
                error!("Failed to start metrics server: {}", e);
            }
        });

        // Start main detection loop
        let mut interval = interval(Duration::from_millis(100));
        
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.detect_opportunities().await {
                        error!("Error detecting opportunities: {}", e);
                    }
                }
                _ = signal::ctrl_c() => {
                    info!("Shutting down arbitrage detector");
                    break;
                }
            }
        }

        Ok(())
    }

    async fn detect_opportunities(&self) -> Result<()> {
        // Get current slot
        let slot = self.rpc_client.get_slot().await?;
        
        // Fetch pool states from cache or RPC
        let pool_states = self.fetch_pool_states().await?;
        
        // Find arbitrage opportunities
        let opportunities = self.engine.find_arbitrage_opportunities(pool_states).await?;
        
        // Process and broadcast opportunities
        for opportunity in opportunities {
            self.monitor.record_opportunity_found();
            
            // Send to Kafka
            if let Err(e) = self.broadcast_opportunity(&opportunity).await {
                error!("Failed to broadcast opportunity: {}", e);
            }
            
            // Store in ClickHouse
            if let Err(e) = self.store_opportunity(&opportunity).await {
                error!("Failed to store opportunity: {}", e);
            }
            
            // Cache the opportunity
            self.cache.set_opportunity(
                opportunity.opportunity_id.clone(),
                cache_manager::OpportunityCacheEntry {
                    id: opportunity.opportunity_id.clone(),
                    profit: opportunity.net_profit_sol,
                    executed: false,
                    timestamp: opportunity.timestamp as i64,
                }
            ).await;
        }
        
        Ok(())
    }

    async fn fetch_pool_states(&self) -> Result<Vec<arbitrage_engine::PoolState>> {
        // Fetch from Redis cache or RPC
        let mut redis = self.redis.write().await;
        let pools: Vec<String> = redis.smembers("tracked_pools").await?;
        
        let mut states = Vec::new();
        for pool_str in pools {
            if let Ok(pool_pubkey) = Pubkey::from_str(&pool_str) {
                // Try cache first
                if let Some(cached) = self.cache.get_pool(&pool_pubkey).await {
                    states.push(arbitrage_engine::PoolState {
                        pool_address: pool_pubkey,
                        token_a: cached.token_a,
                        token_b: cached.token_b,
                        reserve_a: cached.reserves.0,
                        reserve_b: cached.reserves.1,
                        fee_bps: 30, // Default fee
                    });
                } else {
                    // Fetch from RPC if not cached
                    // This would normally fetch actual account data
                    warn!("Pool {} not in cache, skipping", pool_str);
                }
            }
        }
        
        Ok(states)
    }

    async fn broadcast_opportunity(&self, opportunity: &ArbitrageOpportunity) -> Result<()> {
        let payload = serde_json::to_string(opportunity)?;
        let record = FutureRecord::to("arbitrage-events")
            .key(&opportunity.opportunity_id)
            .payload(&payload);
        
        self.kafka_producer
            .send(record, Duration::from_secs(1))
            .await
            .map_err(|(e, _)| anyhow::anyhow!("Kafka send error: {}", e))?;
        
        Ok(())
    }

    async fn store_opportunity(&self, opportunity: &ArbitrageOpportunity) -> Result<()> {
        let query = r#"
            INSERT INTO mev_data.arbitrage_opportunities
            (timestamp, opportunity_id, slot, source_dex, target_dex, 
             token_a, token_b, source_price, target_price, price_diff_percent,
             estimated_profit_sol, estimated_gas_cost, net_profit_sol, 
             confidence_score, execution_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        "#;
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_nanos() as u64;
        
        // Note: ClickHouse insert would go here
        // For now, just log it
        info!("Would store opportunity: {}", opportunity.opportunity_id);
        
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
struct Config {
    solana_rpc_url: String,
    redis_host: String,
    redis_port: u16,
    clickhouse_url: String,
    clickhouse_database: String,
    kafka_brokers: String,
    min_profit_threshold_sol: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            solana_rpc_url: std::env::var("SOLANA_RPC_URL")
                .unwrap_or_else(|_| "https://api.mainnet-beta.solana.com".to_string()),
            redis_host: std::env::var("REDIS_HOST")
                .unwrap_or_else(|_| "localhost".to_string()),
            redis_port: 6379,
            clickhouse_url: std::env::var("CLICKHOUSE_URL")
                .unwrap_or_else(|_| "http://localhost:8123".to_string()),
            clickhouse_database: "mev_data".to_string(),
            kafka_brokers: std::env::var("KAFKA_BROKERS")
                .unwrap_or_else(|_| "localhost:9092".to_string()),
            min_profit_threshold_sol: "0.01".to_string(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "info".to_string())
        )
        .json()
        .init();

    info!("MEV Arbitrage Detector starting up");

    // Load configuration
    let config = Config::default();
    
    // Create and start detector
    let detector = ArbitrageDetector::new(config).await?;
    detector.start().await?;

    Ok(())
}