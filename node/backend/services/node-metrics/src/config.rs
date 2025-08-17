use serde::{Deserialize, Serialize};
use anyhow::Result;
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // RPC endpoints
    pub rpc_endpoints: Vec<String>,
    pub ws_endpoints: Vec<String>,
    pub geyser_endpoints: Vec<String>,
    pub jito_endpoints: Vec<String>,
    
    // Service configuration
    pub ws_port: u16,
    pub metrics_port: u16,
    
    // Redis configuration
    pub redis_url: String,
    pub redis_pool_size: u32,
    
    // ClickHouse configuration
    pub clickhouse_url: String,
    pub clickhouse_database: String,
    
    // Performance tuning
    pub health_check_interval_ms: u64,
    pub metrics_collection_interval_ms: u64,
    pub cache_ttl_seconds: u64,
    pub ring_buffer_size: usize,
    
    // Circuit breaker settings
    pub circuit_breaker_threshold: u32,
    pub circuit_breaker_timeout_ms: u64,
    
    // Latency tracking
    pub latency_percentiles: Vec<f64>,
    pub latency_window_size: usize,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        Ok(Config {
            rpc_endpoints: env::var("RPC_ENDPOINTS")
                .unwrap_or_else(|_| "http://localhost:8899".to_string())
                .split(',')
                .map(|s| s.trim().to_string())
                .collect(),
            
            ws_endpoints: env::var("WS_ENDPOINTS")
                .unwrap_or_else(|_| "ws://localhost:8900".to_string())
                .split(',')
                .map(|s| s.trim().to_string())
                .collect(),
            
            geyser_endpoints: env::var("GEYSER_ENDPOINTS")
                .unwrap_or_else(|_| "grpc://localhost:10000".to_string())
                .split(',')
                .map(|s| s.trim().to_string())
                .collect(),
            
            jito_endpoints: env::var("JITO_ENDPOINTS")
                .unwrap_or_else(|_| "grpc://localhost:8008".to_string())
                .split(',')
                .map(|s| s.trim().to_string())
                .collect(),
            
            ws_port: env::var("WS_PORT")
                .unwrap_or_else(|_| "8081".to_string())
                .parse()?,
            
            metrics_port: env::var("METRICS_PORT")
                .unwrap_or_else(|_| "9090".to_string())
                .parse()?,
            
            redis_url: env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string()),
            
            redis_pool_size: env::var("REDIS_POOL_SIZE")
                .unwrap_or_else(|_| "100".to_string())
                .parse()?,
            
            clickhouse_url: env::var("CLICKHOUSE_URL")
                .unwrap_or_else(|_| "http://localhost:8123".to_string()),
            
            clickhouse_database: env::var("CLICKHOUSE_DATABASE")
                .unwrap_or_else(|_| "mev_metrics".to_string()),
            
            health_check_interval_ms: env::var("HEALTH_CHECK_INTERVAL_MS")
                .unwrap_or_else(|_| "100".to_string())
                .parse()?,
            
            metrics_collection_interval_ms: env::var("METRICS_COLLECTION_INTERVAL_MS")
                .unwrap_or_else(|_| "50".to_string())
                .parse()?,
            
            cache_ttl_seconds: env::var("CACHE_TTL_SECONDS")
                .unwrap_or_else(|_| "1".to_string())
                .parse()?,
            
            ring_buffer_size: env::var("RING_BUFFER_SIZE")
                .unwrap_or_else(|_| "10000".to_string())
                .parse()?,
            
            circuit_breaker_threshold: env::var("CIRCUIT_BREAKER_THRESHOLD")
                .unwrap_or_else(|_| "5".to_string())
                .parse()?,
            
            circuit_breaker_timeout_ms: env::var("CIRCUIT_BREAKER_TIMEOUT_MS")
                .unwrap_or_else(|_| "60000".to_string())
                .parse()?,
            
            latency_percentiles: vec![0.5, 0.9, 0.95, 0.99, 0.999],
            
            latency_window_size: env::var("LATENCY_WINDOW_SIZE")
                .unwrap_or_else(|_| "1000".to_string())
                .parse()?,
        })
    }
}