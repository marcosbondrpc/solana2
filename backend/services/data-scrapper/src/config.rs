use serde::{Deserialize, Serialize};
use anyhow::Result;
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // RPC configuration
    pub rpc_endpoint: String,
    pub rpc_timeout_ms: u64,
    
    // API configuration
    pub api_port: u16,
    
    // ClickHouse configuration
    pub clickhouse_url: String,
    pub clickhouse_database: String,
    pub clickhouse_username: String,
    pub clickhouse_password: String,
    pub clickhouse_batch_size: usize,
    
    // Redis configuration
    pub redis_url: String,
    
    // Kafka configuration
    pub kafka_brokers: String,
    pub kafka_topic_progress: String,
    pub kafka_topic_metrics: String,
    
    // Scrapper configuration
    pub scrapper_workers: usize,
    pub scrapper_batch_size: usize,
    pub scrapper_retry_attempts: u32,
    pub scrapper_retry_delay_ms: u64,
    
    // Storage configuration
    pub storage_path: String,
    pub dataset_path: String,
    pub model_path: String,
    
    // ML configuration
    pub ml_batch_size: usize,
    pub ml_learning_rate: f32,
    pub ml_epochs: usize,
    pub ml_validation_split: f32,
    
    // Performance tuning
    pub max_concurrent_requests: usize,
    pub queue_size: usize,
    pub cache_ttl_seconds: u64,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        Ok(Config {
            rpc_endpoint: env::var("RPC_ENDPOINT")
                .unwrap_or_else(|_| "https://api.mainnet-beta.solana.com".to_string()),
            
            rpc_timeout_ms: env::var("RPC_TIMEOUT_MS")
                .unwrap_or_else(|_| "30000".to_string())
                .parse()?,
            
            api_port: env::var("API_PORT")
                .unwrap_or_else(|_| "8082".to_string())
                .parse()?,
            
            clickhouse_url: env::var("CLICKHOUSE_URL")
                .unwrap_or_else(|_| "http://localhost:8123".to_string()),
            
            clickhouse_database: env::var("CLICKHOUSE_DATABASE")
                .unwrap_or_else(|_| "solana_data".to_string()),
            
            clickhouse_username: env::var("CLICKHOUSE_USERNAME")
                .unwrap_or_else(|_| "default".to_string()),
            
            clickhouse_password: env::var("CLICKHOUSE_PASSWORD")
                .unwrap_or_else(|_| "".to_string()),
            
            clickhouse_batch_size: env::var("CLICKHOUSE_BATCH_SIZE")
                .unwrap_or_else(|_| "1000".to_string())
                .parse()?,
            
            redis_url: env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string()),
            
            kafka_brokers: env::var("KAFKA_BROKERS")
                .unwrap_or_else(|_| "localhost:9092".to_string()),
            
            kafka_topic_progress: env::var("KAFKA_TOPIC_PROGRESS")
                .unwrap_or_else(|_| "scrapper-progress".to_string()),
            
            kafka_topic_metrics: env::var("KAFKA_TOPIC_METRICS")
                .unwrap_or_else(|_| "scrapper-metrics".to_string()),
            
            scrapper_workers: env::var("SCRAPPER_WORKERS")
                .unwrap_or_else(|_| "16".to_string())
                .parse()?,
            
            scrapper_batch_size: env::var("SCRAPPER_BATCH_SIZE")
                .unwrap_or_else(|_| "100".to_string())
                .parse()?,
            
            scrapper_retry_attempts: env::var("SCRAPPER_RETRY_ATTEMPTS")
                .unwrap_or_else(|_| "3".to_string())
                .parse()?,
            
            scrapper_retry_delay_ms: env::var("SCRAPPER_RETRY_DELAY_MS")
                .unwrap_or_else(|_| "1000".to_string())
                .parse()?,
            
            storage_path: env::var("STORAGE_PATH")
                .unwrap_or_else(|_| "/tmp/scrapper".to_string()),
            
            dataset_path: env::var("DATASET_PATH")
                .unwrap_or_else(|_| "/tmp/scrapper/datasets".to_string()),
            
            model_path: env::var("MODEL_PATH")
                .unwrap_or_else(|_| "/tmp/scrapper/models".to_string()),
            
            ml_batch_size: env::var("ML_BATCH_SIZE")
                .unwrap_or_else(|_| "256".to_string())
                .parse()?,
            
            ml_learning_rate: env::var("ML_LEARNING_RATE")
                .unwrap_or_else(|_| "0.001".to_string())
                .parse()?,
            
            ml_epochs: env::var("ML_EPOCHS")
                .unwrap_or_else(|_| "100".to_string())
                .parse()?,
            
            ml_validation_split: env::var("ML_VALIDATION_SPLIT")
                .unwrap_or_else(|_| "0.2".to_string())
                .parse()?,
            
            max_concurrent_requests: env::var("MAX_CONCURRENT_REQUESTS")
                .unwrap_or_else(|_| "100".to_string())
                .parse()?,
            
            queue_size: env::var("QUEUE_SIZE")
                .unwrap_or_else(|_| "10000".to_string())
                .parse()?,
            
            cache_ttl_seconds: env::var("CACHE_TTL_SECONDS")
                .unwrap_or_else(|_| "60".to_string())
                .parse()?,
        })
    }
}