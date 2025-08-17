use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub rpc_url: String,
    pub ws_url: String,
    pub rpc_timeout_ms: u64,
    pub max_concurrent_bundles: usize,
    pub bundle_timeout_ms: u64,
    pub mempool_scan_interval_ms: u64,
    pub opportunity_scan_interval_ms: u64,
    pub clickhouse_url: String,
    pub redis_url: String,
    pub kafka_brokers: Vec<String>,
}

impl Config {
    pub fn load(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            rpc_url: "https://api.mainnet-beta.solana.com".to_string(),
            ws_url: "wss://api.mainnet-beta.solana.com".to_string(),
            rpc_timeout_ms: 5000,
            max_concurrent_bundles: 100,
            bundle_timeout_ms: 50,
            mempool_scan_interval_ms: 10,
            opportunity_scan_interval_ms: 5,
            clickhouse_url: "http://localhost:8123".to_string(),
            redis_url: "redis://localhost:6390".to_string(),
            kafka_brokers: vec!["localhost:9092".to_string()],
        }
    }
}