use std::time::Duration;
use redis::{aio::ConnectionManager, AsyncCommands, Client, RedisResult};
use anyhow::Result;
use tracing::{debug, error, warn};
use serde::{Deserialize, Serialize};
use serde_json;

use crate::metrics_collector::{NodeMetrics, ConnectionHealth};

const METRICS_KEY: &str = "node:metrics:current";
const HEALTH_KEY: &str = "node:health:";
const LATENCY_KEY: &str = "node:latency:";
const TTL_SECONDS: u64 = 60;

pub struct CacheManager {
    conn: ConnectionManager,
}

impl CacheManager {
    pub async fn new(redis_url: &str) -> Result<Self> {
        let client = Client::open(redis_url)?;
        let conn = ConnectionManager::new(client).await?;
        
        Ok(Self { conn })
    }
    
    pub async fn set_metrics(&self, metrics: NodeMetrics) -> Result<()> {
        let mut conn = self.conn.clone();
        let serialized = serde_json::to_string(&metrics)?;
        
        // Use SETEX for automatic expiration
        let _: RedisResult<()> = conn.set_ex(
            METRICS_KEY,
            serialized,
            TTL_SECONDS,
        ).await;
        
        // Also store in time-series format for historical data
        let ts_key = format!("node:metrics:ts:{}", metrics.timestamp);
        let _: RedisResult<()> = conn.set_ex(
            ts_key,
            serde_json::to_string(&metrics)?,
            3600, // Keep for 1 hour
        ).await;
        
        Ok(())
    }
    
    pub async fn get_metrics(&self) -> Result<Option<NodeMetrics>> {
        let mut conn = self.conn.clone();
        
        let result: RedisResult<String> = conn.get(METRICS_KEY).await;
        
        match result {
            Ok(data) => {
                match serde_json::from_str(&data) {
                    Ok(metrics) => Ok(Some(metrics)),
                    Err(e) => {
                        error!("Failed to deserialize metrics: {}", e);
                        Ok(None)
                    }
                }
            }
            Err(_) => Ok(None),
        }
    }
    
    pub async fn set_connection_health(&self, health: &ConnectionHealth) -> Result<()> {
        let mut conn = self.conn.clone();
        let key = format!("{}{}", HEALTH_KEY, health.endpoint);
        let serialized = serde_json::to_string(health)?;
        
        let _: RedisResult<()> = conn.set_ex(
            key,
            serialized,
            TTL_SECONDS,
        ).await;
        
        Ok(())
    }
    
    pub async fn get_connection_health(&self, endpoint: &str) -> Result<Option<ConnectionHealth>> {
        let mut conn = self.conn.clone();
        let key = format!("{}{}", HEALTH_KEY, endpoint);
        
        let result: RedisResult<String> = conn.get(key).await;
        
        match result {
            Ok(data) => {
                match serde_json::from_str(&data) {
                    Ok(health) => Ok(Some(health)),
                    Err(e) => {
                        error!("Failed to deserialize health: {}", e);
                        Ok(None)
                    }
                }
            }
            Err(_) => Ok(None),
        }
    }
    
    pub async fn set_latency_sample(&self, endpoint_type: &str, endpoint: &str, latency_us: u64) -> Result<()> {
        let mut conn = self.conn.clone();
        let key = format!("{}{}:{}", LATENCY_KEY, endpoint_type, endpoint);
        
        // Use Redis sorted set for time-series latency data
        let timestamp = chrono::Utc::now().timestamp_millis() as f64;
        let _: RedisResult<()> = conn.zadd(
            &key,
            latency_us,
            timestamp,
        ).await;
        
        // Keep only last 1000 samples
        let _: RedisResult<()> = conn.zremrangebyrank(
            &key,
            0,
            -1001,
        ).await;
        
        // Set expiration
        let _: RedisResult<()> = conn.expire(
            &key,
            3600,
        ).await;
        
        Ok(())
    }
    
    pub async fn get_latency_samples(&self, endpoint_type: &str, endpoint: &str, count: isize) -> Result<Vec<(f64, u64)>> {
        let mut conn = self.conn.clone();
        let key = format!("{}{}:{}", LATENCY_KEY, endpoint_type, endpoint);
        
        // Get latest samples with scores (timestamps)
        let result: RedisResult<Vec<(String, f64)>> = conn.zrevrange_withscores(
            key,
            0,
            count - 1,
        ).await;
        
        match result {
            Ok(samples) => {
                let parsed: Vec<(f64, u64)> = samples
                    .into_iter()
                    .filter_map(|(member, score)| {
                        member.parse::<u64>().ok().map(|latency| (score, latency))
                    })
                    .collect();
                Ok(parsed)
            }
            Err(_) => Ok(Vec::new()),
        }
    }
    
    pub async fn invalidate_all(&self) -> Result<()> {
        let mut conn = self.conn.clone();
        
        // Delete all node metrics keys
        let pattern = "node:*";
        let keys: RedisResult<Vec<String>> = conn.keys(pattern).await;
        
        if let Ok(keys) = keys {
            if !keys.is_empty() {
                let _: RedisResult<()> = conn.del(keys).await;
            }
        }
        
        Ok(())
    }
    
    /// Perform atomic increment for counters
    pub async fn increment_counter(&self, key: &str, value: i64) -> Result<i64> {
        let mut conn = self.conn.clone();
        let full_key = format!("node:counter:{}", key);
        
        let result: RedisResult<i64> = conn.incr(full_key.clone(), value).await;
        
        // Set expiration on first increment
        if let Ok(count) = result {
            if count == value {
                let _: RedisResult<()> = conn.expire(full_key, 3600).await;
            }
            Ok(count)
        } else {
            Ok(0)
        }
    }
    
    /// Get or set with atomic operation
    pub async fn get_or_set<T>(&self, key: &str, generator: impl FnOnce() -> T) -> Result<T>
    where
        T: Serialize + for<'de> Deserialize<'de>,
    {
        let mut conn = self.conn.clone();
        let full_key = format!("node:cache:{}", key);
        
        // Try to get first
        let result: RedisResult<String> = conn.get(&full_key).await;
        
        match result {
            Ok(data) => {
                match serde_json::from_str(&data) {
                    Ok(value) => Ok(value),
                    Err(_) => {
                        // Generate new value
                        let value = generator();
                        let serialized = serde_json::to_string(&value)?;
                        let _: RedisResult<()> = conn.set_ex(
                            full_key,
                            serialized,
                            TTL_SECONDS,
                        ).await;
                        Ok(value)
                    }
                }
            }
            Err(_) => {
                // Generate new value
                let value = generator();
                let serialized = serde_json::to_string(&value)?;
                let _: RedisResult<()> = conn.set_ex(
                    full_key,
                    serialized,
                    TTL_SECONDS,
                ).await;
                Ok(value)
            }
        }
    }
}