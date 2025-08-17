use crate::error::{MissionControlError, Result};
use crate::models::*;
use bincode;
use chrono::{DateTime, Duration, Utc};
use redis::aio::ConnectionManager;
use redis::{AsyncCommands, Client};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info, warn};

const CACHE_TTL_SECONDS: i64 = 5; // 5 second cache for real-time data
const CACHE_TTL_LONG_SECONDS: i64 = 60; // 1 minute for less volatile data

#[derive(Clone)]
pub struct CacheManager {
    redis_conn: Arc<ConnectionManager>,
}

impl CacheManager {
    pub async fn new(redis_url: &str) -> Result<Self> {
        let client = Client::open(redis_url)
            .map_err(|e| MissionControlError::CacheError(e))?;
        
        let conn = ConnectionManager::new(client).await
            .map_err(|e| MissionControlError::CacheError(e))?;
        
        info!("Connected to Redis cache");
        
        Ok(Self {
            redis_conn: Arc::new(conn),
        })
    }
    
    // Generic cache operations
    async fn get_cached<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Result<Option<T>> {
        let mut conn = self.redis_conn.as_ref().clone();
        
        let data: Option<Vec<u8>> = conn.get(key).await
            .map_err(|e| MissionControlError::CacheError(e))?;
        
        match data {
            Some(bytes) => {
                let value = bincode::deserialize(&bytes)
                    .map_err(|e| MissionControlError::SerializationError(e))?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }
    
    async fn set_cached<T: Serialize>(&self, key: &str, value: &T, ttl: i64) -> Result<()> {
        let mut conn = self.redis_conn.as_ref().clone();
        
        let bytes = bincode::serialize(value)
            .map_err(|e| MissionControlError::SerializationError(e))?;
        
        conn.set_ex(key, bytes, ttl as u64).await
            .map_err(|e| MissionControlError::CacheError(e))?;
        
        Ok(())
    }
    
    // Mission Control specific cache operations
    pub async fn get_node_summary(&self) -> Result<Option<NodeSummary>> {
        self.get_cached("mc:node_summary").await
    }
    
    pub async fn set_node_summary(&self, summary: &NodeSummary) -> Result<()> {
        self.set_cached("mc:node_summary", summary, CACHE_TTL_SECONDS).await
    }
    
    pub async fn get_consensus_health(&self) -> Result<Option<ConsensusHealth>> {
        self.get_cached("mc:consensus_health").await
    }
    
    pub async fn set_consensus_health(&self, health: &ConsensusHealth) -> Result<()> {
        self.set_cached("mc:consensus_health", health, CACHE_TTL_SECONDS).await
    }
    
    pub async fn get_cluster_performance(&self) -> Result<Option<ClusterPerformance>> {
        self.get_cached("mc:cluster_performance").await
    }
    
    pub async fn set_cluster_performance(&self, perf: &ClusterPerformance) -> Result<()> {
        self.set_cached("mc:cluster_performance", perf, CACHE_TTL_SECONDS).await
    }
    
    pub async fn get_jito_status(&self) -> Result<Option<JitoStatus>> {
        self.get_cached("mc:jito_status").await
    }
    
    pub async fn set_jito_status(&self, status: &JitoStatus) -> Result<()> {
        self.set_cached("mc:jito_status", status, CACHE_TTL_SECONDS).await
    }
    
    pub async fn get_rpc_metrics(&self) -> Result<Option<RpcMetrics>> {
        self.get_cached("mc:rpc_metrics").await
    }
    
    pub async fn set_rpc_metrics(&self, metrics: &RpcMetrics) -> Result<()> {
        self.set_cached("mc:rpc_metrics", metrics, CACHE_TTL_SECONDS).await
    }
    
    pub async fn get_timing_waterfall(&self) -> Result<Option<TimingWaterfall>> {
        self.get_cached("mc:timing_waterfall").await
    }
    
    pub async fn set_timing_waterfall(&self, waterfall: &TimingWaterfall) -> Result<()> {
        self.set_cached("mc:timing_waterfall", waterfall, CACHE_TTL_SECONDS).await
    }
    
    pub async fn get_tip_intelligence(&self) -> Result<Option<TipIntelligence>> {
        self.get_cached("mc:tip_intelligence").await
    }
    
    pub async fn set_tip_intelligence(&self, intel: &TipIntelligence) -> Result<()> {
        self.set_cached("mc:tip_intelligence", intel, CACHE_TTL_LONG_SECONDS).await
    }
    
    pub async fn get_bundle_success(&self) -> Result<Option<BundleSuccess>> {
        self.get_cached("mc:bundle_success").await
    }
    
    pub async fn set_bundle_success(&self, success: &BundleSuccess) -> Result<()> {
        self.set_cached("mc:bundle_success", success, CACHE_TTL_SECONDS).await
    }
    
    pub async fn get_quic_health(&self) -> Result<Option<QuicHealth>> {
        self.get_cached("mc:quic_health").await
    }
    
    pub async fn set_quic_health(&self, health: &QuicHealth) -> Result<()> {
        self.set_cached("mc:quic_health", health, CACHE_TTL_SECONDS).await
    }
    
    pub async fn get_qos_peering(&self) -> Result<Option<QosPeering>> {
        self.get_cached("mc:qos_peering").await
    }
    
    pub async fn set_qos_peering(&self, peering: &QosPeering) -> Result<()> {
        self.set_cached("mc:qos_peering", peering, CACHE_TTL_SECONDS).await
    }
    
    pub async fn get_gossip_metrics(&self) -> Result<Option<GossipMetrics>> {
        self.get_cached("mc:gossip_metrics").await
    }
    
    pub async fn set_gossip_metrics(&self, metrics: &GossipMetrics) -> Result<()> {
        self.set_cached("mc:gossip_metrics", metrics, CACHE_TTL_SECONDS).await
    }
    
    pub async fn get_overview(&self) -> Result<Option<MissionControlOverview>> {
        self.get_cached("mc:overview").await
    }
    
    pub async fn set_overview(&self, overview: &MissionControlOverview) -> Result<()> {
        self.set_cached("mc:overview", overview, CACHE_TTL_SECONDS).await
    }
    
    // Batch operations for performance
    pub async fn get_all_cached_metrics(&self) -> Result<CachedMetrics> {
        let mut conn = self.redis_conn.as_ref().clone();
        
        // Use Redis pipeline for batch fetching
        let keys = vec![
            "mc:node_summary",
            "mc:consensus_health",
            "mc:cluster_performance",
            "mc:jito_status",
            "mc:rpc_metrics",
            "mc:quic_health",
            "mc:qos_peering",
        ];
        
        let mut cached_metrics = CachedMetrics::default();
        
        for key in keys {
            let data: Option<Vec<u8>> = conn.get(key).await.ok().flatten();
            
            if let Some(bytes) = data {
                match key {
                    "mc:node_summary" => {
                        cached_metrics.node_summary = bincode::deserialize(&bytes).ok();
                    }
                    "mc:consensus_health" => {
                        cached_metrics.consensus_health = bincode::deserialize(&bytes).ok();
                    }
                    "mc:cluster_performance" => {
                        cached_metrics.cluster_performance = bincode::deserialize(&bytes).ok();
                    }
                    "mc:jito_status" => {
                        cached_metrics.jito_status = bincode::deserialize(&bytes).ok();
                    }
                    "mc:rpc_metrics" => {
                        cached_metrics.rpc_metrics = bincode::deserialize(&bytes).ok();
                    }
                    "mc:quic_health" => {
                        cached_metrics.quic_health = bincode::deserialize(&bytes).ok();
                    }
                    "mc:qos_peering" => {
                        cached_metrics.qos_peering = bincode::deserialize(&bytes).ok();
                    }
                    _ => {}
                }
            }
        }
        
        Ok(cached_metrics)
    }
    
    // Invalidation helpers
    pub async fn invalidate_all(&self) -> Result<()> {
        let mut conn = self.redis_conn.as_ref().clone();
        
        let pattern = "mc:*";
        let keys: Vec<String> = conn.keys(pattern).await
            .map_err(|e| MissionControlError::CacheError(e))?;
        
        if !keys.is_empty() {
            conn.del(keys).await
                .map_err(|e| MissionControlError::CacheError(e))?;
        }
        
        info!("Invalidated all cache entries");
        Ok(())
    }
    
    pub async fn invalidate_key(&self, key: &str) -> Result<()> {
        let mut conn = self.redis_conn.as_ref().clone();
        
        conn.del(key).await
            .map_err(|e| MissionControlError::CacheError(e))?;
        
        Ok(())
    }
    
    // Health check
    pub async fn health_check(&self) -> bool {
        let mut conn = self.redis_conn.as_ref().clone();
        
        match redis::cmd("PING").query_async::<_, String>(&mut conn).await {
            Ok(response) => response == "PONG",
            Err(e) => {
                error!("Redis health check failed: {}", e);
                false
            }
        }
    }
}

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct CachedMetrics {
    pub node_summary: Option<NodeSummary>,
    pub consensus_health: Option<ConsensusHealth>,
    pub cluster_performance: Option<ClusterPerformance>,
    pub jito_status: Option<JitoStatus>,
    pub rpc_metrics: Option<RpcMetrics>,
    pub quic_health: Option<QuicHealth>,
    pub qos_peering: Option<QosPeering>,
    pub timestamp: DateTime<Utc>,
}

impl CachedMetrics {
    pub fn is_stale(&self, max_age: Duration) -> bool {
        Utc::now() - self.timestamp > max_age
    }
}