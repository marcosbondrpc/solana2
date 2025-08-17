use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::time::{Duration, Instant};
use tokio::time;
use dashmap::DashMap;
use parking_lot::RwLock;
use crossbeam::queue::ArrayQueue;
use arc_swap::ArcSwap;
use anyhow::Result;
use tracing::{debug, error, info, warn};
use prometheus::{register_gauge_vec, register_histogram_vec, GaugeVec, HistogramVec};
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::clock::Slot;
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::cache_manager::CacheManager;
use crate::latency_tracker::LatencyTracker;
use crate::ring_buffer::RingBuffer;

lazy_static::lazy_static! {
    static ref LATENCY_GAUGE: GaugeVec = register_gauge_vec!(
        "node_latency_microseconds",
        "Node latency in microseconds",
        &["endpoint_type", "endpoint"]
    ).unwrap();
    
    static ref CONNECTION_STATUS: GaugeVec = register_gauge_vec!(
        "node_connection_status",
        "Connection status (1 = healthy, 0 = unhealthy)",
        &["endpoint_type", "endpoint"]
    ).unwrap();
    
    static ref NETWORK_METRICS: GaugeVec = register_gauge_vec!(
        "network_metrics",
        "Network metrics",
        &["metric_type"]
    ).unwrap();
    
    static ref LATENCY_HISTOGRAM: HistogramVec = register_histogram_vec!(
        "node_latency_histogram",
        "Node latency histogram",
        &["endpoint_type", "endpoint"],
        vec![0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    ).unwrap();
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetrics {
    pub timestamp: i64,
    pub slot: u64,
    pub block_height: u64,
    pub tps: f64,
    pub block_time_ms: u64,
    pub peer_count: u32,
    pub leader_schedule: Vec<u64>,
    pub epoch_info: EpochInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochInfo {
    pub epoch: u64,
    pub slot_index: u64,
    pub slots_in_epoch: u64,
    pub absolute_slot: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionHealth {
    pub endpoint: String,
    pub endpoint_type: EndpointType,
    pub is_healthy: bool,
    pub latency_us: u64,
    pub last_check: i64,
    pub consecutive_failures: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EndpointType {
    RPC,
    WebSocket,
    Geyser,
    Jito,
}

pub struct MetricsCollector {
    config: Config,
    cache: Arc<CacheManager>,
    
    // Lock-free metrics storage
    current_slot: Arc<AtomicU64>,
    current_block_height: Arc<AtomicU64>,
    current_tps: Arc<ArcSwap<f64>>,
    block_time_ms: Arc<AtomicU64>,
    peer_count: Arc<AtomicU64>,
    
    // Connection health tracking with lock-free map
    connection_health: Arc<DashMap<String, ConnectionHealth>>,
    
    // Latency tracking with ring buffers
    rpc_latencies: Arc<RwLock<LatencyTracker>>,
    ws_latencies: Arc<RwLock<LatencyTracker>>,
    geyser_latencies: Arc<RwLock<LatencyTracker>>,
    jito_latencies: Arc<RwLock<LatencyTracker>>,
    
    // High-performance queue for metrics events
    metrics_queue: Arc<ArrayQueue<NodeMetrics>>,
    
    // RPC clients pool
    rpc_clients: Arc<Vec<Arc<RpcClient>>>,
}

impl MetricsCollector {
    pub async fn new(config: Config, cache: Arc<CacheManager>) -> Result<Self> {
        // Initialize RPC clients pool
        let mut rpc_clients = Vec::new();
        for endpoint in &config.rpc_endpoints {
            let client = Arc::new(RpcClient::new(endpoint.clone()));
            rpc_clients.push(client);
        }
        
        // Initialize metrics queue with capacity for burst events
        let metrics_queue = Arc::new(ArrayQueue::new(100000));
        
        Ok(Self {
            config: config.clone(),
            cache,
            current_slot: Arc::new(AtomicU64::new(0)),
            current_block_height: Arc::new(AtomicU64::new(0)),
            current_tps: Arc::new(ArcSwap::from_pointee(0.0)),
            block_time_ms: Arc::new(AtomicU64::new(0)),
            peer_count: Arc::new(AtomicU64::new(0)),
            connection_health: Arc::new(DashMap::new()),
            rpc_latencies: Arc::new(RwLock::new(LatencyTracker::new(config.ring_buffer_size))),
            ws_latencies: Arc::new(RwLock::new(LatencyTracker::new(config.ring_buffer_size))),
            geyser_latencies: Arc::new(RwLock::new(LatencyTracker::new(config.ring_buffer_size))),
            jito_latencies: Arc::new(RwLock::new(LatencyTracker::new(config.ring_buffer_size))),
            metrics_queue,
            rpc_clients: Arc::new(rpc_clients),
        })
    }
    
    pub async fn start_collection(self: Arc<Self>) -> Result<()> {
        info!("Starting metrics collection with {}ms interval", self.config.metrics_collection_interval_ms);
        
        let mut interval = time::interval(Duration::from_millis(self.config.metrics_collection_interval_ms));
        interval.set_missed_tick_behavior(time::MissedTickBehavior::Skip);
        
        loop {
            interval.tick().await;
            
            // Spawn concurrent collection tasks
            let collector = self.clone();
            tokio::spawn(async move {
                if let Err(e) = collector.collect_metrics().await {
                    error!("Error collecting metrics: {}", e);
                }
            });
        }
    }
    
    async fn collect_metrics(&self) -> Result<()> {
        let start = Instant::now();
        
        // Collect from multiple RPC endpoints concurrently
        let mut tasks = Vec::new();
        
        for (idx, client) in self.rpc_clients.iter().enumerate() {
            let client = client.clone();
            let endpoint = self.config.rpc_endpoints[idx].clone();
            
            tasks.push(tokio::spawn(async move {
                let latency_start = Instant::now();
                
                // Get slot with timeout
                let slot_result = tokio::time::timeout(
                    Duration::from_millis(1000),
                    client.get_slot()
                ).await;
                
                let latency_us = latency_start.elapsed().as_micros() as u64;
                
                match slot_result {
                    Ok(Ok(slot)) => {
                        Some((endpoint, slot, latency_us, true))
                    }
                    _ => {
                        Some((endpoint, 0, latency_us, false))
                    }
                }
            }));
        }
        
        // Collect results
        let mut best_slot = 0u64;
        let mut best_latency = u64::MAX;
        let mut healthy_count = 0;
        
        for task in tasks {
            if let Ok(Some((endpoint, slot, latency_us, is_healthy))) = task.await {
                // Update connection health
                self.update_connection_health(
                    endpoint.clone(),
                    EndpointType::RPC,
                    is_healthy,
                    latency_us,
                ).await;
                
                if is_healthy {
                    healthy_count += 1;
                    if latency_us < best_latency {
                        best_latency = latency_us;
                        best_slot = slot;
                    }
                }
                
                // Record metrics
                LATENCY_GAUGE
                    .with_label_values(&["rpc", &endpoint])
                    .set(latency_us as f64);
                
                LATENCY_HISTOGRAM
                    .with_label_values(&["rpc", &endpoint])
                    .observe(latency_us as f64 / 1000.0);
                
                CONNECTION_STATUS
                    .with_label_values(&["rpc", &endpoint])
                    .set(if is_healthy { 1.0 } else { 0.0 });
            }
        }
        
        // Update atomic values
        self.current_slot.store(best_slot, Ordering::Relaxed);
        
        // Get additional metrics from best endpoint
        if healthy_count > 0 && !self.rpc_clients.is_empty() {
            let client = &self.rpc_clients[0];
            
            // Get block height
            if let Ok(Ok(height)) = tokio::time::timeout(
                Duration::from_millis(500),
                client.get_block_height()
            ).await {
                self.current_block_height.store(height, Ordering::Relaxed);
                NETWORK_METRICS
                    .with_label_values(&["block_height"])
                    .set(height as f64);
            }
            
            // Get recent performance samples for TPS calculation
            if let Ok(Ok(samples)) = tokio::time::timeout(
                Duration::from_millis(500),
                client.get_recent_performance_samples(Some(1))
            ).await {
                if let Some(sample) = samples.first() {
                    let tps = sample.num_transactions as f64 / sample.sample_period_secs as f64;
                    self.current_tps.store(Arc::new(tps));
                    
                    NETWORK_METRICS
                        .with_label_values(&["tps"])
                        .set(tps);
                }
            }
            
            // Get cluster nodes for peer count
            if let Ok(Ok(nodes)) = tokio::time::timeout(
                Duration::from_millis(500),
                client.get_cluster_nodes()
            ).await {
                let peer_count = nodes.len() as u64;
                self.peer_count.store(peer_count, Ordering::Relaxed);
                
                NETWORK_METRICS
                    .with_label_values(&["peer_count"])
                    .set(peer_count as f64);
            }
        }
        
        // Cache current metrics
        let metrics = self.get_current_metrics().await;
        self.cache.set_metrics(metrics.clone()).await?;
        
        // Push to metrics queue
        let _ = self.metrics_queue.push(metrics);
        
        debug!("Metrics collection completed in {:?}", start.elapsed());
        
        Ok(())
    }
    
    async fn update_connection_health(
        &self,
        endpoint: String,
        endpoint_type: EndpointType,
        is_healthy: bool,
        latency_us: u64,
    ) {
        let mut health = self.connection_health.entry(endpoint.clone()).or_insert_with(|| {
            ConnectionHealth {
                endpoint: endpoint.clone(),
                endpoint_type: endpoint_type.clone(),
                is_healthy: false,
                latency_us: 0,
                last_check: 0,
                consecutive_failures: 0,
            }
        });
        
        health.is_healthy = is_healthy;
        health.latency_us = latency_us;
        health.last_check = chrono::Utc::now().timestamp_millis();
        
        if !is_healthy {
            health.consecutive_failures += 1;
        } else {
            health.consecutive_failures = 0;
        }
        
        // Update latency tracker
        match endpoint_type {
            EndpointType::RPC => {
                self.rpc_latencies.write().add_sample(latency_us);
            }
            EndpointType::WebSocket => {
                self.ws_latencies.write().add_sample(latency_us);
            }
            EndpointType::Geyser => {
                self.geyser_latencies.write().add_sample(latency_us);
            }
            EndpointType::Jito => {
                self.jito_latencies.write().add_sample(latency_us);
            }
        }
    }
    
    pub async fn get_current_metrics(&self) -> NodeMetrics {
        NodeMetrics {
            timestamp: chrono::Utc::now().timestamp_millis(),
            slot: self.current_slot.load(Ordering::Relaxed),
            block_height: self.current_block_height.load(Ordering::Relaxed),
            tps: **self.current_tps.load(),
            block_time_ms: self.block_time_ms.load(Ordering::Relaxed),
            peer_count: self.peer_count.load(Ordering::Relaxed) as u32,
            leader_schedule: Vec::new(), // TODO: Implement leader schedule tracking
            epoch_info: EpochInfo {
                epoch: 0, // TODO: Get from RPC
                slot_index: 0,
                slots_in_epoch: 432000,
                absolute_slot: self.current_slot.load(Ordering::Relaxed),
            },
        }
    }
    
    pub async fn get_connection_health(&self) -> Vec<ConnectionHealth> {
        self.connection_health
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }
    
    pub async fn get_latency_stats(&self, endpoint_type: EndpointType) -> LatencyStats {
        let tracker = match endpoint_type {
            EndpointType::RPC => self.rpc_latencies.read(),
            EndpointType::WebSocket => self.ws_latencies.read(),
            EndpointType::Geyser => self.geyser_latencies.read(),
            EndpointType::Jito => self.jito_latencies.read(),
        };
        
        tracker.get_stats()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub min: u64,
    pub max: u64,
    pub mean: f64,
    pub median: f64,
    pub p95: f64,
    pub p99: f64,
    pub p999: f64,
    pub std_dev: f64,
}