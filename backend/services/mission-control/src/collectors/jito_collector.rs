use crate::error::{MissionControlError, Result};
use crate::metrics::MetricsRecorder;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use futures::StreamExt;
use parking_lot::RwLock;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::interval;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{error, info, warn};

const JITO_API_BASE: &str = "https://mainnet.block-engine.jito.wtf/api/v1";
const TIP_STREAM_WS: &str = "wss://mainnet.block-engine.jito.wtf/api/v1/tips";
const BUNDLE_STREAM_WS: &str = "wss://mainnet.block-engine.jito.wtf/api/v1/bundles";
const AUCTION_TICK_MS: u64 = 50;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JitoTip {
    pub slot: u64,
    pub timestamp: DateTime<Utc>,
    pub tip_lamports: u64,
    pub tipper: String,
    pub bundle_id: Option<String>,
    pub region: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JitoBundleStatus {
    pub bundle_id: String,
    pub status: BundleStatus,
    pub timestamp: DateTime<Utc>,
    pub simulation_time_ms: Option<f64>,
    pub rejection_reason: Option<String>,
    pub landed_slot: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BundleStatus {
    Pending,
    Simulated,
    Accepted,
    Rejected,
    Landed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ShredStreamMetrics {
    pub packets_per_second: u64,
    pub gaps_detected: u64,
    pub reorders_detected: u64,
    pub latency_samples: Vec<f64>,
}

pub struct JitoCollector {
    http_client: Client,
    metrics_recorder: Arc<MetricsRecorder>,
    tip_history: Arc<RwLock<VecDeque<JitoTip>>>,
    bundle_history: Arc<DashMap<String, JitoBundleStatus>>,
    shredstream_metrics: Arc<RwLock<ShredStreamMetrics>>,
    regions: Vec<JitoRegion>,
    tip_receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<JitoTip>>>>,
    bundle_receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<JitoBundleStatus>>>>,
}

#[derive(Clone)]
struct JitoRegion {
    name: String,
    endpoint: String,
    grpc_endpoint: String,
    latency: Arc<RwLock<f64>>,
    connected: Arc<RwLock<bool>>,
}

impl JitoCollector {
    pub fn new(metrics_recorder: Arc<MetricsRecorder>) -> Result<Self> {
        let http_client = Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .map_err(|e| MissionControlError::JitoError(e.to_string()))?;

        let regions = vec![
            JitoRegion {
                name: "frankfurt".to_string(),
                endpoint: "https://frankfurt.mainnet.block-engine.jito.wtf".to_string(),
                grpc_endpoint: "grpc://frankfurt.mainnet.block-engine.jito.wtf:443".to_string(),
                latency: Arc::new(RwLock::new(0.0)),
                connected: Arc::new(RwLock::new(false)),
            },
            JitoRegion {
                name: "amsterdam".to_string(),
                endpoint: "https://amsterdam.mainnet.block-engine.jito.wtf".to_string(),
                grpc_endpoint: "grpc://amsterdam.mainnet.block-engine.jito.wtf:443".to_string(),
                latency: Arc::new(RwLock::new(0.0)),
                connected: Arc::new(RwLock::new(false)),
            },
            JitoRegion {
                name: "tokyo".to_string(),
                endpoint: "https://tokyo.mainnet.block-engine.jito.wtf".to_string(),
                grpc_endpoint: "grpc://tokyo.mainnet.block-engine.jito.wtf:443".to_string(),
                latency: Arc::new(RwLock::new(0.0)),
                connected: Arc::new(RwLock::new(false)),
            },
            JitoRegion {
                name: "ny".to_string(),
                endpoint: "https://ny.mainnet.block-engine.jito.wtf".to_string(),
                grpc_endpoint: "grpc://ny.mainnet.block-engine.jito.wtf:443".to_string(),
                latency: Arc::new(RwLock::new(0.0)),
                connected: Arc::new(RwLock::new(false)),
            },
        ];

        Ok(Self {
            http_client,
            metrics_recorder,
            tip_history: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            bundle_history: Arc::new(DashMap::new()),
            shredstream_metrics: Arc::new(RwLock::new(ShredStreamMetrics {
                packets_per_second: 0,
                gaps_detected: 0,
                reorders_detected: 0,
                latency_samples: Vec::new(),
            })),
            regions,
            tip_receiver: Arc::new(RwLock::new(None)),
            bundle_receiver: Arc::new(RwLock::new(None)),
        })
    }

    pub async fn start_collection(&self) {
        // Start WebSocket connections for tips and bundles
        let (tip_tx, tip_rx) = mpsc::unbounded_channel();
        let (bundle_tx, bundle_rx) = mpsc::unbounded_channel();
        
        *self.tip_receiver.write() = Some(tip_rx);
        *self.bundle_receiver.write() = Some(bundle_rx);
        
        // Spawn tip stream handler
        let tip_tx_clone = tip_tx.clone();
        tokio::spawn(async move {
            if let Err(e) = Self::connect_tip_stream(tip_tx_clone).await {
                error!("Tip stream connection failed: {}", e);
            }
        });
        
        // Spawn bundle stream handler
        let bundle_tx_clone = bundle_tx.clone();
        tokio::spawn(async move {
            if let Err(e) = Self::connect_bundle_stream(bundle_tx_clone).await {
                error!("Bundle stream connection failed: {}", e);
            }
        });
        
        // Start region monitoring
        for region in &self.regions {
            let region_clone = region.clone();
            let metrics = Arc::clone(&self.metrics_recorder);
            
            tokio::spawn(async move {
                Self::monitor_region(region_clone, metrics).await;
            });
        }
        
        // Start ShredStream monitoring
        let shredstream_metrics = Arc::clone(&self.shredstream_metrics);
        let metrics = Arc::clone(&self.metrics_recorder);
        tokio::spawn(async move {
            Self::monitor_shredstream(shredstream_metrics, metrics).await;
        });
        
        // Main collection loop
        let mut ticker = interval(Duration::from_secs(1));
        
        loop {
            ticker.tick().await;
            
            // Process incoming tips
            if let Some(ref mut receiver) = *self.tip_receiver.write() {
                while let Ok(tip) = receiver.try_recv() {
                    self.process_tip(tip);
                }
            }
            
            // Process bundle statuses
            if let Some(ref mut receiver) = *self.bundle_receiver.write() {
                while let Ok(status) = receiver.try_recv() {
                    self.process_bundle_status(status);
                }
            }
            
            // Calculate aggregate metrics
            self.calculate_tip_metrics();
            self.calculate_bundle_metrics();
            self.update_shredstream_metrics();
        }
    }

    async fn connect_tip_stream(tx: mpsc::UnboundedSender<JitoTip>) -> Result<()> {
        let (ws_stream, _) = connect_async(TIP_STREAM_WS)
            .await
            .map_err(|e| MissionControlError::WebSocketError(e.to_string()))?;
        
        let (_, mut read) = ws_stream.split();
        
        info!("Connected to Jito tip stream");
        
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(tip) = serde_json::from_str::<JitoTip>(&text) {
                        let _ = tx.send(tip);
                    }
                }
                Ok(Message::Close(_)) => {
                    warn!("Tip stream closed, reconnecting...");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    return Self::connect_tip_stream(tx).await;
                }
                Err(e) => {
                    error!("Tip stream error: {}", e);
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    return Self::connect_tip_stream(tx).await;
                }
                _ => {}
            }
        }
        
        Ok(())
    }

    async fn connect_bundle_stream(tx: mpsc::UnboundedSender<JitoBundleStatus>) -> Result<()> {
        let (ws_stream, _) = connect_async(BUNDLE_STREAM_WS)
            .await
            .map_err(|e| MissionControlError::WebSocketError(e.to_string()))?;
        
        let (_, mut read) = ws_stream.split();
        
        info!("Connected to Jito bundle stream");
        
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(status) = serde_json::from_str::<JitoBundleStatus>(&text) {
                        let _ = tx.send(status);
                    }
                }
                Ok(Message::Close(_)) => {
                    warn!("Bundle stream closed, reconnecting...");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    return Self::connect_bundle_stream(tx).await;
                }
                Err(e) => {
                    error!("Bundle stream error: {}", e);
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    return Self::connect_bundle_stream(tx).await;
                }
                _ => {}
            }
        }
        
        Ok(())
    }

    async fn monitor_region(region: JitoRegion, metrics: Arc<MetricsRecorder>) {
        let client = Client::new();
        let mut ticker = interval(Duration::from_secs(5));
        
        loop {
            ticker.tick().await;
            
            let start = Instant::now();
            let health_url = format!("{}/health", region.endpoint);
            
            match client.get(&health_url).send().await {
                Ok(response) => {
                    let latency = start.elapsed().as_secs_f64() * 1000.0;
                    *region.latency.write() = latency;
                    *region.connected.write() = response.status().is_success();
                    
                    metrics.record_jito_region_latency(&region.name, latency);
                    
                    if response.status().is_success() {
                        info!("Jito region {} healthy, latency: {:.2}ms", region.name, latency);
                    } else {
                        warn!("Jito region {} unhealthy: {}", region.name, response.status());
                    }
                }
                Err(e) => {
                    *region.connected.write() = false;
                    error!("Failed to check Jito region {} health: {}", region.name, e);
                }
            }
        }
    }

    async fn monitor_shredstream(
        metrics: Arc<RwLock<ShredStreamMetrics>>,
        recorder: Arc<MetricsRecorder>,
    ) {
        let mut ticker = interval(Duration::from_millis(100));
        let mut last_packet_count = 0u64;
        let mut packet_sequence = Vec::new();
        
        loop {
            ticker.tick().await;
            
            // Simulate ShredStream metrics (in production, connect to actual ShredStream)
            let current_packets = recorder.get_shredstream_packets();
            let pps = (current_packets - last_packet_count) * 10; // Convert to per second
            
            // Detect gaps and reorders
            packet_sequence.push(current_packets);
            if packet_sequence.len() > 100 {
                packet_sequence.remove(0);
            }
            
            let gaps = Self::detect_gaps(&packet_sequence);
            let reorders = Self::detect_reorders(&packet_sequence);
            
            // Update metrics
            let mut metrics = metrics.write();
            metrics.packets_per_second = pps;
            metrics.gaps_detected = gaps;
            metrics.reorders_detected = reorders;
            
            // Record latency sample
            let latency = recorder.get_shredstream_latency();
            metrics.latency_samples.push(latency);
            
            if metrics.latency_samples.len() > 1000 {
                metrics.latency_samples.remove(0);
            }
            
            last_packet_count = current_packets;
            
            // Update recorder
            recorder.set_shredstream_pps(pps);
            recorder.increment_shredstream_gaps(gaps);
            recorder.increment_shredstream_reorders(reorders);
        }
    }

    fn detect_gaps(sequence: &[u64]) -> u64 {
        let mut gaps = 0;
        for i in 1..sequence.len() {
            if sequence[i] > sequence[i - 1] + 1 {
                gaps += 1;
            }
        }
        gaps
    }

    fn detect_reorders(sequence: &[u64]) -> u64 {
        let mut reorders = 0;
        for i in 1..sequence.len() {
            if sequence[i] < sequence[i - 1] {
                reorders += 1;
            }
        }
        reorders
    }

    fn process_tip(&self, tip: JitoTip) {
        // Add to history
        let mut history = self.tip_history.write();
        history.push_back(tip.clone());
        
        // Keep only last 10000 tips
        if history.len() > 10000 {
            history.pop_front();
        }
        
        // Update metrics
        self.metrics_recorder.record_jito_tip(tip.tip_lamports);
        self.metrics_recorder.increment_jito_tips_total();
    }

    fn process_bundle_status(&self, status: JitoBundleStatus) {
        // Store in history
        self.bundle_history.insert(status.bundle_id.clone(), status.clone());
        
        // Update metrics based on status
        match status.status {
            BundleStatus::Accepted => {
                self.metrics_recorder.increment_bundles_accepted();
                if let Some(sim_time) = status.simulation_time_ms {
                    self.metrics_recorder.record_bundle_simulation_time(sim_time);
                }
            }
            BundleStatus::Rejected => {
                self.metrics_recorder.increment_bundles_rejected();
                if let Some(reason) = &status.rejection_reason {
                    self.metrics_recorder.record_bundle_rejection(reason);
                }
            }
            BundleStatus::Landed => {
                self.metrics_recorder.increment_bundles_landed();
            }
            _ => {}
        }
    }

    fn calculate_tip_metrics(&self) {
        let history = self.tip_history.read();
        
        if history.is_empty() {
            return;
        }
        
        let mut tip_amounts: Vec<u64> = history.iter()
            .map(|t| t.tip_lamports)
            .collect();
        
        tip_amounts.sort_unstable();
        
        let p25 = tip_amounts[tip_amounts.len() / 4];
        let p50 = tip_amounts[tip_amounts.len() / 2];
        let p75 = tip_amounts[tip_amounts.len() * 3 / 4];
        let p90 = tip_amounts[tip_amounts.len() * 9 / 10];
        let p95 = tip_amounts[tip_amounts.len() * 95 / 100];
        let p99 = tip_amounts[tip_amounts.len() * 99 / 100];
        
        self.metrics_recorder.set_tip_percentiles(p25, p50, p75, p90, p95, p99);
        
        let avg_tip = tip_amounts.iter().sum::<u64>() / tip_amounts.len() as u64;
        self.metrics_recorder.set_avg_tip(avg_tip);
        
        let max_tip = *tip_amounts.last().unwrap();
        self.metrics_recorder.set_max_tip_24h(max_tip);
    }

    fn calculate_bundle_metrics(&self) {
        let total = self.bundle_history.len() as u64;
        if total == 0 {
            return;
        }
        
        let mut accepted = 0u64;
        let mut rejected = 0u64;
        let mut landed = 0u64;
        let mut rejection_reasons = std::collections::HashMap::new();
        
        for entry in self.bundle_history.iter() {
            match &entry.status {
                BundleStatus::Accepted => accepted += 1,
                BundleStatus::Rejected => {
                    rejected += 1;
                    if let Some(reason) = &entry.rejection_reason {
                        *rejection_reasons.entry(reason.clone()).or_insert(0) += 1;
                    }
                }
                BundleStatus::Landed => landed += 1,
                _ => {}
            }
        }
        
        let acceptance_rate = accepted as f64 / total as f64;
        self.metrics_recorder.set_bundle_acceptance_rate(acceptance_rate);
        
        // Clean old entries
        if self.bundle_history.len() > 10000 {
            let cutoff = Utc::now() - chrono::Duration::hours(24);
            self.bundle_history.retain(|_, v| v.timestamp > cutoff);
        }
    }

    fn update_shredstream_metrics(&self) {
        let metrics = self.shredstream_metrics.read();
        
        if !metrics.latency_samples.is_empty() {
            let mut sorted = metrics.latency_samples.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let p50 = sorted[sorted.len() / 2];
            let p99 = sorted[sorted.len() * 99 / 100];
            
            self.metrics_recorder.set_shredstream_latency_p50(p50);
            self.metrics_recorder.set_shredstream_latency_p99(p99);
        }
    }

    pub async fn get_tip_history(&self) -> Vec<JitoTip> {
        self.tip_history.read().iter().cloned().collect()
    }

    pub async fn get_bundle_stats(&self) -> (u64, u64, f64) {
        let total = self.bundle_history.len() as u64;
        let accepted = self.bundle_history.iter()
            .filter(|e| matches!(e.status, BundleStatus::Accepted))
            .count() as u64;
        let rate = if total > 0 { accepted as f64 / total as f64 } else { 0.0 };
        
        (total, accepted, rate)
    }
}