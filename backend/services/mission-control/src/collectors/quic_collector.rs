use crate::error::{MissionControlError, Result};
use crate::metrics::MetricsRecorder;
use dashmap::DashMap;
use parking_lot::RwLock;
use quinn::{ClientConfig, Endpoint, TransportConfig};
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::interval;
use tracing::{error, info, warn};

const QUIC_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CONCURRENT_CONNECTIONS: usize = 1024;
const MAX_STREAMS_PER_PEER: usize = 128;
const COLLECTION_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Debug, Clone)]
pub struct QuicMetrics {
    pub handshake_success_rate: f64,
    pub concurrent_connections: u64,
    pub open_streams: u64,
    pub throttling_events: u64,
    pub error_code_15_count: u64,
    pub avg_handshake_time_ms: f64,
    pub packet_loss_rate: f64,
    pub retransmission_rate: f64,
    pub pps_current: u64,
    pub pps_limit: u64,
}

#[derive(Clone)]
struct ConnectionMetrics {
    peer_id: String,
    stake: u64,
    handshake_time: Duration,
    streams_opened: u64,
    packets_sent: u64,
    packets_lost: u64,
    retransmissions: u64,
    last_activity: Instant,
}

#[derive(Clone)]
struct StakeWeightedPeer {
    address: SocketAddr,
    stake: u64,
    priority: u8,
    connection_quality: f64,
    latency_ms: f64,
    packet_loss: f64,
}

pub struct QuicCollector {
    endpoint: Arc<Endpoint>,
    metrics_recorder: Arc<MetricsRecorder>,
    connection_metrics: Arc<DashMap<String, ConnectionMetrics>>,
    stake_weighted_peers: Arc<RwLock<Vec<StakeWeightedPeer>>>,
    handshake_times: Arc<RwLock<VecDeque<f64>>>,
    throttling_counter: Arc<RwLock<u64>>,
    error_code_15_counter: Arc<RwLock<u64>>,
    pps_limiter: Arc<RwLock<PpsLimiter>>,
}

#[derive(Clone)]
struct PpsLimiter {
    current_pps: u64,
    max_pps: u64,
    bucket_size: u64,
    tokens_available: u64,
    last_refill: Instant,
}

impl PpsLimiter {
    fn new(max_pps: u64) -> Self {
        Self {
            current_pps: 0,
            max_pps,
            bucket_size: max_pps * 2,
            tokens_available: max_pps,
            last_refill: Instant::now(),
        }
    }

    fn try_acquire(&mut self, packets: u64) -> bool {
        self.refill();
        
        if self.tokens_available >= packets {
            self.tokens_available -= packets;
            self.current_pps += packets;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let elapsed = self.last_refill.elapsed();
        let tokens_to_add = (elapsed.as_secs_f64() * self.max_pps as f64) as u64;
        
        self.tokens_available = (self.tokens_available + tokens_to_add).min(self.bucket_size);
        
        if elapsed >= Duration::from_secs(1) {
            self.current_pps = 0;
            self.last_refill = Instant::now();
        }
    }
}

impl QuicCollector {
    pub async fn new(
        tpu_endpoint: SocketAddr,
        stake: u64,
        metrics_recorder: Arc<MetricsRecorder>,
    ) -> Result<Self> {
        // Configure QUIC client
        let mut transport_config = TransportConfig::default();
        transport_config.max_concurrent_bidi_streams(MAX_STREAMS_PER_PEER as u32);
        transport_config.max_concurrent_uni_streams(MAX_STREAMS_PER_PEER as u32);
        transport_config.max_idle_timeout(Some(Duration::from_secs(30).try_into().unwrap()));
        transport_config.keep_alive_interval(Some(Duration::from_secs(10)));
        
        let mut client_config = ClientConfig::new(Arc::new(
            rustls::ClientConfig::builder()
                .with_safe_defaults()
                .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
                .with_no_client_auth()
        ));
        client_config.transport_config(Arc::new(transport_config));
        
        let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap())
            .map_err(|e| MissionControlError::ConfigError(e.to_string()))?;
        endpoint.set_default_client_config(client_config);
        
        // Calculate PPS limit based on stake
        let max_pps = Self::calculate_pps_limit(stake);
        
        Ok(Self {
            endpoint: Arc::new(endpoint),
            metrics_recorder,
            connection_metrics: Arc::new(DashMap::new()),
            stake_weighted_peers: Arc::new(RwLock::new(Vec::new())),
            handshake_times: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            throttling_counter: Arc::new(RwLock::new(0)),
            error_code_15_counter: Arc::new(RwLock::new(0)),
            pps_limiter: Arc::new(RwLock::new(PpsLimiter::new(max_pps))),
        })
    }

    fn calculate_pps_limit(stake: u64) -> u64 {
        // Stake-weighted QoS calculation
        const BASE_PPS: u64 = 100;
        const STAKE_MULTIPLIER: f64 = 0.0001;
        
        let stake_bonus = (stake as f64 * STAKE_MULTIPLIER) as u64;
        BASE_PPS + stake_bonus.min(10000) // Cap at 10k PPS
    }

    pub async fn start_collection(&self) {
        let mut ticker = interval(COLLECTION_INTERVAL);
        
        loop {
            ticker.tick().await;
            
            // Monitor active connections
            self.monitor_connections().await;
            
            // Calculate metrics
            self.calculate_metrics().await;
            
            // Simulate QoS peering
            self.monitor_qos_peering().await;
            
            // Clean up old connections
            self.cleanup_stale_connections().await;
        }
    }

    async fn monitor_connections(&self) {
        let connections = self.connection_metrics.len() as u64;
        self.metrics_recorder.set_quic_concurrent_connections(connections);
        
        let mut total_streams = 0u64;
        let mut total_packets_sent = 0u64;
        let mut total_packets_lost = 0u64;
        let mut total_retransmissions = 0u64;
        
        for entry in self.connection_metrics.iter() {
            let metrics = entry.value();
            total_streams += metrics.streams_opened;
            total_packets_sent += metrics.packets_sent;
            total_packets_lost += metrics.packets_lost;
            total_retransmissions += metrics.retransmissions;
        }
        
        self.metrics_recorder.set_quic_open_streams(total_streams);
        
        if total_packets_sent > 0 {
            let packet_loss_rate = total_packets_lost as f64 / total_packets_sent as f64;
            let retransmission_rate = total_retransmissions as f64 / total_packets_sent as f64;
            
            self.metrics_recorder.set_quic_packet_loss_rate(packet_loss_rate);
            self.metrics_recorder.set_quic_retransmission_rate(retransmission_rate);
        }
    }

    async fn calculate_metrics(&self) {
        // Calculate handshake success rate
        let handshake_times = self.handshake_times.read();
        if !handshake_times.is_empty() {
            let avg_handshake = handshake_times.iter().sum::<f64>() / handshake_times.len() as f64;
            self.metrics_recorder.set_quic_avg_handshake_time(avg_handshake);
            
            // Success rate based on handshake completion
            let success_count = handshake_times.iter()
                .filter(|&&t| t < QUIC_HANDSHAKE_TIMEOUT.as_secs_f64() * 1000.0)
                .count();
            let success_rate = success_count as f64 / handshake_times.len() as f64;
            self.metrics_recorder.set_quic_handshake_success_rate(success_rate);
        }
        
        // Record throttling events
        let throttling = *self.throttling_counter.read();
        self.metrics_recorder.set_quic_throttling_events(throttling);
        
        // Record error code 15 (rate limiting)
        let error_15 = *self.error_code_15_counter.read();
        self.metrics_recorder.set_quic_error_code_15(error_15);
        
        // PPS metrics
        let limiter = self.pps_limiter.read();
        self.metrics_recorder.set_quic_current_pps(limiter.current_pps);
        self.metrics_recorder.set_quic_pps_limit(limiter.max_pps);
    }

    async fn monitor_qos_peering(&self) {
        // Simulate stake-weighted QoS peering
        let mut peers = self.stake_weighted_peers.write();
        
        // Update peer metrics
        for peer in peers.iter_mut() {
            // Simulate connection quality based on stake
            peer.connection_quality = 0.5 + (peer.stake as f64 / 1_000_000_000.0).min(0.5);
            
            // Calculate priority level (0-255, higher is better)
            peer.priority = ((peer.stake as f64 / 1_000_000.0).min(255.0)) as u8;
            
            // Simulate latency (lower for higher stake)
            peer.latency_ms = 10.0 + (100.0 / (1.0 + peer.stake as f64 / 1_000_000.0));
            
            // Simulate packet loss (lower for higher stake)
            peer.packet_loss = 0.01 / (1.0 + peer.stake as f64 / 10_000_000.0);
        }
        
        // Sort peers by priority
        peers.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        // Record QoS metrics
        if !peers.is_empty() {
            let our_stake = peers.first().map(|p| p.stake).unwrap_or(0);
            let total_stake: u64 = peers.iter().map(|p| p.stake).sum();
            
            self.metrics_recorder.set_qos_our_stake(our_stake);
            self.metrics_recorder.set_qos_total_stake(total_stake);
            self.metrics_recorder.set_qos_peer_count(peers.len() as u64);
        }
    }

    pub async fn establish_connection(&self, peer: SocketAddr) -> Result<()> {
        let start = Instant::now();
        
        match self.endpoint.connect(peer, "solana-tpu") {
            Ok(connecting) => {
                match tokio::time::timeout(QUIC_HANDSHAKE_TIMEOUT, connecting).await {
                    Ok(Ok(connection)) => {
                        let handshake_time = start.elapsed();
                        let handshake_ms = handshake_time.as_secs_f64() * 1000.0;
                        
                        // Record handshake time
                        let mut times = self.handshake_times.write();
                        times.push_back(handshake_ms);
                        if times.len() > 1000 {
                            times.pop_front();
                        }
                        
                        // Store connection metrics
                        let peer_id = peer.to_string();
                        self.connection_metrics.insert(
                            peer_id.clone(),
                            ConnectionMetrics {
                                peer_id,
                                stake: 0, // Would be fetched from chain
                                handshake_time,
                                streams_opened: 0,
                                packets_sent: 0,
                                packets_lost: 0,
                                retransmissions: 0,
                                last_activity: Instant::now(),
                            },
                        );
                        
                        info!("QUIC connection established to {} in {:.2}ms", peer, handshake_ms);
                        Ok(())
                    }
                    Ok(Err(e)) => {
                        warn!("QUIC connection failed to {}: {}", peer, e);
                        
                        // Check for rate limiting error
                        if e.to_string().contains("15") {
                            *self.error_code_15_counter.write() += 1;
                        }
                        
                        Err(MissionControlError::RpcError(e.to_string()))
                    }
                    Err(_) => {
                        warn!("QUIC handshake timeout to {}", peer);
                        Err(MissionControlError::Timeout)
                    }
                }
            }
            Err(e) => {
                error!("Failed to initiate QUIC connection to {}: {}", peer, e);
                Err(MissionControlError::RpcError(e.to_string()))
            }
        }
    }

    pub async fn send_transaction(&self, peer: SocketAddr, data: &[u8]) -> Result<()> {
        // Check PPS limit
        let mut limiter = self.pps_limiter.write();
        if !limiter.try_acquire(1) {
            *self.throttling_counter.write() += 1;
            return Err(MissionControlError::RpcError("PPS limit exceeded".into()));
        }
        
        // Get or establish connection
        let peer_id = peer.to_string();
        
        if let Some(mut entry) = self.connection_metrics.get_mut(&peer_id) {
            entry.packets_sent += 1;
            entry.last_activity = Instant::now();
        } else {
            self.establish_connection(peer).await?;
        }
        
        // In production, would send actual transaction data
        self.metrics_recorder.increment_quic_packets_sent();
        
        Ok(())
    }

    async fn cleanup_stale_connections(&self) {
        let stale_timeout = Duration::from_secs(60);
        let now = Instant::now();
        
        self.connection_metrics.retain(|_, metrics| {
            now.duration_since(metrics.last_activity) < stale_timeout
        });
    }

    pub async fn get_metrics(&self) -> QuicMetrics {
        let handshake_times = self.handshake_times.read();
        let avg_handshake = if !handshake_times.is_empty() {
            handshake_times.iter().sum::<f64>() / handshake_times.len() as f64
        } else {
            0.0
        };
        
        let success_count = handshake_times.iter()
            .filter(|&&t| t < QUIC_HANDSHAKE_TIMEOUT.as_secs_f64() * 1000.0)
            .count();
        let success_rate = if !handshake_times.is_empty() {
            success_count as f64 / handshake_times.len() as f64
        } else {
            0.0
        };
        
        let mut total_packets_sent = 0u64;
        let mut total_packets_lost = 0u64;
        let mut total_retransmissions = 0u64;
        let mut total_streams = 0u64;
        
        for entry in self.connection_metrics.iter() {
            let metrics = entry.value();
            total_streams += metrics.streams_opened;
            total_packets_sent += metrics.packets_sent;
            total_packets_lost += metrics.packets_lost;
            total_retransmissions += metrics.retransmissions;
        }
        
        let packet_loss_rate = if total_packets_sent > 0 {
            total_packets_lost as f64 / total_packets_sent as f64
        } else {
            0.0
        };
        
        let retransmission_rate = if total_packets_sent > 0 {
            total_retransmissions as f64 / total_packets_sent as f64
        } else {
            0.0
        };
        
        let limiter = self.pps_limiter.read();
        
        QuicMetrics {
            handshake_success_rate: success_rate,
            concurrent_connections: self.connection_metrics.len() as u64,
            open_streams: total_streams,
            throttling_events: *self.throttling_counter.read(),
            error_code_15_count: *self.error_code_15_counter.read(),
            avg_handshake_time_ms: avg_handshake,
            packet_loss_rate,
            retransmission_rate,
            pps_current: limiter.current_pps,
            pps_limit: limiter.max_pps,
        }
    }
}

// Skip TLS verification for TPU connections
struct SkipServerVerification;

impl rustls::client::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::Certificate,
        _intermediates: &[rustls::Certificate],
        _server_name: &rustls::ServerName,
        _scts: &mut dyn Iterator<Item = &[u8]>,
        _ocsp_response: &[u8],
        _now: std::time::SystemTime,
    ) -> std::result::Result<rustls::client::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::ServerCertVerified::assertion())
    }
}