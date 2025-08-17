use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSummary {
    pub node_id: String,
    pub version: String,
    pub status: NodeStatus,
    pub uptime_seconds: u64,
    pub identity_pubkey: String,
    pub vote_account: Option<String>,
    pub rpc_endpoint: String,
    pub gossip_endpoint: String,
    pub tpu_endpoint: String,
    pub shred_version: u16,
    pub feature_set: u32,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusHealth {
    pub slot: u64,
    pub epoch: u64,
    pub slot_index: u64,
    pub slots_per_epoch: u64,
    pub optimistic_slot: u64,
    pub root_slot: u64,
    pub first_available_block: u64,
    pub last_vote: u64,
    pub validator_activated_stake: u64,
    pub validator_delinquent_stake: u64,
    pub total_active_stake: u64,
    pub consensus_participation: f64,
    pub tower_sync_status: TowerSyncStatus,
    pub skip_rate: f64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TowerSyncStatus {
    Synced,
    Behind,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterPerformance {
    pub tps_current: f64,
    pub tps_1min: f64,
    pub tps_5min: f64,
    pub tps_peak_24h: f64,
    pub block_time_ms: f64,
    pub block_production_rate: f64,
    pub skip_rate: f64,
    pub total_validators: u64,
    pub active_validators: u64,
    pub delinquent_validators: u64,
    pub network_inflation_rate: f64,
    pub fee_burn_percentage: f64,
    pub priority_fee_median: u64,
    pub priority_fee_p75: u64,
    pub priority_fee_p95: u64,
    pub compute_unit_price_micro_lamports: u64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitoStatus {
    pub block_engine_status: BlockEngineStatus,
    pub regions: Vec<JitoRegion>,
    pub tip_feed_active: bool,
    pub bundles_submitted_24h: u64,
    pub bundles_accepted_24h: u64,
    pub bundles_rejected_24h: u64,
    pub acceptance_rate: f64,
    pub avg_tip_lamports: u64,
    pub max_tip_lamports_24h: u64,
    pub auction_tick_rate_ms: u64,
    pub shredstream_status: ShredStreamStatus,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockEngineStatus {
    pub connected: bool,
    pub region: String,
    pub latency_ms: f64,
    pub last_heartbeat: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitoRegion {
    pub name: String,
    pub endpoint: String,
    pub connected: bool,
    pub latency_ms: f64,
    pub packets_per_second: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShredStreamStatus {
    pub active: bool,
    pub packets_per_second: u64,
    pub gaps_detected: u64,
    pub reorders_detected: u64,
    pub latency_p50_ms: f64,
    pub latency_p99_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcMetrics {
    pub endpoints: Vec<RpcEndpointMetrics>,
    pub total_requests_24h: u64,
    pub total_errors_24h: u64,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub method_latencies: HashMap<String, MethodLatency>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcEndpointMetrics {
    pub endpoint: String,
    pub healthy: bool,
    pub requests_per_second: f64,
    pub error_rate: f64,
    pub avg_latency_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodLatency {
    pub method: String,
    pub count: u64,
    pub avg_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingWaterfall {
    pub total_latency_ms: f64,
    pub stages: Vec<TimingStage>,
    pub bottleneck: String,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStage {
    pub name: String,
    pub start_ms: f64,
    pub duration_ms: f64,
    pub percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TipIntelligence {
    pub current_epoch_tips: u64,
    pub tip_percentiles: TipPercentiles,
    pub top_tippers: Vec<TopTipper>,
    pub tip_efficiency_score: f64,
    pub estimated_roi_percentage: f64,
    pub optimal_tip_lamports: u64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TipPercentiles {
    pub p25: u64,
    pub p50: u64,
    pub p75: u64,
    pub p90: u64,
    pub p95: u64,
    pub p99: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopTipper {
    pub address: String,
    pub total_tips: u64,
    pub avg_tip: u64,
    pub success_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleSuccess {
    pub total_bundles: u64,
    pub successful_bundles: u64,
    pub failed_bundles: u64,
    pub success_rate: f64,
    pub rejection_reasons: HashMap<String, u64>,
    pub avg_simulation_time_ms: f64,
    pub avg_confirmation_time_ms: f64,
    pub profit_extracted_lamports: u64,
    pub gas_used_compute_units: u64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuicHealth {
    pub handshake_success_rate: f64,
    pub concurrent_connections: u64,
    pub max_connections: u64,
    pub open_streams: u64,
    pub throttling_events: u64,
    pub error_code_15_count: u64,
    pub avg_handshake_time_ms: f64,
    pub packet_loss_rate: f64,
    pub retransmission_rate: f64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QosPeering {
    pub stake_weighted_qos_enabled: bool,
    pub total_stake: u64,
    pub our_stake: u64,
    pub qos_priority_level: u8,
    pub peer_connections: Vec<PeerConnection>,
    pub pps_limit: u64,
    pub current_pps: u64,
    pub throttled_packets: u64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerConnection {
    pub peer_id: String,
    pub stake: u64,
    pub connection_quality: f64,
    pub latency_ms: f64,
    pub packet_loss: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipMetrics {
    pub gossip_peers: u64,
    pub gossip_messages_per_second: f64,
    pub repair_peers: u64,
    pub repair_requests_per_second: f64,
    pub broadcast_peers: u64,
    pub broadcast_shreds_per_second: f64,
    pub push_messages: u64,
    pub pull_requests: u64,
    pub pull_responses: u64,
    pub prune_messages: u64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionControlOverview {
    pub node_summary: NodeSummary,
    pub consensus_health: ConsensusHealth,
    pub cluster_performance: ClusterPerformance,
    pub jito_status: JitoStatus,
    pub rpc_metrics: RpcMetrics,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreflightChecks {
    pub checks: Vec<PreflightCheck>,
    pub all_passed: bool,
    pub safe_to_proceed: bool,
    pub warnings: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreflightCheck {
    pub name: String,
    pub passed: bool,
    pub message: String,
    pub severity: CheckSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeControlRequest {
    pub action: NodeAction,
    pub force: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeAction {
    Start,
    Stop,
    Restart,
    CatchUp,
    SetLogLevel(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeControlResponse {
    pub success: bool,
    pub action: NodeAction,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}