use bytes::Bytes;
use ethnum::U256;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use solana_sdk::{
    pubkey::Pubkey,
    signature::Signature,
    transaction::Transaction,
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use thiserror::Error;

/// Ultra-high precision price representation using U256 for MEV calculations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Price {
    /// Numerator for price calculation (amount out)
    pub numerator: U256,
    /// Denominator for price calculation (amount in)
    pub denominator: U256,
    /// Decimal places for display
    pub decimals: u8,
}

impl Price {
    #[inline(always)]
    pub fn from_amounts(amount_out: u64, amount_in: u64, decimals: u8) -> Self {
        Self {
            numerator: U256::from(amount_out),
            denominator: U256::from(amount_in),
            decimals,
        }
    }

    #[inline(always)]
    pub fn to_decimal(&self) -> Decimal {
        let num = Decimal::from_str_exact(&self.numerator.to_string()).unwrap_or_default();
        let den = Decimal::from_str_exact(&self.denominator.to_string()).unwrap_or(Decimal::ONE);
        num / den
    }

    #[inline(always)]
    pub fn calculate_output(&self, input_amount: u64) -> u64 {
        let input = U256::from(input_amount);
        let output = (input * self.numerator) / self.denominator;
        output.as_u64()
    }
}

/// DEX types we support for arbitrage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DexType {
    Raydium,
    RaydiumCLMM,
    Orca,
    OrcaWhirlpool,
    Phoenix,
    Meteora,
    MeteoraLBP,
    OpenBook,
    OpenBookV2,
    Lifinity,
    Lifinity2,
    Aldrin,
    Saber,
    Invariant,
}

/// Pool state for ultra-fast access
#[derive(Debug, Clone)]
pub struct PoolState {
    pub dex: DexType,
    pub pool_address: Pubkey,
    pub token_a: Pubkey,
    pub token_b: Pubkey,
    pub reserve_a: u64,
    pub reserve_b: u64,
    pub fee_bps: u16,
    pub last_update: Instant,
    pub price_a_to_b: Price,
    pub price_b_to_a: Price,
    /// Pre-calculated sqrt price for concentrated liquidity pools
    pub sqrt_price: Option<U256>,
    /// Tick information for concentrated liquidity
    pub current_tick: Option<i32>,
    /// Liquidity for concentrated liquidity pools
    pub liquidity: Option<u128>,
}

/// Arbitrage opportunity with profit calculation
#[derive(Debug, Clone)]
pub struct ArbitrageOpportunity {
    pub id: u64,
    pub path: Vec<ArbitrageStep>,
    pub input_token: Pubkey,
    pub output_token: Pubkey,
    pub input_amount: u64,
    pub estimated_output: u64,
    pub profit_amount: u64,
    pub profit_bps: u16,
    pub gas_cost: u64,
    pub net_profit: i64,
    pub discovered_at: Instant,
    pub expires_at: Instant,
}

#[derive(Debug, Clone)]
pub struct ArbitrageStep {
    pub dex: DexType,
    pub pool: Pubkey,
    pub input_token: Pubkey,
    pub output_token: Pubkey,
    pub expected_amount_in: u64,
    pub expected_amount_out: u64,
    pub slippage_bps: u16,
}

/// Transaction bundle for Jito submission
#[derive(Debug, Clone)]
pub struct JitoBundle {
    pub transactions: Vec<Transaction>,
    pub tip_lamports: u64,
    pub bundle_id: String,
    pub priority: BundlePriority,
    pub simulation_result: Option<SimulationResult>,
    /// Pre-encoded transaction bytes for reuse across relay submissions
    pub pre_encoded: Option<Vec<Vec<u8>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundlePriority {
    UltraHigh,  // For highest value MEV
    High,       // Standard MEV extraction
    Medium,     // Lower value opportunities
    Low,        // Background operations
}

#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub success: bool,
    pub profit: i64,
    pub gas_used: u64,
    pub compute_units: u64,
    pub logs: Vec<String>,
}

/// QUIC connection state for ultra-low latency
#[derive(Debug, Clone)]
pub struct QuicConnectionState {
    pub endpoint: String,
    pub connected: bool,
    pub latency_us: u64,
    pub packet_loss_rate: f32,
    pub congestion_window: u32,
    pub rtt_smoothed: Duration,
    pub last_activity: Instant,
}

/// System performance metrics
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub tx_processing_us: u64,
    pub quic_rtt_us: u64,
    pub dex_parsing_us: u64,
    pub arb_detection_us: u64,
    pub bundle_submission_ms: u64,
    pub cpu_usage_percent: f32,
    pub memory_usage_mb: u64,
    pub network_throughput_mbps: f32,
}

/// Mempool transaction for analysis
#[derive(Debug, Clone)]
pub struct MempoolTransaction {
    pub signature: Signature,
    pub transaction: Transaction,
    pub received_at: Instant,
    pub priority_fee: u64,
    pub compute_units: u64,
    pub accounts: Vec<Pubkey>,
    pub program_ids: Vec<Pubkey>,
}

/// Alert types for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    HighLatency { threshold_us: u64, actual_us: u64 },
    ArbitrageOpportunity { profit_lamports: u64 },
    SystemResourceHigh { resource: String, usage: f32 },
    TransactionFailed { signature: String, error: String },
    NetworkDegradation { endpoint: String, packet_loss: f32 },
    BundleRejected { bundle_id: String, reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Custom errors for the system
#[derive(Error, Debug)]
pub enum SystemError {
    #[error("QUIC connection failed: {0}")]
    QuicConnectionError(String),
    
    #[error("DEX parsing error: {0}")]
    DexParsingError(String),
    
    #[error("Arbitrage calculation error: {0}")]
    ArbitrageError(String),
    
    #[error("Jito bundle error: {0}")]
    JitoBundleError(String),
    
    #[error("Transaction simulation failed: {0}")]
    SimulationError(String),
    
    #[error("Insufficient profit: expected {expected}, got {actual}")]
    InsufficientProfit { expected: u64, actual: u64 },
    
    #[error("System resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),
}

/// Thread-safe pool cache for zero-copy access
pub type PoolCache = Arc<dashmap::DashMap<Pubkey, Arc<RwLock<PoolState>>>>;

/// Configuration for the entire system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub jito_config: JitoConfig,
    pub quic_config: QuicConfig,
    pub arbitrage_config: ArbitrageConfig,
    pub performance_config: PerformanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitoConfig {
    pub block_engine_url: String,
    pub relayer_urls: Vec<String>,
    pub auth_keypair_path: String,
    pub tip_account: Pubkey,
    pub min_tip_lamports: u64,
    pub max_tip_lamports: u64,
    pub bundle_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuicConfig {
    pub leader_endpoints: Vec<String>,
    pub max_concurrent_streams: u32,
    pub initial_rtt_ms: u64,
    pub max_idle_timeout_ms: u64,
    pub keep_alive_interval_ms: u64,
    pub congestion_controller: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageConfig {
    pub min_profit_lamports: u64,
    pub max_hops: usize,
    pub slippage_tolerance_bps: u16,
    pub simulation_compute_limit: u64,
    pub max_input_sol: f64,
    pub priority_fee_lamports: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub cpu_affinity_mask: u64,
    pub memory_pool_size_mb: u64,
    pub io_uring_enabled: bool,
    pub thread_priority: i32,
    pub batch_size: usize,
}

pub type Result<T> = std::result::Result<T, SystemError>;