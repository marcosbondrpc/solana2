#![feature(portable_simd)]
#![allow(clippy::too_many_arguments)]

pub mod congestion;
pub mod dual_path;
pub mod telemetry;
pub mod pipeline;
pub mod mev_engine;
pub mod optimization;

use std::sync::Arc;
use parking_lot::RwLock;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static ALLOC: mimalloc_sys::MiMalloc = mimalloc_sys::MiMalloc;

/// Core MEV configuration for ultra-low latency operations
#[derive(Debug, Clone)]
pub struct MevConfig {
    /// Target block time in microseconds
    pub target_block_time_us: u64,
    /// Maximum inflight packets for congestion control
    pub max_inflight_packets: u16,
    /// Optimistic RTT initialization in microseconds
    pub initial_rtt_us: u64,
    /// Jito tip calculation basis points
    pub jito_tip_bps: u16,
    /// Enable hardware timestamping
    pub enable_hw_timestamps: bool,
    /// SIMD optimization level (0-2)
    pub simd_level: u8,
    /// Number of submission threads
    pub submission_threads: usize,
    /// Enable kernel bypass with io_uring
    pub enable_io_uring: bool,
}

impl Default for MevConfig {
    fn default() -> Self {
        Self {
            target_block_time_us: 400_000, // 400ms Solana slot time
            max_inflight_packets: 24,
            initial_rtt_us: 1600,
            jito_tip_bps: 50, // 0.5% tip
            enable_hw_timestamps: true,
            simd_level: 2,
            submission_threads: 4,
            enable_io_uring: true,
        }
    }
}

/// Global MEV context shared across all components
pub struct MevContext {
    pub config: Arc<MevConfig>,
    pub metrics: Arc<telemetry::MetricsCollector>,
    pub submission_stats: Arc<RwLock<SubmissionStats>>,
}

#[derive(Default)]
pub struct SubmissionStats {
    pub total_submitted: u64,
    pub total_landed: u64,
    pub total_profit_lamports: u64,
    pub avg_latency_us: u64,
    pub min_latency_us: u64,
    pub max_latency_us: u64,
}

/// Initialize the MEV backend with optimal settings
pub async fn initialize(config: MevConfig) -> anyhow::Result<Arc<MevContext>> {
    // Set process priority to real-time
    optimization::set_realtime_priority()?;
    
    // Pin threads to specific CPU cores
    optimization::setup_cpu_affinity()?;
    
    // Configure huge pages for memory
    optimization::enable_huge_pages()?;
    
    // Initialize metrics collector
    let metrics = Arc::new(telemetry::MetricsCollector::new());
    
    // Create context
    let context = Arc::new(MevContext {
        config: Arc::new(config),
        metrics,
        submission_stats: Arc::new(RwLock::new(SubmissionStats::default())),
    });
    
    Ok(context)
}