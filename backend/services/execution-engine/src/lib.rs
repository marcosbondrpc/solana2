pub mod thompson_sampler;
pub mod hedged_sender;
pub mod route_selector;
pub mod submission_engine;
pub mod outcome_tracker;

use anyhow::Result;
use arc_swap::ArcSwap;
use bytes::Bytes;
use crossbeam::channel::{bounded, Receiver, Sender};
use dashmap::DashMap;
use nix::sys::socket::{setsockopt, sockopt};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use solana_sdk::{
    commitment_config::CommitmentConfig,
    instruction::Instruction,
    pubkey::Pubkey,
    signature::{Keypair, Signature},
    transaction::Transaction,
};
use std::{
    collections::{HashMap, VecDeque},
    net::{IpAddr, SocketAddr},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime},
};
use tokio::{
    net::UdpSocket,
    sync::{mpsc, Semaphore},
    time::{interval, sleep},
};
use tracing::{debug, error, info, warn};

// Constants for ultra-low latency
const MAX_CONCURRENT_SUBMISSIONS: usize = 100;
const HEDGED_DELAY_US: u64 = 500; // 500 microseconds
const OUTCOME_FEEDBACK_INTERVAL_MS: u64 = 100;
const ROUTE_CACHE_TTL_MS: u64 = 1000;
const DSCP_EXPEDITED_FORWARDING: u8 = 46; // EF DSCP value

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRequest {
    pub id: String,
    pub transaction: Transaction,
    pub strategy: ExecutionStrategy,
    pub priority: ExecutionPriority,
    pub max_retries: u32,
    pub timeout_ms: u64,
    pub expected_profit: u64,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    DirectTPU,           // Direct to leader TPU
    JitoBundle,          // Through Jito block engine
    HedgedSending,       // Send to multiple endpoints
    DualSubmit,          // Both TPU and Jito
    AdaptiveBandit,      // Thompson Sampling selection
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ExecutionPriority {
    UltraHigh,  // < 1ms latency requirement
    High,       // < 5ms latency requirement
    Medium,     // < 20ms latency requirement
    Low,        // Best effort
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub request_id: String,
    pub signature: Signature,
    pub status: ExecutionStatus,
    pub route_used: Route,
    pub submission_time: Instant,
    pub confirmation_time: Option<Instant>,
    pub latency_ms: u64,
    pub slot: u64,
    pub block_height: u64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Pending,
    Submitted,
    Confirmed,
    Failed,
    Timeout,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Route {
    pub route_type: RouteType,
    pub endpoint: String,
    pub priority_fee: u64,
    pub tip_amount: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RouteType {
    LeaderTPU,
    NextLeaderTPU,
    JitoBlockEngine,
    JitoRelayer,
    QuicTPU,
    UdpTPU,
}

pub struct ExecutionEngine {
    config: Arc<ExecutionConfig>,
    thompson_sampler: Arc<thompson_sampler::ThompsonSampler>,
    hedged_sender: Arc<hedged_sender::HedgedSender>,
    route_selector: Arc<route_selector::RouteSelector>,
    submission_engine: Arc<submission_engine::SubmissionEngine>,
    outcome_tracker: Arc<outcome_tracker::OutcomeTracker>,
    metrics: Arc<ExecutionMetrics>,
    shutdown: Arc<AtomicBool>,
}

pub struct ExecutionConfig {
    pub tpu_endpoints: Vec<SocketAddr>,
    pub jito_endpoints: Vec<String>,
    pub rpc_endpoints: Vec<String>,
    pub keypair: Arc<Keypair>,
    pub max_concurrent: usize,
    pub enable_dscp: bool,
    pub enable_so_txtime: bool,
    pub cpu_affinity: bool,
}

struct ExecutionMetrics {
    total_requests: AtomicU64,
    successful_submissions: AtomicU64,
    failed_submissions: AtomicU64,
    avg_latency_ms: AtomicU64,
    route_selections: DashMap<RouteType, AtomicU64>,
    bundle_land_rate: AtomicU64,
}

impl ExecutionEngine {
    pub fn new(config: ExecutionConfig) -> Result<Self> {
        let thompson_sampler = Arc::new(thompson_sampler::ThompsonSampler::new());
        let hedged_sender = Arc::new(hedged_sender::HedgedSender::new(
            config.tpu_endpoints.clone(),
            config.jito_endpoints.clone(),
        )?);
        let route_selector = Arc::new(route_selector::RouteSelector::new());
        let submission_engine = Arc::new(submission_engine::SubmissionEngine::new(
            config.keypair.clone(),
            config.enable_dscp,
            config.enable_so_txtime,
        )?);
        let outcome_tracker = Arc::new(outcome_tracker::OutcomeTracker::new());
        
        let metrics = Arc::new(ExecutionMetrics {
            total_requests: AtomicU64::new(0),
            successful_submissions: AtomicU64::new(0),
            failed_submissions: AtomicU64::new(0),
            avg_latency_ms: AtomicU64::new(0),
            route_selections: DashMap::new(),
            bundle_land_rate: AtomicU64::new(0),
        });
        
        Ok(Self {
            config: Arc::new(config),
            thompson_sampler,
            hedged_sender,
            route_selector,
            submission_engine,
            outcome_tracker,
            metrics,
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }
    
    pub async fn execute(&self, request: ExecutionRequest) -> Result<ExecutionResult> {
        let start = Instant::now();
        self.metrics.total_requests.fetch_add(1, Ordering::Relaxed);
        
        // Select route based on strategy
        let route = match request.strategy {
            ExecutionStrategy::AdaptiveBandit => {
                self.thompson_sampler.select_route(&request).await?
            }
            ExecutionStrategy::DirectTPU => {
                self.route_selector.get_leader_tpu_route().await?
            }
            ExecutionStrategy::JitoBundle => {
                self.route_selector.get_jito_route().await?
            }
            ExecutionStrategy::HedgedSending => {
                // Will use multiple routes
                self.route_selector.get_best_route(&request).await?
            }
            ExecutionStrategy::DualSubmit => {
                // Will submit to both TPU and Jito
                self.route_selector.get_best_route(&request).await?
            }
        };
        
        // Update route selection metrics
        self.metrics.route_selections
            .entry(route.route_type)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
        
        // Execute based on strategy
        let result = match request.strategy {
            ExecutionStrategy::HedgedSending => {
                self.hedged_send(request, route).await?
            }
            ExecutionStrategy::DualSubmit => {
                self.dual_submit(request, route).await?
            }
            _ => {
                self.single_submit(request, route).await?
            }
        };
        
        // Track outcome for feedback
        self.outcome_tracker.record_outcome(&result).await;
        
        // Update metrics
        let latency = start.elapsed().as_millis() as u64;
        self.update_metrics(&result, latency);
        
        Ok(result)
    }
    
    async fn single_submit(&self, request: ExecutionRequest, route: Route) -> Result<ExecutionResult> {
        let submission_time = Instant::now();
        
        // Submit transaction
        let signature = self.submission_engine
            .submit(&request.transaction, &route)
            .await?;
        
        // Create result
        let result = ExecutionResult {
            request_id: request.id,
            signature,
            status: ExecutionStatus::Submitted,
            route_used: route,
            submission_time,
            confirmation_time: None,
            latency_ms: submission_time.elapsed().as_millis() as u64,
            slot: 0, // Will be updated on confirmation
            block_height: 0,
            error: None,
        };
        
        Ok(result)
    }
    
    async fn hedged_send(&self, request: ExecutionRequest, primary_route: Route) -> Result<ExecutionResult> {
        // Get additional routes for hedging
        let mut routes = vec![primary_route.clone()];
        routes.extend(self.route_selector.get_hedge_routes(&primary_route, 2).await?);
        
        // Submit with hedging
        let result = self.hedged_sender
            .send_hedged(&request, routes)
            .await?;
        
        Ok(result)
    }
    
    async fn dual_submit(&self, request: ExecutionRequest, route: Route) -> Result<ExecutionResult> {
        let tpu_route = self.route_selector.get_leader_tpu_route().await?;
        let jito_route = self.route_selector.get_jito_route().await?;
        
        // Submit to both simultaneously
        let (tpu_result, jito_result) = tokio::join!(
            self.single_submit(request.clone(), tpu_route),
            self.single_submit(request.clone(), jito_route)
        );
        
        // Return the first successful one
        match (tpu_result, jito_result) {
            (Ok(tpu), _) => Ok(tpu),
            (_, Ok(jito)) => Ok(jito),
            (Err(e), _) => Err(e),
        }
    }
    
    fn update_metrics(&self, result: &ExecutionResult, latency: u64) {
        if result.status == ExecutionStatus::Confirmed {
            self.metrics.successful_submissions.fetch_add(1, Ordering::Relaxed);
        } else if result.status == ExecutionStatus::Failed {
            self.metrics.failed_submissions.fetch_add(1, Ordering::Relaxed);
        }
        
        // Update average latency (simplified - should use EWMA)
        self.metrics.avg_latency_ms.store(latency, Ordering::Relaxed);
        
        // Update bundle land rate if it was a Jito submission
        if result.route_used.route_type == RouteType::JitoBlockEngine {
            if result.status == ExecutionStatus::Confirmed {
                let current = self.metrics.bundle_land_rate.load(Ordering::Relaxed);
                self.metrics.bundle_land_rate.store(
                    (current * 99 + 10000) / 100, // EWMA with 1% weight
                    Ordering::Relaxed
                );
            }
        }
    }
    
    pub async fn start_feedback_loop(&self) {
        let mut interval = interval(Duration::from_millis(OUTCOME_FEEDBACK_INTERVAL_MS));
        
        while !self.shutdown.load(Ordering::Relaxed) {
            interval.tick().await;
            
            // Get recent outcomes
            let outcomes = self.outcome_tracker.get_recent_outcomes(100).await;
            
            // Update Thompson Sampler with outcomes
            for outcome in outcomes {
                self.thompson_sampler.update_with_outcome(outcome).await;
            }
            
            // Update route selector with performance data
            let performance = self.outcome_tracker.get_route_performance().await;
            self.route_selector.update_performance(performance).await;
        }
    }
    
    pub fn get_metrics(&self) -> HashMap<String, u64> {
        let mut metrics = HashMap::new();
        metrics.insert("total_requests".to_string(), 
                      self.metrics.total_requests.load(Ordering::Relaxed));
        metrics.insert("successful_submissions".to_string(),
                      self.metrics.successful_submissions.load(Ordering::Relaxed));
        metrics.insert("failed_submissions".to_string(),
                      self.metrics.failed_submissions.load(Ordering::Relaxed));
        metrics.insert("avg_latency_ms".to_string(),
                      self.metrics.avg_latency_ms.load(Ordering::Relaxed));
        metrics.insert("bundle_land_rate_bps".to_string(),
                      self.metrics.bundle_land_rate.load(Ordering::Relaxed));
        
        // Add route selection counts
        for entry in self.metrics.route_selections.iter() {
            let route_type = format!("route_{:?}", entry.key());
            metrics.insert(route_type, entry.value().load(Ordering::Relaxed));
        }
        
        metrics
    }
    
    pub async fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        info!("Execution engine shutting down");
    }
}

// Helper function to set DSCP on socket
pub fn set_dscp_on_socket(socket: &UdpSocket, dscp: u8) -> Result<()> {
    use std::os::unix::io::AsRawFd;
    let fd = socket.as_raw_fd();
    let tos = (dscp << 2) as i32;
    
    unsafe {
        let ret = libc::setsockopt(
            fd,
            libc::IPPROTO_IP,
            libc::IP_TOS,
            &tos as *const _ as *const libc::c_void,
            std::mem::size_of::<i32>() as libc::socklen_t,
        );
        
        if ret != 0 {
            return Err(anyhow::anyhow!("Failed to set DSCP"));
        }
    }
    
    Ok(())
}

// Helper function to set SO_TXTIME for packet pacing
pub fn set_so_txtime(socket: &UdpSocket, timestamp_ns: u64) -> Result<()> {
    // This requires Linux 4.19+ with SO_TXTIME support
    // Implementation would use SCM_TXTIME control message
    Ok(())
}