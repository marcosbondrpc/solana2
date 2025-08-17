mod path_selector;

use arc_swap::ArcSwap;
use async_channel::{bounded, Receiver, Sender};
use backoff::ExponentialBackoff;
use bytes::Bytes;
use crossbeam::queue::ArrayQueue;
use dashmap::DashMap;
use futures::{stream::FuturesUnordered, StreamExt};
use jito_block_engine::{
    block_engine_client::BlockEngineClient,
    bundle::Bundle,
    searcher::SearcherClient,
};
use jito_block_engine::bundle::Bundle as ProtoBundle;
use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use prometheus::{
    register_histogram_vec, register_int_counter_vec, register_int_gauge_vec, register_int_gauge,
    HistogramVec, IntCounterVec, IntGaugeVec, IntGauge,
};
use rust_common::{Dedupe, MpmcRing};
use shared_types::{
    Alert, AlertSeverity, AlertType, BundlePriority, JitoBundle, JitoConfig, Result,
    SimulationResult, SystemError,
};
use solana_client::rpc_client::RpcClient;
use solana_sdk::{
    commitment_config::CommitmentConfig,
    compute_budget::ComputeBudgetInstruction,
    instruction::Instruction,
    native_token::LAMPORTS_PER_SOL,
    pubkey::Pubkey,
    signature::{Keypair, Signature, Signer},
    transaction::Transaction,
};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::{mpsc, Notify, RwLock as TokioRwLock, Semaphore},
    task::JoinHandle,
    time::{interval, sleep, timeout},
};
use tokio_util::sync::CancellationToken;
use tonic::{transport::Channel, Request, Status};
use tower::ServiceBuilder;
use tracing::{debug, error, info, trace, warn};
pub mod submitters {
    pub mod jito_quic {
        use anyhow::Result;
        use quinn::Endpoint;
        use std::net::SocketAddr;

        pub struct JitoPool {
            ep: Endpoint,
            addrs: Vec<SocketAddr>,
        }

        impl JitoPool {
            pub async fn new(addrs: Vec<SocketAddr>) -> Result<Self> {
                let ep = Endpoint::client("[::]:0".parse()?)?;
                Ok(Self { ep, addrs })
            }

            pub async fn submit_bundle(&self, bytes: &[u8]) -> Result<()> {
                // Simplified single-address send; extend with health/rotation
                let addr = self.addrs[0];
                let qc = self.ep.connect(addr, "jito").unwrap().await?;
                let mut s = qc.open_uni().await?;
                s.write_all(bytes).await?;
                s.finish()?;
                Ok(())
            }
        }
    }

    pub mod tpu_udp {
        use anyhow::Result;
        use socket2::{Domain, Protocol, Socket, Type};
        use std::net::SocketAddr;

        pub struct TpuSender {
            sock: Socket,
            dsts: Vec<SocketAddr>,
        }

        impl TpuSender {
            pub fn new(dsts: Vec<SocketAddr>) -> Result<Self> {
                let sock = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
                sock.set_nonblocking(true)?;
                Ok(Self { sock, dsts })
            }

            pub fn send(&self, buf: &[u8]) -> Result<()> {
                for d in &self.dsts {
                    let sa = socket2::SockAddr::from(*d);
                    let _ = self.sock.send_to(buf, &sa)?;
                }
                Ok(())
            }
        }
    }
}

// Metrics
static BUNDLE_SUBMISSIONS: Lazy<IntCounterVec> = Lazy::new(|| {
    register_int_counter_vec!(
        "jito_bundle_submissions_total",
        "Total bundle submissions",
        &["status", "priority"]
    )
    .unwrap()
});

static BUNDLE_LATENCY: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "jito_bundle_latency_ms",
        "Bundle submission latency in milliseconds",
        &["stage"]
    )
    .unwrap()
});

static ACTIVE_BUNDLES: Lazy<IntGaugeVec> = Lazy::new(|| {
    register_int_gauge_vec!("jito_active_bundles", "Currently active bundles", &["priority"])
        .unwrap()
});

static BUNDLE_Q_DEPTH: Lazy<IntGauge> = Lazy::new(|| {
    register_int_gauge!("jito_bundle_q_depth", "Current bundle queue depth").unwrap()
});

const MAX_BUNDLE_SIZE: usize = 5;
const BUNDLE_TIMEOUT_MS: u64 = 5000;
const MAX_RETRIES: u32 = 3;
const CONNECTION_POOL_SIZE: usize = 16;
const SIMULATION_BATCH_SIZE: usize = 32;
const QUEUE_HIGH_WATERMARK: usize = 9000;  // 90% of 10000 capacity
const TIP_BETA_MULTIPLIER: f64 = 1.5;      // Soft clamp multiplier
const TIP_PROFIT_CAP_RATIO: f64 = 0.15;    // Cap tip at 15% of profit

/// Ultra-high-performance Jito integration engine
pub struct JitoEngine {
    config: Arc<JitoConfig>,
    block_engine_clients: Arc<Vec<BlockEngineClient<Channel>>>,
    searcher_client: Arc<SearcherClient>,
    rpc_client: Arc<RpcClient>,
    auth_keypair: Arc<Keypair>,
    
    // Bundle management
    bundle_queue: Arc<ArrayQueue<JitoBundle>>,
    bundle_q: Arc<MpmcRing<JitoBundle>>,
    dedupe: Dedupe,
    pending_bundles: Arc<DashMap<String, BundleState>>,
    bundle_results: Arc<DashMap<String, ()>>,
    
    // Performance optimization
    connection_pool: Arc<Vec<Channel>>,
    simulation_cache: Arc<DashMap<u64, SimulationResult>>,
    tip_calculator: Arc<TipCalculator>,
    
    // Channels for high-throughput processing
    bundle_sender: Sender<JitoBundle>,
    bundle_receiver: Receiver<JitoBundle>,
    result_sender: Sender<()>,
    result_receiver: Receiver<()>,
    
    // State management
    is_running: Arc<AtomicBool>,
    submission_count: Arc<AtomicU64>,
    success_count: Arc<AtomicU64>,
    
    // Synchronization
    notify_new_bundle: Arc<Notify>,
    shutdown_notify: Arc<Notify>,
}

#[derive(Debug, Clone)]
struct BundleState {
    bundle: JitoBundle,
    submission_time: Instant,
    retry_count: u32,
    last_error: Option<String>,
}

struct TipCalculator {
    base_tip: u64,
    max_tip: u64,
    recent_tips: Arc<RwLock<VecDeque<u64>>>,
    percentile_95: Arc<AtomicU64>,
}

impl JitoEngine {
    pub async fn new(config: JitoConfig) -> Result<Self> {
        let auth_keypair = Arc::new(
            Keypair::from_bytes(
                &std::fs::read(&config.auth_keypair_path)
                    .map_err(|e| SystemError::JitoBundleError(format!("Failed to read keypair: {}", e)))?,
            )
            .map_err(|e| SystemError::JitoBundleError(format!("Invalid keypair: {}", e)))?,
        );

        // Initialize multiple block engine clients for redundancy
        let mut block_engine_clients = Vec::with_capacity(config.relayer_urls.len());
        for url in &config.relayer_urls {
            let channel = Channel::from_shared(url.clone())
                .map_err(|e| SystemError::JitoBundleError(format!("Invalid URL: {}", e)))?
                .keep_alive_timeout(Duration::from_secs(10))
                .timeout(Duration::from_millis(config.bundle_timeout_ms))
                .connect()
                .await
                .map_err(|e| SystemError::JitoBundleError(format!("Connection failed: {}", e)))?;
            
            block_engine_clients.push(BlockEngineClient::new(channel));
        }

        let rpc_client = Arc::new(RpcClient::new_with_commitment(
            config.block_engine_url.clone(),
            CommitmentConfig::confirmed(),
        ));

        let searcher_client = Arc::new(SearcherClient::new(
            auth_keypair.clone(),
            config.block_engine_url.clone(),
        ));

        // Create high-performance channels
        let (bundle_sender, bundle_receiver) = bounded(10000);
        let (result_sender, result_receiver) = bounded(10000);

        // Initialize connection pool
        let mut connection_pool = Vec::with_capacity(CONNECTION_POOL_SIZE);
        for _ in 0..CONNECTION_POOL_SIZE {
            let channel = Channel::from_shared(config.block_engine_url.clone())
                .unwrap()
                .connect()
                .await
                .map_err(|e| SystemError::JitoBundleError(format!("Pool connection failed: {}", e)))?;
            connection_pool.push(channel);
        }

        let tip_calculator = Arc::new(TipCalculator {
            base_tip: config.min_tip_lamports,
            max_tip: config.max_tip_lamports,
            recent_tips: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            percentile_95: Arc::new(AtomicU64::new(config.min_tip_lamports)),
        });

        Ok(Self {
            config: Arc::new(config),
            block_engine_clients: Arc::new(block_engine_clients),
            searcher_client,
            rpc_client,
            auth_keypair,
            bundle_queue: Arc::new(ArrayQueue::new(10000)),
            bundle_q: Arc::new(MpmcRing::with_capacity_pow2(1 << 16)),
            dedupe: Dedupe::new(Duration::from_secs(8)),
            pending_bundles: Arc::new(DashMap::new()),
            bundle_results: Arc::new(DashMap::new()),
            connection_pool: Arc::new(connection_pool),
            simulation_cache: Arc::new(DashMap::new()),
            tip_calculator,
            bundle_sender,
            bundle_receiver,
            result_sender,
            result_receiver,
            is_running: Arc::new(AtomicBool::new(false)),
            submission_count: Arc::new(AtomicU64::new(0)),
            success_count: Arc::new(AtomicU64::new(0)),
            notify_new_bundle: Arc::new(Notify::new()),
            shutdown_notify: Arc::new(Notify::new()),
        })
    }

    /// Start the Jito engine with multiple worker threads
    pub async fn start(&self) -> Result<()> {
        if self.is_running.swap(true, Ordering::SeqCst) {
            return Ok(());
        }

        info!("Starting Jito Engine with {} relayers", self.config.relayer_urls.len());

        // Spawn bundle processor workers
        let mut workers = Vec::new();
        for i in 0..4 {
            workers.push(self.spawn_bundle_processor(i));
        }

        // Spawn submission workers for each relay
        for (i, _url) in self.config.relayer_urls.iter().enumerate() {
            workers.push(self.spawn_submission_worker(i));
        }

        // Spawn monitoring and cleanup tasks
        workers.push(self.spawn_monitor_task());
        workers.push(self.spawn_cleanup_task());

        // Wait for shutdown signal
        self.shutdown_notify.notified().await;
        
        // Graceful shutdown
        self.is_running.store(false, Ordering::SeqCst);
        for worker in workers {
            let _ = worker.await;
        }

        Ok(())
    }

    /// Submit a bundle with ultra-low latency
    pub async fn submit_bundle(&self, mut bundle: JitoBundle) -> Result<String> {
        let start = Instant::now();
        
        // Quick validation
        if bundle.transactions.is_empty() || bundle.transactions.len() > MAX_BUNDLE_SIZE {
            return Err(SystemError::JitoBundleError(
                "Invalid bundle size".to_string(),
            ));
        }

        // Pre-encode transactions once for reuse across relays
        if bundle.pre_encoded.is_none() {
            let mut encoded = Vec::with_capacity(bundle.transactions.len());
            for tx in &bundle.transactions {
                let tx_bytes = bincode::serialize(tx)
                    .map_err(|e| SystemError::JitoBundleError(format!("Pre-encoding failed: {}", e)))?;
                encoded.push(tx_bytes);
            }
            bundle.pre_encoded = Some(encoded);
        }

        // Dedupe bundles by stable BLAKE3 fingerprint over packet bytes
        if let Some(ref pre) = bundle.pre_encoded {
            let key = rust_common::Dedupe::hash_packets(pre.iter().map(|v| v.as_slice()));
            if !self.dedupe.first_seen(key) {
                BUNDLE_SUBMISSIONS
                    .with_label_values(&["dropped", &format!("{:?}", bundle.priority)])
                    .inc();
                return Err(SystemError::JitoBundleError("Duplicate bundle".to_string()));
            }
        }

        // Calculate optimal tip with new policy
        let tip = self.calculate_optimal_tip(&bundle).await;
        bundle.tip_lamports = tip;

        // Generate bundle ID
        let bundle_id = format!(
            "{}-{}",
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos(),
            self.submission_count.fetch_add(1, Ordering::Relaxed)
        );
        bundle.bundle_id = bundle_id.clone();

        // Backpressure: Drop low-priority bundles when queue is congested
        let queue_len = self.bundle_queue.len();
        if queue_len > QUEUE_HIGH_WATERMARK {
            match bundle.priority {
                BundlePriority::Low => {
                    warn!("Dropping low-priority bundle due to queue congestion: {}/{}", 
                          queue_len, self.bundle_queue.capacity());
                    BUNDLE_SUBMISSIONS
                        .with_label_values(&["dropped", "Low"])
                        .inc();
                    return Err(SystemError::JitoBundleError(
                        "Queue congested, low-priority bundle dropped".to_string()
                    ));
                }
                BundlePriority::Medium if queue_len > QUEUE_HIGH_WATERMARK + 500 => {
                    warn!("Dropping medium-priority bundle due to severe congestion");
                    BUNDLE_SUBMISSIONS
                        .with_label_values(&["dropped", "Medium"])
                        .inc();
                    return Err(SystemError::JitoBundleError(
                        "Queue severely congested, medium-priority bundle dropped".to_string()
                    ));
                }
                _ => {} // High and UltraHigh always go through
            }
        }

        // Fast path for high priority bundles
        if matches!(bundle.priority, BundlePriority::UltraHigh) {
            self.fast_submit_bundle(bundle.clone()).await?;
        } else {
            // Queue for batch processing (lock-free ring)
            if self.bundle_q.push_many(std::iter::once(bundle.clone())) == 0 {
                return Err(SystemError::JitoBundleError("Queue full".to_string()));
            }
            self.notify_new_bundle.notify_one();
        }

        // Update queue depth gauge
        BUNDLE_Q_DEPTH.set(self.bundle_q.len() as i64);

        // Track metrics
        BUNDLE_SUBMISSIONS
            .with_label_values(&["submitted", &format!("{:?}", bundle.priority)])
            .inc();
        BUNDLE_LATENCY
            .with_label_values(&["submission"])
            .observe(start.elapsed().as_millis() as f64);

        Ok(bundle_id)
    }

    /// Ultra-fast bundle submission for high priority with first-success cancellation
    async fn fast_submit_bundle(&self, bundle: JitoBundle) -> Result<()> {
        let start = Instant::now();
        
        // Create cancellation token for first-success pattern
        let cancel_token = CancellationToken::new();
        
        // Parallel submission to multiple relayers
        let mut futures = FuturesUnordered::new();
        
        for (i, client) in self.block_engine_clients.iter().enumerate() {
            let bundle_clone = bundle.clone();
            let client_clone = client.clone();
            let cancel_clone = cancel_token.clone();
            
            futures.push(async move {
                tokio::select! {
                    result = Self::submit_to_relay(client_clone, bundle_clone, i) => {
                        result
                    }
                    _ = cancel_clone.cancelled() => {
                        debug!("Relay {} submission cancelled after first success", i);
                        Err(SystemError::JitoBundleError("Cancelled".to_string()))
                    }
                }
            });
        }

        // Wait for first successful submission
        while let Some(result) = futures.next().await {
            if result.is_ok() {
                // Cancel all other relay attempts
                cancel_token.cancel();
                
                info!(
                    "Bundle {} submitted successfully in {}us (other relays cancelled)",
                    bundle.bundle_id,
                    start.elapsed().as_micros()
                );
                
                // Drain remaining futures to clean up
                while futures.next().await.is_some() {}
                
                return Ok(());
            }
        }

        Err(SystemError::JitoBundleError(
            "All relay submissions failed".to_string(),
        ))
    }

    /// Submit bundle to specific relay
    async fn submit_to_relay(
        client: BlockEngineClient<Channel>,
        bundle: JitoBundle,
        relay_index: usize,
    ) -> Result<()> {
        let proto_bundle = Self::convert_to_proto_bundle(&bundle)?;
        
        let request = Request::new(proto_bundle);
        
        match timeout(Duration::from_millis(BUNDLE_TIMEOUT_MS), client.send_bundle(request)).await {
            Ok(Ok(response)) => {
                debug!("Bundle submitted to relay {}: {:?}", relay_index, response);
                Ok(())
            }
            Ok(Err(e)) => {
                warn!("Relay {} rejected bundle: {}", relay_index, e);
                Err(SystemError::JitoBundleError(format!("Relay error: {}", e)))
            }
            Err(_) => {
                warn!("Relay {} timeout", relay_index);
                Err(SystemError::JitoBundleError("Timeout".to_string()))
            }
        }
    }

    /// Convert to protobuf bundle format (reuses pre-encoded transactions)
    fn convert_to_proto_bundle(bundle: &JitoBundle) -> Result<ProtoBundle> {
        let mut proto_bundle = ProtoBundle::default();
        
        // Use pre-encoded transactions if available, otherwise encode on-the-fly
        if let Some(pre_encoded) = &bundle.pre_encoded {
            proto_bundle.transactions = pre_encoded.clone();
        } else {
            for tx in &bundle.transactions {
                let tx_bytes = bincode::serialize(tx)
                    .map_err(|e| SystemError::JitoBundleError(format!("Serialization failed: {}", e)))?;
                proto_bundle.transactions.push(tx_bytes);
            }
        }
        
        Ok(proto_bundle)
    }

    /// Calculate optimal tip with quantile clamp + profit cap policy
    async fn calculate_optimal_tip(&self, bundle: &JitoBundle) -> u64 {
        let base_tip = self.tip_calculator.percentile_95.load(Ordering::Relaxed);
        
        // Priority multiplier
        let priority_multiplier = match bundle.priority {
            BundlePriority::UltraHigh => 3.0,
            BundlePriority::High => 2.0,
            BundlePriority::Medium => 1.5,
            BundlePriority::Low => 1.0,
        };
        
        // Calculate base tip with priority
        let priority_tip = (base_tip as f64 * priority_multiplier) as u64;
        
        // Apply soft clamp with beta multiplier (1.5x the 95th percentile)
        let soft_clamped_tip = if priority_tip > base_tip {
            let excess = priority_tip - base_tip;
            let clamped_excess = ((excess as f64) * TIP_BETA_MULTIPLIER) as u64;
            base_tip + clamped_excess.min(base_tip * 2) // Never exceed 3x base
        } else {
            priority_tip
        };
        
        // Profit-based adjustment with cap
        let final_tip = if let Some(sim) = &bundle.simulation_result {
            if sim.profit > 0 {
                // Cap tip adjustment at 15% of profit
                let profit_tip = (sim.profit as f64 * TIP_PROFIT_CAP_RATIO) as u64;
                let total_tip = soft_clamped_tip + profit_tip;
                
                // Also ensure tip doesn't exceed profit (negative EV protection)
                total_tip.min(sim.profit as u64)
            } else {
                soft_clamped_tip
            }
        } else {
            soft_clamped_tip
        };
        
        // Apply global min/max bounds
        final_tip
            .max(self.config.min_tip_lamports)
            .min(self.config.max_tip_lamports)
    }

    /// Simulate bundle for profit calculation
    pub async fn simulate_bundle(&self, bundle: &JitoBundle) -> Result<SimulationResult> {
        let start = Instant::now();
        
        // Check cache first
        let cache_key = Self::calculate_bundle_hash(bundle);
        if let Some(cached) = self.simulation_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Perform simulation
        let mut total_profit = 0i64;
        let mut total_gas = 0u64;
        let mut total_compute = 0u64;
        let mut logs = Vec::new();

        for tx in &bundle.transactions {
            let sim_result = self.rpc_client
                .simulate_transaction(tx)
                .map_err(|e| SystemError::SimulationError(format!("RPC error: {}", e)))?;

            if let Some(error) = sim_result.value.err {
                return Err(SystemError::SimulationError(format!(
                    "Transaction failed: {:?}",
                    error
                )));
            }

            // Extract metrics from logs
            let mut tx_used_cu: Option<u64> = None;
            if let Some(tx_logs) = sim_result.value.logs.clone() {
                logs.extend(tx_logs.clone());
                for log in tx_logs {
                    if log.contains("compute_units") {
                        if let Some(units) = Self::extract_compute_units(&log) {
                            total_compute += units;
                            tx_used_cu = Some(units);
                        }
                    }
                }
            }

            // Calculate gas cost (prefer actual used CU when available)
            total_gas += Self::calculate_gas_cost(tx, tx_used_cu);
        }

        // Calculate net profit
        let est_out = bundle
            .simulation_result
            .as_ref()
            .map(|s| s.profit.max(0) as u64)
            .unwrap_or(0);
        total_profit = (est_out as i64) - (total_gas as i64);

        let result = SimulationResult {
            success: true,
            profit: total_profit,
            gas_used: total_gas,
            compute_units: total_compute,
            logs,
        };

        // Cache result
        self.simulation_cache.insert(cache_key, result.clone());

        debug!(
            "Bundle simulation completed in {}us, profit: {} lamports",
            start.elapsed().as_micros(),
            total_profit
        );

        Ok(result)
    }

    /// Calculate bundle hash for caching
    fn calculate_bundle_hash(bundle: &JitoBundle) -> u64 {
        // Use BLAKE3 fingerprint (first 8 bytes) over pre-encoded packets (or serialize on the fly)
        let mut packets: Vec<Vec<u8>> = Vec::new();
        if let Some(pre) = &bundle.pre_encoded {
            packets = pre.clone();
        } else {
            for tx in &bundle.transactions {
                if let Ok(bytes) = bincode::serialize(tx) {
                    packets.push(bytes);
                }
            }
        }
        let key = rust_common::Dedupe::hash_packets(packets.iter().map(|v| v.as_slice()));
        // Truncate to u64 for cache key
        u64::from_le_bytes([key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7]])
    }

    /// Extract compute units from log
    fn extract_compute_units(log: &str) -> Option<u64> {
        log.split_whitespace()
            .find_map(|s| s.parse::<u64>().ok())
    }

    /// Calculate gas cost for transaction
    fn calculate_gas_cost(tx: &Transaction, used_cu_override: Option<u64>) -> u64 {
        use solana_sdk::compute_budget::{self, ComputeBudgetInstruction};

        // Base fee (legacy) plus dynamic priority fee = cu_price_micro * used_cu
        // Note: We prefer actual consumed CU parsed from logs elsewhere; if not available,
        // we fall back to SetComputeUnitLimit value as an upper bound.
        let base_fee = 5000u64; // 0.000005 SOL
        let mut cu_price_micro: u64 = 0;
        let mut cu_limit: u64 = 0;

        let account_keys: &[Pubkey] = &tx.message.account_keys;
        for ix in &tx.message.instructions {
            if *ix.program_id(account_keys) == compute_budget::id() {
                // ComputeBudgetInstruction is bincode-deserializable from ix.data
                if let Ok(cb_ix) = bincode::deserialize::<ComputeBudgetInstruction>(&ix.data) {
                    match cb_ix {
                        ComputeBudgetInstruction::SetComputeUnitPrice(micro_lamports) => {
                            cu_price_micro = micro_lamports as u64;
                        }
                        ComputeBudgetInstruction::SetComputeUnitLimit(units) => {
                            cu_limit = units as u64;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Prefer actual used CU from simulation logs when provided; otherwise fallback to declared limit.
        let used_cu = used_cu_override.unwrap_or(cu_limit);
        let priority_fee = cu_price_micro.saturating_mul(used_cu);

        base_fee.saturating_add(priority_fee)
    }

    /// Spawn bundle processor worker
    fn spawn_bundle_processor(&self, worker_id: usize) -> JoinHandle<()> {
        let bundle_queue = self.bundle_queue.clone();
        let bundle_sender = self.bundle_sender.clone();
        let notify = self.notify_new_bundle.clone();
        let is_running = self.is_running.clone();

        tokio::spawn(async move {
            info!("Bundle processor {} started", worker_id);
            
            while is_running.load(Ordering::Relaxed) {
                // Wait for new bundles
                tokio::select! {
                    _ = notify.notified() => {},
                    _ = tokio::time::sleep(Duration::from_millis(10)) => {},
                }

                // Process bundles in batch
                let mut batch = Vec::with_capacity(SIMULATION_BATCH_SIZE);
                while let Some(bundle) = bundle_queue.pop() {
                    batch.push(bundle);
                    if batch.len() >= SIMULATION_BATCH_SIZE {
                        break;
                    }
                }

                // Send to submission pipeline
                for bundle in batch {
                    if let Err(e) = bundle_sender.send(bundle).await {
                        error!("Failed to send bundle: {}", e);
                    }
                }
            }
            
            info!("Bundle processor {} stopped", worker_id);
        })
    }

    /// Spawn submission worker for specific relay
    fn spawn_submission_worker(&self, relay_index: usize) -> JoinHandle<()> {
        let bundle_receiver = self.bundle_receiver.clone();
        let client = self.block_engine_clients[relay_index].clone();
        let is_running = self.is_running.clone();
        let success_count = self.success_count.clone();

        tokio::spawn(async move {
            info!("Submission worker {} started", relay_index);
            
            while is_running.load(Ordering::Relaxed) {
                match bundle_receiver.recv().await {
                    Ok(bundle) => {
                        match Self::submit_to_relay(client.clone(), bundle.clone(), relay_index).await {
                            Ok(_) => {
                                success_count.fetch_add(1, Ordering::Relaxed);
                                BUNDLE_SUBMISSIONS
                                    .with_label_values(&["success", &format!("{:?}", bundle.priority)])
                                    .inc();
                            }
                            Err(e) => {
                                error!("Relay {} submission failed: {}", relay_index, e);
                                BUNDLE_SUBMISSIONS
                                    .with_label_values(&["failed", &format!("{:?}", bundle.priority)])
                                    .inc();
                            }
                        }
                    }
                    Err(_) => {
                        // Channel closed
                        break;
                    }
                }
            }
            
            info!("Submission worker {} stopped", relay_index);
        })
    }

    /// Spawn monitoring task
    fn spawn_monitor_task(&self) -> JoinHandle<()> {
        let is_running = self.is_running.clone();
        let submission_count = self.submission_count.clone();
        let success_count = self.success_count.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));
            
            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;
                
                let submissions = submission_count.load(Ordering::Relaxed);
                let successes = success_count.load(Ordering::Relaxed);
                let success_rate = if submissions > 0 {
                    (successes as f64 / submissions as f64) * 100.0
                } else {
                    0.0
                };
                
                info!(
                    "Jito Engine Stats - Submissions: {}, Success Rate: {:.2}%",
                    submissions, success_rate
                );
            }
        })
    }

    /// Spawn cleanup task
    fn spawn_cleanup_task(&self) -> JoinHandle<()> {
        let pending_bundles = self.pending_bundles.clone();
        let bundle_results = self.bundle_results.clone();
        let simulation_cache = self.simulation_cache.clone();
        let is_running = self.is_running.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));
            
            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;
                
                // Clean old pending bundles
                let now = Instant::now();
                pending_bundles.retain(|_, state| {
                    now.duration_since(state.submission_time) < Duration::from_secs(300)
                });
                
                // Clean old results
                if bundle_results.len() > 10000 {
                    bundle_results.clear();
                }
                
                // Clean simulation cache
                if simulation_cache.len() > 1000 {
                    simulation_cache.clear();
                }
                
                debug!("Cleanup completed - pending: {}, results: {}, cache: {}",
                    pending_bundles.len(),
                    bundle_results.len(),
                    simulation_cache.len()
                );
            }
        })
    }

    /// Shutdown the engine gracefully
    pub async fn shutdown(&self) {
        info!("Shutting down Jito Engine");
        self.is_running.store(false, Ordering::SeqCst);
        self.shutdown_notify.notify_waiters();
    }
}