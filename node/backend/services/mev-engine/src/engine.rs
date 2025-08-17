use anyhow::Result;
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, broadcast, Semaphore};
use tokio::time::interval;
use tracing::{info, warn, debug, error};

use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::{
    pubkey::Pubkey,
    signature::Signature,
    transaction::Transaction,
};

use crate::config::Config;
use crate::mempool::MempoolMonitor;
use crate::bundle::{Bundle, BundleBuilder};
use crate::submission::SubmissionEngine;
use crate::metrics::Metrics;

const MAX_CONCURRENT_BUNDLES: usize = 100;
const BUNDLE_TIMEOUT_MS: u64 = 50;

pub struct MevEngine {
    config: Arc<Config>,
    rpc_client: Arc<RpcClient>,
    mempool_monitor: Arc<MempoolMonitor>,
    submission_engine: Arc<SubmissionEngine>,
    bundle_builder: Arc<BundleBuilder>,
    metrics: Arc<Metrics>,
    
    // Bundle tracking
    active_bundles: Arc<DashMap<Signature, Bundle>>,
    bundle_semaphore: Arc<Semaphore>,
    
    // Control channels
    shutdown_tx: broadcast::Sender<()>,
    shutdown_rx: broadcast::Receiver<()>,
}

impl MevEngine {
    pub async fn new(config: Config, worker_threads: usize) -> Result<Self> {
        info!("Initializing MEV Engine with {} workers", worker_threads);
        
        let config = Arc::new(config);
        
        // Initialize RPC client with custom transport
        let rpc_client = Arc::new(
            RpcClient::new_with_timeout(
                config.rpc_url.clone(),
                Duration::from_millis(config.rpc_timeout_ms),
            )
        );
        
        // Initialize components
        let mempool_monitor = Arc::new(
            MempoolMonitor::new(config.clone(), rpc_client.clone()).await?
        );
        
        let submission_engine = Arc::new(
            SubmissionEngine::new(config.clone(), rpc_client.clone()).await?
        );
        
        let bundle_builder = Arc::new(
            BundleBuilder::new(config.clone())
        );
        
        let metrics = Arc::new(Metrics::new()?);
        
        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        
        Ok(Self {
            config,
            rpc_client,
            mempool_monitor,
            submission_engine,
            bundle_builder,
            metrics,
            active_bundles: Arc::new(DashMap::new()),
            bundle_semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_BUNDLES)),
            shutdown_tx,
            shutdown_rx,
        })
    }
    
    pub async fn run(&self) -> Result<()> {
        info!("Starting MEV engine main loop");
        
        // Start component tasks
        let mempool_task = self.start_mempool_monitoring();
        let opportunity_task = self.start_opportunity_scanner();
        let submission_task = self.start_submission_handler();
        let metrics_task = self.start_metrics_reporter();
        
        // Wait for all tasks
        let mut shutdown_rx = self.shutdown_rx.resubscribe();
        tokio::select! {
            res = mempool_task => {
                if let Err(e) = res {
                    error!("Mempool monitoring failed: {}", e);
                }
            }
            res = opportunity_task => {
                if let Err(e) = res {
                    error!("Opportunity scanner failed: {}", e);
                }
            }
            res = submission_task => {
                if let Err(e) = res {
                    error!("Submission handler failed: {}", e);
                }
            }
            res = metrics_task => {
                if let Err(e) = res {
                    error!("Metrics reporter failed: {}", e);
                }
            }
            _ = shutdown_rx.recv() => {
                info!("Received shutdown signal");
            }
        }
        
        Ok(())
    }
    
    async fn start_mempool_monitoring(&self) -> Result<()> {
        let mempool = self.mempool_monitor.clone();
        let metrics = self.metrics.clone();
        let mut shutdown_rx = self.shutdown_rx.resubscribe();
        
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_millis(10));
            
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        let start = Instant::now();
                        
                        match mempool.scan_transactions().await {
                            Ok(txs) => {
                                metrics.record_mempool_scan(txs.len(), start.elapsed());
                                debug!("Scanned {} transactions", txs.len());
                            }
                            Err(e) => {
                                warn!("Mempool scan error: {}", e);
                                metrics.increment_errors("mempool_scan");
                            }
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        info!("Mempool monitoring stopped");
                        break;
                    }
                }
            }
            
            Ok::<(), anyhow::Error>(())
        }).await?
    }
    
    async fn start_opportunity_scanner(&self) -> Result<()> {
        let bundle_builder = self.bundle_builder.clone();
        let submission_engine = self.submission_engine.clone();
        let active_bundles = self.active_bundles.clone();
        let semaphore = self.bundle_semaphore.clone();
        let metrics = self.metrics.clone();
        let mut shutdown_rx = self.shutdown_rx.resubscribe();
        
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_millis(5));
            
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        // Acquire semaphore permit
                        let permit = match semaphore.try_acquire() {
                            Ok(p) => p,
                            Err(_) => {
                                debug!("Bundle limit reached, skipping scan");
                                continue;
                            }
                        };
                        
                        let start = Instant::now();
                        
                        // Scan for opportunities
                        match bundle_builder.find_opportunities().await {
                            Ok(opportunities) => {
                                let opps_len = opportunities.len();
                                for opp in opportunities {
                                    // Build bundle
                                    match bundle_builder.build_bundle(opp).await {
                                        Ok(bundle) => {
                                            let sig = bundle.signature();
                                            active_bundles.insert(sig, bundle.clone());
                                            
                                            // Submit bundle
                                            if let Err(e) = submission_engine.submit(bundle).await {
                                                error!("Bundle submission failed: {}", e);
                                                active_bundles.remove(&sig);
                                            }
                                        }
                                        Err(e) => {
                                            warn!("Bundle building failed: {}", e);
                                        }
                                    }
                                }
                                
                                metrics.record_opportunity_scan(
                                    opps_len,
                                    start.elapsed()
                                );
                            }
                            Err(e) => {
                                error!("Opportunity scan error: {}", e);
                                metrics.increment_errors("opportunity_scan");
                            }
                        }
                        
                        drop(permit);
                    }
                    _ = shutdown_rx.recv() => {
                        info!("Opportunity scanner stopped");
                        break;
                    }
                }
            }
            
            Ok::<(), anyhow::Error>(())
        }).await?
    }
    
    async fn start_submission_handler(&self) -> Result<()> {
        let submission_engine = self.submission_engine.clone();
        let active_bundles = self.active_bundles.clone();
        let metrics = self.metrics.clone();
        let mut shutdown_rx = self.shutdown_rx.resubscribe();
        
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_millis(100));
            
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        // Check bundle statuses
                        let mut expired = Vec::new();
                        
                        for entry in active_bundles.iter() {
                            let (sig, bundle) = entry.pair();
                            
                            if bundle.is_expired() {
                                expired.push(*sig);
                                metrics.increment_bundle_expired();
                            } else if let Ok(confirmed) = submission_engine.check_status(sig).await {
                                if confirmed {
                                    metrics.increment_bundle_success();
                                    expired.push(*sig);
                                }
                            }
                        }
                        
                        // Remove expired/confirmed bundles
                        for sig in expired {
                            active_bundles.remove(&sig);
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        info!("Submission handler stopped");
                        break;
                    }
                }
            }
            
            Ok::<(), anyhow::Error>(())
        }).await?
    }
    
    async fn start_metrics_reporter(&self) -> Result<()> {
        let metrics = self.metrics.clone();
        let mut shutdown_rx = self.shutdown_rx.resubscribe();
        
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(10));
            
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        metrics.report();
                    }
                    _ = shutdown_rx.recv() => {
                        info!("Metrics reporter stopped");
                        break;
                    }
                }
            }
            
            Ok::<(), anyhow::Error>(())
        }).await?
    }
    
    pub async fn shutdown(&self) -> Result<()> {
        info!("Initiating MEV engine shutdown");
        
        // Send shutdown signal
        let _ = self.shutdown_tx.send(());
        
        // Wait for active bundles to complete
        let deadline = Instant::now() + Duration::from_secs(5);
        while !self.active_bundles.is_empty() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        
        if !self.active_bundles.is_empty() {
            warn!("Forcing shutdown with {} active bundles", self.active_bundles.len());
        }
        
        Ok(())
    }
}