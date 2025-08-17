use anyhow::Result;
use std::sync::Arc;
use tokio::runtime::Builder;
use nix::sched::{CpuSet, sched_setaffinity};
use nix::unistd::Pid;
use tracing::{info, error};

mod core;
mod network;
mod ml_inference;
mod submission;
mod database;
mod monitoring;
mod risk_management;

use crate::core::SandwichDetector;
use crate::monitoring::MetricsCollector;
use crate::risk_management::RiskManager;

fn setup_cpu_affinity(core_id: usize) -> Result<()> {
    let mut cpu_set = CpuSet::new();
    cpu_set.set(core_id)?;
    sched_setaffinity(Pid::from_raw(0), &cpu_set)?;
    Ok(())
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("mev_sandwich=debug")
        .json()
        .init();
    
    // Setup SCHED_FIFO priority
    unsafe {
        let param = libc::sched_param { sched_priority: 99 };
        libc::sched_setscheduler(0, libc::SCHED_FIFO, &param);
    }
    
    // Pin to specific CPU cores for network and ML threads
    let network_cores = vec![0, 1, 2, 3];  // Network processing
    let ml_cores = vec![4, 5];            // ML inference
    let submission_cores = vec![6, 7];    // Bundle submission
    
    // Create runtime with pinned threads
    let runtime = Builder::new_multi_thread()
        .worker_threads(16)
        .enable_all()
        .thread_name("mev-sandwich")
        .on_thread_start(move || {
            // Pin threads to cores
            let thread_id = std::thread::current().id();
            info!("Starting thread {:?}", thread_id);
        })
        .build()?;
    
    runtime.block_on(async {
        info!("MEV Sandwich Detector v1.0 - Independent Ultra-Low Latency Engine");
        
        // Initialize components
        let metrics = Arc::new(MetricsCollector::new());
        let risk_manager = Arc::new(RiskManager::new());
        
        // Start the sandwich detector
        let detector = SandwichDetector::new(
            metrics.clone(),
            risk_manager.clone(),
            network_cores,
            ml_cores,
            submission_cores,
        ).await?;
        
        // Run the detector
        detector.run().await
    })
}