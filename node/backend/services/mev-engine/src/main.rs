use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tokio::signal;
use tracing::{info, error};

mod engine;
mod mempool;
mod bundle;
mod submission;
mod metrics;
mod config;

use engine::MevEngine;
use config::Config;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "config.toml")]
    config: String,
    
    /// Enable production mode optimizations
    #[arg(long)]
    production: bool,
    
    /// Number of worker threads
    #[arg(short, long, default_value = "16")]
    workers: usize,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    // Parse arguments
    let args = Args::parse();
    
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("mev_engine=debug,info")
        .with_target(false)
        .json()
        .init();
    
    info!("Starting MEV Engine v{}", env!("CARGO_PKG_VERSION"));
    
    // Load configuration
    let config = Config::load(&args.config)?;
    
    // Set thread affinity for optimal performance
    if args.production {
        set_cpu_affinity()?;
        set_memory_policy()?;
    }
    
    // Initialize MEV engine
    let engine = Arc::new(MevEngine::new(config, args.workers).await?);
    
    // Start engine
    let engine_handle = {
        let engine = engine.clone();
        tokio::spawn(async move {
            if let Err(e) = engine.run().await {
                error!("MEV engine error: {}", e);
            }
        })
    };
    
    // Wait for shutdown signal
    shutdown_signal().await;
    
    info!("Shutting down MEV engine...");
    engine.shutdown().await?;
    engine_handle.await?;
    
    info!("MEV engine stopped");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

fn set_cpu_affinity() -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        use nix::sched::{sched_setaffinity, CpuSet};
        use nix::unistd::Pid;
        
        let mut cpu_set = CpuSet::new();
        // Pin to high-performance cores (typically 0-7 on modern CPUs)
        for i in 0..8 {
            cpu_set.set(i)?;
        }
        sched_setaffinity(Pid::from_raw(0), &cpu_set)?;
        info!("CPU affinity set to cores 0-7");
    }
    Ok(())
}

fn set_memory_policy() -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        use libc::{mlockall, MCL_CURRENT, MCL_FUTURE};
        
        unsafe {
            // Lock all current and future memory pages
            if mlockall(MCL_CURRENT | MCL_FUTURE) != 0 {
                error!("Failed to lock memory pages");
            } else {
                info!("Memory pages locked");
            }
        }
    }
    Ok(())
}