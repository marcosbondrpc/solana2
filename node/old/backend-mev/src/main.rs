use backend_mev::{
    initialize, MevConfig,
    congestion::configure_transport,
    dual_path::{DualPathSubmitter, SubmissionRequest, SubmissionPriority},
    telemetry::{MetricsCollector, TelemetryExporter},
    pipeline::{IoUringReceiver, ZeroCopyPipeline, PacketPool},
    mev_engine::MevEngine,
    optimization,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Builder;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use solana_sdk::signature::Keypair;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "backend_mev=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    tracing::info!("Starting MEV Backend with Ultra-High Performance Mode");
    
    // Load configuration
    let config = MevConfig {
        target_block_time_us: 400_000,
        max_inflight_packets: 24,
        initial_rtt_us: 1600,
        jito_tip_bps: 50,
        enable_hw_timestamps: true,
        simd_level: 2,
        submission_threads: 4,
        enable_io_uring: true,
    };
    
    // Initialize system optimizations
    optimization::initialize_all()?;
    
    // Initialize MEV context
    let context = initialize(config.clone()).await?;
    
    // Create packet pool for zero-allocation
    let packet_pool = Arc::new(PacketPool::new(10_000));
    
    // Initialize dual-path submitter
    let auth_keypair = Arc::new(Keypair::new());
    let submitter = Arc::new(DualPathSubmitter::new(
        vec![
            "tpu1.solana.com:8003".to_string(),
            "tpu2.solana.com:8003".to_string(),
            "tpu3.solana.com:8003".to_string(),
        ],
        "jito.block-engine.com:8005".to_string(),
        auth_keypair.clone(),
        config.jito_tip_bps,
    )?);
    
    // Start submission workers
    submitter.start_workers(config.submission_threads).await;
    
    // Initialize MEV engine
    let mev_engine = Arc::new(MevEngine::new());
    
    // Create processing pipeline
    let (packet_tx, packet_rx) = crossbeam::channel::bounded(10_000);
    let (opportunity_tx, opportunity_rx) = crossbeam::channel::bounded(1_000);
    
    // Start io_uring packet receiver
    if config.enable_io_uring {
        let mut receiver = IoUringReceiver::new(
            "0.0.0.0:8899",
            packet_pool.clone(),
            packet_tx.clone(),
        )?;
        
        tokio::spawn(async move {
            if let Err(e) = receiver.start_receiving().await {
                tracing::error!("IoUring receiver error: {}", e);
            }
        });
    }
    
    // Start MEV processing workers
    for _ in 0..config.submission_threads {
        let engine = mev_engine.clone();
        let rx = packet_rx.clone();
        let tx = opportunity_tx.clone();
        
        tokio::spawn(async move {
            while let Ok(packet) = rx.recv() {
                let opportunities = engine.process_update(&packet.data);
                for opp in opportunities {
                    let _ = tx.try_send(opp);
                }
            }
        });
    }
    
    // Start opportunity execution workers
    for _ in 0..2 {
        let engine = mev_engine.clone();
        let submitter = submitter.clone();
        let rx = opportunity_rx.clone();
        let executor = auth_keypair.clone();
        
        tokio::spawn(async move {
            while let Ok(opportunity) = rx.recv() {
                // Convert opportunity to submission request
                let request = match &opportunity {
                    backend_mev::mev_engine::MevOpportunity::Arbitrage(arb) => {
                        if let Ok(tx) = engine.arbitrage.build_transaction(arb, &executor) {
                            Some(SubmissionRequest {
                                transaction: tx,
                                estimated_profit: arb.profit_lamports,
                                deadline: tokio::time::Instant::now() + Duration::from_millis(100),
                                priority: if arb.profit_lamports > 100_000_000 {
                                    SubmissionPriority::UltraHigh
                                } else {
                                    SubmissionPriority::High
                                },
                                bundle_id: Some(arb.id.clone()),
                            })
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                
                if let Some(req) = request {
                    if let Err(e) = submitter.submit(req).await {
                        tracing::error!("Submission error: {}", e);
                    }
                }
            }
        });
    }
    
    // Start telemetry exporter
    let telemetry = TelemetryExporter::new(
        context.metrics.clone(),
        Duration::from_secs(10),
    );
    telemetry.start().await;
    
    // Start Prometheus metrics server
    tokio::spawn(async move {
        let metrics_addr = "0.0.0.0:9090".parse().unwrap();
        tracing::info!("Starting metrics server on {}", metrics_addr);
        
        let app = axum::Router::new()
            .route("/metrics", axum::routing::get(metrics_handler));
        
        axum::Server::bind(&metrics_addr)
            .serve(app.into_make_service())
            .await
            .unwrap();
    });
    
    tracing::info!("MEV Backend initialized and running");
    tracing::info!("Performance targets:");
    tracing::info!("  - Packet processing: <100μs");
    tracing::info!("  - Submission latency: <1ms");
    tracing::info!("  - Throughput: 100k+ TPS");
    tracing::info!("  - Zero allocations in hot path");
    
    // Keep running
    tokio::signal::ctrl_c().await?;
    tracing::info!("Shutting down MEV Backend");
    
    Ok(())
}

async fn metrics_handler() -> String {
    use prometheus::{Encoder, TextEncoder};
    
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}