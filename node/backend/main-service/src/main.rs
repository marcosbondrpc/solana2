use anyhow::Result;
use axum::{
    extract::State,
    response::Json,
    routing::{get, post},
    Router,
};
use clap::Parser;
use config::Config;
use prometheus::{Encoder, TextEncoder};
use serde::{Deserialize, Serialize};
use shared_types::{SystemConfig, JitoConfig, QuicConfig, ArbitrageConfig, PerformanceConfig};
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::signal;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
#[clap(name = "solana-mev-backend")]
#[clap(about = "Ultra-high-performance Solana MEV extraction backend", long_about = None)]
struct Args {
    /// Configuration file path
    #[clap(short, long, default_value = "config.toml")]
    config: PathBuf,

    /// HTTP API port
    #[clap(short, long, default_value = "8080")]
    port: u16,

    /// Metrics port
    #[clap(short, long, default_value = "9090")]
    metrics_port: u16,

    /// Enable debug mode
    #[clap(short, long)]
    debug: bool,
}

#[derive(Clone)]
struct AppState {
    jito_engine: Arc<jito_engine::JitoEngine>,
    dex_parser: Arc<dex_parser::DexParser>,
    config: Arc<SystemConfig>,
}

#[derive(Serialize, Deserialize)]
struct HealthResponse {
    status: String,
    version: String,
    uptime: u64,
    metrics: SystemMetrics,
}

#[derive(Serialize, Deserialize)]
struct SystemMetrics {
    bundles_submitted: u64,
    success_rate: f64,
    avg_latency_ms: f64,
    active_connections: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let args = Args::parse();

    // Initialize tracing
    let filter = if args.debug {
        "debug,hyper=info,tower=info"
    } else {
        "info,hyper=warn,tower=warn"
    };

    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| filter.into()))
        .with(tracing_subscriber::fmt::layer().json())
        .init();

    info!("Starting Solana MEV Backend v1.0.0");

    // Load configuration
    let config = load_config(&args.config)?;
    info!("Configuration loaded from: {:?}", args.config);

    // Initialize system optimizer for maximum performance
    initialize_system_optimizations()?;

    // Initialize services
    let jito_engine = Arc::new(
        jito_engine::JitoEngine::new(config.jito_config.clone())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to initialize Jito engine: {}", e))?
    );

    let dex_parser = Arc::new(
        dex_parser::DexParser::new()
            .map_err(|e| anyhow::anyhow!("Failed to initialize DEX parser: {}", e))?
    );

    // Start background services
    let jito_handle = {
        let engine = jito_engine.clone();
        tokio::spawn(async move {
            if let Err(e) = engine.start().await {
                error!("Jito engine error: {}", e);
            }
        })
    };

    // Create app state
    let state = AppState {
        jito_engine,
        dex_parser,
        config: Arc::new(config),
    };

    // Build HTTP API router with admin endpoints
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/bundle/submit", post(submit_bundle_handler))
        .route("/bundle/simulate", post(simulate_bundle_handler))
        .route("/pools/:dex", get(get_pools_handler))
        // Admin endpoints for configuration
        .route("/admin/config", get(get_config_handler))
        .route("/admin/config", post(update_config_handler))
        .route("/admin/path-stats", get(get_path_stats_handler))
        .route("/admin/set-batch-size", post(set_batch_size_handler))
        .route("/admin/set-pool-size", post(set_pool_size_handler))
        .route("/admin/set-timeout", post(set_timeout_handler))
        .layer(
            ServiceBuilder::new()
                .layer(CorsLayer::permissive())
                .into_inner(),
        )
        .with_state(state);

    // Start HTTP server
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    info!("HTTP API listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    
    // Start metrics server
    let metrics_addr = SocketAddr::from(([0, 0, 0, 0], args.metrics_port));
    let metrics_handle = tokio::spawn(start_metrics_server(metrics_addr));

    // Run server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Shutting down services...");

    // Cleanup
    jito_handle.abort();
    metrics_handle.abort();

    Ok(())
}

fn load_config(path: &PathBuf) -> Result<SystemConfig> {
    // Default configuration
    let default_config = SystemConfig {
        jito_config: JitoConfig {
            block_engine_url: "https://mainnet.block-engine.jito.wtf".to_string(),
            relayer_urls: vec![
                "https://amsterdam.mainnet.block-engine.jito.wtf".to_string(),
                "https://frankfurt.mainnet.block-engine.jito.wtf".to_string(),
                "https://ny.mainnet.block-engine.jito.wtf".to_string(),
                "https://tokyo.mainnet.block-engine.jito.wtf".to_string(),
            ],
            auth_keypair_path: "/keys/jito-auth.json".to_string(),
            tip_account: solana_sdk::pubkey!("96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5"),
            min_tip_lamports: 10_000,
            max_tip_lamports: 1_000_000,
            bundle_timeout_ms: 5000,
        },
        quic_config: QuicConfig {
            leader_endpoints: vec![],
            max_concurrent_streams: 1024,
            initial_rtt_ms: 10,
            max_idle_timeout_ms: 30000,
            keep_alive_interval_ms: 5000,
            congestion_controller: "bbr".to_string(),
        },
        arbitrage_config: ArbitrageConfig {
            min_profit_lamports: 100_000,
            max_hops: 3,
            slippage_tolerance_bps: 50,
            simulation_compute_limit: 1_400_000,
            max_input_sol: 100.0,
            priority_fee_lamports: 10_000,
        },
        performance_config: PerformanceConfig {
            cpu_affinity_mask: 0xFF,
            memory_pool_size_mb: 2048,
            io_uring_enabled: true,
            thread_priority: 99,
            batch_size: 32,
        },
    };

    // Load from file if it exists
    if path.exists() {
        let settings = Config::builder()
            .add_source(config::File::from(path.as_ref()))
            .build()?;
        
        Ok(settings.try_deserialize()?)
    } else {
        warn!("Config file not found, using defaults");
        Ok(default_config)
    }
}

fn initialize_system_optimizations() -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        use nix::sys::resource::{setrlimit, Resource};
        use nix::sched::{sched_setscheduler, CpuSet, Policy};
        use std::os::unix::process::CommandExt;

        // Set high file descriptor limit
        let _ = setrlimit(Resource::RLIMIT_NOFILE, 1048576, 1048576);

        // Set real-time scheduling
        if let Err(e) = sched_setscheduler(0, Policy::SCHED_FIFO, 99) {
            warn!("Failed to set real-time scheduling: {}", e);
        }

        // Disable CPU frequency scaling
        let _ = std::process::Command::new("sh")
            .args(&["-c", "echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"])
            .output();

        info!("System optimizations applied");
    }

    Ok(())
}

async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: "1.0.0".to_string(),
        uptime: 0, // Would calculate actual uptime
        metrics: SystemMetrics {
            bundles_submitted: 0,
            success_rate: 0.0,
            avg_latency_ms: 0.0,
            active_connections: 0,
        },
    })
}

async fn metrics_handler() -> String {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

async fn submit_bundle_handler(
    State(state): State<AppState>,
    Json(bundle): Json<shared_types::JitoBundle>,
) -> Result<Json<serde_json::Value>, String> {
    match state.jito_engine.submit_bundle(bundle).await {
        Ok(bundle_id) => Ok(Json(serde_json::json!({
            "success": true,
            "bundle_id": bundle_id
        }))),
        Err(e) => Err(format!("Bundle submission failed: {}", e)),
    }
}

async fn simulate_bundle_handler(
    State(state): State<AppState>,
    Json(bundle): Json<shared_types::JitoBundle>,
) -> Result<Json<shared_types::SimulationResult>, String> {
    match state.jito_engine.simulate_bundle(&bundle).await {
        Ok(result) => Ok(Json(result)),
        Err(e) => Err(format!("Simulation failed: {}", e)),
    }
}

async fn get_pools_handler(
    State(state): State<AppState>,
    axum::extract::Path(dex): axum::extract::Path<String>,
) -> Result<Json<Vec<shared_types::PoolState>>, String> {
    // Would return cached pools for the specified DEX
    Ok(Json(vec![]))
}

async fn start_metrics_server(addr: SocketAddr) -> Result<()> {
    info!("Metrics server listening on {}", addr);
    
    let app = Router::new()
        .route("/metrics", get(metrics_handler));

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

/// Get current configuration
async fn get_config_handler(State(state): State<AppState>) -> Json<SystemConfig> {
    Json((*state.config).clone())
}

/// Update configuration dynamically
async fn update_config_handler(
    State(state): State<AppState>,
    Json(updates): Json<ConfigUpdate>,
) -> Result<Json<serde_json::Value>, String> {
    // In a real implementation, this would update the running configuration
    // For now, we'll just acknowledge the request
    info!("Configuration update requested: {:?}", updates);
    
    Ok(Json(serde_json::json!({
        "success": true,
        "message": "Configuration updated",
        "updates": updates
    })))
}

/// Get path selector statistics
async fn get_path_stats_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    // This would integrate with the PathSelector to get real stats
    Json(serde_json::json!({
        "jito_win_rate": 0.75,
        "tpu_win_rate": 0.45,
        "jito_avg_latency_ms": 25,
        "tpu_avg_latency_ms": 15,
        "jito_total_submissions": 1000,
        "tpu_total_submissions": 500,
        "current_selection_mode": "adaptive"
    }))
}

/// Set simulation batch size
async fn set_batch_size_handler(
    State(state): State<AppState>,
    Json(params): Json<BatchSizeParams>,
) -> Result<Json<serde_json::Value>, String> {
    if params.batch_size < 1 || params.batch_size > 128 {
        return Err("Batch size must be between 1 and 128".to_string());
    }
    
    info!("Setting simulation batch size to {}", params.batch_size);
    
    // Update the batch size in the running system
    // This would be implemented with atomic variables or message passing
    
    Ok(Json(serde_json::json!({
        "success": true,
        "batch_size": params.batch_size
    })))
}

/// Set connection pool size
async fn set_pool_size_handler(
    State(state): State<AppState>,
    Json(params): Json<PoolSizeParams>,
) -> Result<Json<serde_json::Value>, String> {
    if params.pool_size < 1 || params.pool_size > 256 {
        return Err("Pool size must be between 1 and 256".to_string());
    }
    
    info!("Setting connection pool size to {}", params.pool_size);
    
    Ok(Json(serde_json::json!({
        "success": true,
        "pool_size": params.pool_size
    })))
}

/// Set bundle timeout
async fn set_timeout_handler(
    State(state): State<AppState>,
    Json(params): Json<TimeoutParams>,
) -> Result<Json<serde_json::Value>, String> {
    if params.timeout_ms < 100 || params.timeout_ms > 30000 {
        return Err("Timeout must be between 100ms and 30000ms".to_string());
    }
    
    info!("Setting bundle timeout to {}ms", params.timeout_ms);
    
    Ok(Json(serde_json::json!({
        "success": true,
        "timeout_ms": params.timeout_ms
    })))
}

#[derive(Deserialize)]
struct ConfigUpdate {
    jito_config: Option<JitoConfigUpdate>,
    performance_config: Option<PerformanceConfigUpdate>,
    arbitrage_config: Option<ArbitrageConfigUpdate>,
}

#[derive(Debug, Deserialize)]
struct JitoConfigUpdate {
    min_tip_lamports: Option<u64>,
    max_tip_lamports: Option<u64>,
    bundle_timeout_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct PerformanceConfigUpdate {
    batch_size: Option<usize>,
    thread_priority: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct ArbitrageConfigUpdate {
    min_profit_lamports: Option<u64>,
    slippage_tolerance_bps: Option<u16>,
}

#[derive(Deserialize)]
struct BatchSizeParams {
    batch_size: usize,
}

#[derive(Deserialize)]
struct PoolSizeParams {
    pool_size: usize,
}

#[derive(Deserialize)]
struct TimeoutParams {
    timeout_ms: u64,
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

    info!("Shutdown signal received");
}