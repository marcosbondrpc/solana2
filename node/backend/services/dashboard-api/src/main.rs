use anyhow::Result;
use axum::{
    extract::{ws::WebSocketUpgrade, Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use std::{
    net::SocketAddr,
    sync::Arc,
    time::Duration,
};
use tokio::{net::TcpListener, signal};
use tower_http::{
    compression::CompressionLayer,
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::{error, info};

mod config;
mod handlers;
mod metrics;
mod models;
mod services;
mod websocket;

use config::Config;
use services::{
    node_metrics::NodeMetricsService,
    scrapper::ScrapperService,
    cache::CacheManager,
    database::DatabaseService,
    kafka::KafkaProducer,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    /// Port to listen on
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Enable production mode optimizations
    #[arg(long)]
    production: bool,

    /// Number of worker threads
    #[arg(short, long, default_value = "32")]
    workers: usize,
}

/// Shared application state
#[derive(Clone)]
struct AppState {
    node_metrics: Arc<NodeMetricsService>,
    scrapper: Arc<ScrapperService>,
    cache: Arc<CacheManager>,
    db: Arc<DatabaseService>,
    kafka: Arc<KafkaProducer>,
    config: Arc<Config>,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing with JSON output for production
    tracing_subscriber::fmt()
        .with_env_filter("dashboard_api=debug,tower_http=debug,info")
        .with_target(false)
        .json()
        .init();

    info!("Starting Dashboard API v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let config = Arc::new(Config::load(&args.config)?);

    // Set production optimizations
    if args.production {
        set_cpu_affinity()?;
        set_memory_policy()?;
        set_thread_priority()?;
    }

    // Initialize services
    info!("Initializing services...");
    
    let cache = Arc::new(CacheManager::new(&config.redis).await?);
    let db = Arc::new(DatabaseService::new(&config.clickhouse).await?);
    let kafka = Arc::new(KafkaProducer::new(&config.kafka)?);
    let node_metrics = Arc::new(NodeMetricsService::new(
        config.clone(),
        cache.clone(),
        kafka.clone(),
    ).await?);
    let scrapper = Arc::new(ScrapperService::new(
        config.clone(),
        db.clone(),
        cache.clone(),
        kafka.clone(),
    ).await?);

    // Create shared state
    let state = AppState {
        node_metrics,
        scrapper,
        cache,
        db,
        kafka,
        config,
    };

    // Start background services
    start_background_services(state.clone());

    // Build router with all endpoints
    let app = build_router(state);

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    let listener = TcpListener::bind(addr).await?;
    
    info!("Dashboard API listening on {}", addr);

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;

    info!("Dashboard API stopped");
    Ok(())
}

fn build_router(state: AppState) -> Router {
    Router::new()
        // Node metrics endpoints
        .route("/api/node/metrics", get(handlers::node::get_metrics))
        .route("/api/node/latency", get(handlers::node::get_latency))
        .route("/api/node/connections", get(handlers::node::get_connections))
        .route("/api/node/network", get(handlers::node::get_network_stats))
        
        // Historical data scraping endpoints
        .route("/api/scrapper/datasets", get(handlers::scrapper::list_datasets))
        .route("/api/scrapper/datasets", post(handlers::scrapper::create_dataset))
        .route("/api/scrapper/datasets/:id", get(handlers::scrapper::get_dataset))
        .route("/api/scrapper/datasets/:id", delete(handlers::scrapper::delete_dataset))
        .route("/api/scrapper/collect", post(handlers::scrapper::start_collection))
        .route("/api/scrapper/collect/:id/stop", post(handlers::scrapper::stop_collection))
        .route("/api/scrapper/export", post(handlers::scrapper::export_data))
        .route("/api/scrapper/import", post(handlers::scrapper::import_data))
        
        // ML model endpoints
        .route("/api/scrapper/models", get(handlers::ml::list_models))
        .route("/api/scrapper/models", post(handlers::ml::create_model))
        .route("/api/scrapper/train", post(handlers::ml::start_training))
        .route("/api/scrapper/train/:id", get(handlers::ml::get_training_status))
        .route("/api/scrapper/models/:id/predict", post(handlers::ml::predict))
        
        // WebSocket endpoints
        .route("/ws/node-metrics", get(websocket::node_metrics_ws))
        .route("/ws/scrapper-progress", get(websocket::scrapper_progress_ws))
        
        // Health and metrics
        .route("/health", get(health_check))
        .route("/metrics", get(prometheus_metrics))
        
        // Add middleware
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(CompressionLayer::new())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now(),
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

async fn prometheus_metrics() -> impl IntoResponse {
    use prometheus::{Encoder, TextEncoder};
    
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    
    String::from_utf8(buffer).unwrap()
}

fn start_background_services(state: AppState) {
    // Start metrics collection
    let metrics_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(100)); // 10Hz update rate
        loop {
            interval.tick().await;
            if let Err(e) = metrics_state.node_metrics.collect_metrics().await {
                error!("Failed to collect metrics: {}", e);
            }
        }
    });

    // Start cache warmup
    let cache_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        loop {
            interval.tick().await;
            if let Err(e) = cache_state.cache.warm_cache().await {
                error!("Failed to warm cache: {}", e);
            }
        }
    });

    // Start database compaction
    let db_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(3600)); // Hourly
        loop {
            interval.tick().await;
            if let Err(e) = db_state.db.optimize_tables().await {
                error!("Failed to optimize database: {}", e);
            }
        }
    });
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
        // Use different cores than MEV engine for isolation
        for i in 8..16 {
            cpu_set.set(i)?;
        }
        sched_setaffinity(Pid::from_raw(0), &cpu_set)?;
        info!("CPU affinity set to cores 8-15");
    }
    Ok(())
}

fn set_memory_policy() -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        use libc::{mlockall, MCL_CURRENT, MCL_FUTURE};
        
        unsafe {
            if mlockall(MCL_CURRENT | MCL_FUTURE) != 0 {
                error!("Failed to lock memory pages");
            } else {
                info!("Memory pages locked");
            }
        }
    }
    Ok(())
}

fn set_thread_priority() -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        use libc::{setpriority, PRIO_PROCESS};
        
        unsafe {
            // Set high priority but not real-time to avoid blocking system
            if setpriority(PRIO_PROCESS, 0, -10) != 0 {
                error!("Failed to set thread priority");
            } else {
                info!("Thread priority set to -10");
            }
        }
    }
    Ok(())
}