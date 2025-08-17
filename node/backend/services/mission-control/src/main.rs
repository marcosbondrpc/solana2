mod api;
mod cache;
mod collectors;
mod error;
mod metrics;
mod models;
mod ws;

use crate::api::{ApiState, *};
use crate::cache::CacheManager;
use crate::collectors::{JitoCollector, QuicCollector, RpcCollector};
use crate::metrics::MetricsRecorder;
use crate::ws::{WsState, *};
use axum::{
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{info, Level};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive(Level::INFO.into())
                .add_directive("mission_control=debug".parse()?)
        )
        .with_target(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .json()
        .finish();
    
    tracing::subscriber::set_global_default(subscriber)?;
    
    info!("Starting Solana Mission Control Service");
    
    // Load configuration
    dotenv::dotenv().ok();
    let redis_url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let rpc_endpoints: Vec<String> = std::env::var("RPC_ENDPOINTS")
        .unwrap_or_else(|_| "https://api.mainnet-beta.solana.com".to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let tpu_endpoint: SocketAddr = std::env::var("TPU_ENDPOINT")
        .unwrap_or_else(|_| "127.0.0.1:8003".to_string())
        .parse()?;
    let our_stake: u64 = std::env::var("VALIDATOR_STAKE")
        .unwrap_or_else(|_| "1000000000000".to_string())
        .parse()?;
    
    // Initialize components
    info!("Initializing cache manager");
    let cache = Arc::new(CacheManager::new(&redis_url).await?);
    
    info!("Initializing metrics recorder");
    let metrics_recorder = Arc::new(MetricsRecorder::new()?);
    
    info!("Initializing RPC collector");
    let rpc_collector = Arc::new(RpcCollector::new(rpc_endpoints, Arc::clone(&metrics_recorder))?);
    
    info!("Initializing Jito collector");
    let jito_collector = Arc::new(JitoCollector::new(Arc::clone(&metrics_recorder))?);
    
    info!("Initializing QUIC collector");
    let quic_collector = Arc::new(
        QuicCollector::new(tpu_endpoint, our_stake, Arc::clone(&metrics_recorder)).await?
    );
    
    // Start background collectors
    info!("Starting background collectors");
    
    let rpc_collector_clone = Arc::clone(&rpc_collector);
    tokio::spawn(async move {
        rpc_collector_clone.start_collection().await;
    });
    
    let jito_collector_clone = Arc::clone(&jito_collector);
    tokio::spawn(async move {
        jito_collector_clone.start_collection().await;
    });
    
    let quic_collector_clone = Arc::clone(&quic_collector);
    tokio::spawn(async move {
        quic_collector_clone.start_collection().await;
    });
    
    // Create API state
    let api_state = Arc::new(ApiState {
        cache: Arc::clone(&cache),
        rpc_collector: Arc::clone(&rpc_collector),
        jito_collector: Arc::clone(&jito_collector),
        quic_collector: Arc::clone(&quic_collector),
    });
    
    // Create WebSocket state
    let ws_state = Arc::new(WsState {
        cache: Arc::clone(&cache),
    });
    
    // Build REST API routes
    let api_routes = Router::new()
        .route("/api/mission-control/overview", get(get_overview))
        .route("/api/mission-control/node-summary", get(get_node_summary))
        .route("/api/mission-control/consensus-health", get(get_consensus_health))
        .route("/api/mission-control/cluster-perf", get(get_cluster_performance))
        .route("/api/mission-control/jito-status", get(get_jito_status))
        .route("/api/mission-control/rpc-metrics", get(get_rpc_metrics))
        .route("/api/mission-control/timing-waterfall", get(get_timing_waterfall))
        .route("/api/mission-control/tip-intelligence", get(get_tip_intelligence))
        .route("/api/mission-control/bundle-success", get(get_bundle_success))
        .route("/api/mission-control/shredstream", get(get_shredstream_metrics))
        .route("/api/mission-control/quic-health", get(get_quic_health))
        .route("/api/mission-control/qos-peering", get(get_qos_peering))
        .route("/api/mission-control/gossip-metrics", get(get_gossip_metrics))
        .route("/api/mission-control/node-control", post(node_control))
        .route("/api/mission-control/preflight-checks", get(get_preflight_checks))
        .with_state(api_state);
    
    // Build WebSocket routes
    let ws_routes = Router::new()
        .route("/ws/mission-control", get(ws_mission_control))
        .route("/ws/jito-tips", get(ws_jito_tips))
        .route("/ws/bundle-status", get(ws_bundle_status))
        .with_state(ws_state);
    
    // Prometheus metrics endpoint
    let metrics_recorder_clone = Arc::clone(&metrics_recorder);
    let metrics_route = Router::new()
        .route("/metrics", get(move || async move {
            metrics_recorder_clone.export_metrics()
        }));
    
    // Health check endpoint
    let cache_clone = Arc::clone(&cache);
    let health_route = Router::new()
        .route("/health", get(move || async move {
            let redis_healthy = cache_clone.health_check().await;
            if redis_healthy {
                (axum::http::StatusCode::OK, "healthy")
            } else {
                (axum::http::StatusCode::SERVICE_UNAVAILABLE, "unhealthy")
            }
        }));
    
    // Combine all routes
    let app = Router::new()
        .merge(api_routes)
        .merge(ws_routes)
        .merge(metrics_route)
        .merge(health_route)
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any)
        )
        .layer(TraceLayer::new_for_http());
    
    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("Mission Control service listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}