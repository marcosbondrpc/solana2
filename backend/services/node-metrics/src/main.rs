use anyhow::Result;
use axum::{
    extract::State,
    response::Json,
    routing::{get, post},
    Router,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::interval;
use tracing::{error, info};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NodeMetrics {
    rpc_latency: f64,
    websocket_latency: f64,
    geyser_latency: f64,
    jito_latency: f64,
    block_height: u64,
    slot: u64,
    tps: f64,
    peers: u32,
    timestamp: u64,
}

#[derive(Clone)]
struct AppState {
    metrics: Arc<DashMap<String, NodeMetrics>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    info!("Starting Node Metrics Service on port 8081");

    let state = AppState {
        metrics: Arc::new(DashMap::new()),
    };

    // Start metrics collection task
    let metrics_state = state.clone();
    tokio::spawn(async move {
        collect_metrics(metrics_state).await;
    });

    // Build API routes
    let app = Router::new()
        .route("/api/node/metrics", get(get_metrics))
        .route("/api/node/health", get(health_check))
        .route("/api/node/config", post(update_config))
        .with_state(state);

    // Start server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8081")
        .await?;
    
    info!("Node Metrics API listening on http://0.0.0.0:8081");
    
    axum::serve(listener, app).await?;

    Ok(())
}

async fn collect_metrics(state: AppState) {
    let mut ticker = interval(Duration::from_millis(100));
    let start_time = Instant::now();
    
    loop {
        ticker.tick().await;
        
        // Simulate metrics collection with realistic values
        let elapsed = start_time.elapsed().as_secs();
        let metrics = NodeMetrics {
            rpc_latency: 20.0 + (elapsed as f64 * 0.1).sin() * 10.0,
            websocket_latency: 15.0 + (elapsed as f64 * 0.2).cos() * 8.0,
            geyser_latency: 25.0 + (elapsed as f64 * 0.15).sin() * 12.0,
            jito_latency: 18.0 + (elapsed as f64 * 0.25).cos() * 9.0,
            block_height: 250_000_000 + elapsed * 2,
            slot: 280_000_000 + elapsed * 4,
            tps: 3000.0 + (elapsed as f64 * 0.3).sin() * 500.0,
            peers: 1000 + ((elapsed % 100) as u32),
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        };
        
        state.metrics.insert("current".to_string(), metrics);
    }
}

async fn get_metrics(State(state): State<AppState>) -> Json<serde_json::Value> {
    let metrics = state.metrics.get("current")
        .map(|m| m.clone())
        .unwrap_or_else(|| NodeMetrics {
            rpc_latency: 0.0,
            websocket_latency: 0.0,
            geyser_latency: 0.0,
            jito_latency: 0.0,
            block_height: 0,
            slot: 0,
            tps: 0.0,
            peers: 0,
            timestamp: 0,
        });
    
    Json(serde_json::json!({
        "status": "success",
        "data": metrics
    }))
}

async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "node-metrics",
        "timestamp": chrono::Utc::now().timestamp_millis()
    }))
}

async fn update_config(Json(config): Json<serde_json::Value>) -> Json<serde_json::Value> {
    info!("Received config update: {:?}", config);
    
    Json(serde_json::json!({
        "status": "success",
        "message": "Configuration updated"
    }))
}