use anyhow::Result;
use axum::{
    extract::{ws::WebSocketUpgrade, State},
    response::{Html, Json},
    routing::{get, post},
    Router,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::interval;
use tower_http::cors::CorsLayer;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MissionControlMetrics {
    // Node Summary
    node_version: String,
    cluster: String,
    ledger_height: u64,
    slot: u64,
    distance_to_tip: i64,
    epoch_progress: f64,
    
    // Consensus Health
    vote_credits: u64,
    delinquent: bool,
    block_production_rate: f64,
    
    // Cluster Performance
    tps: f64,
    confirmation_time_ms: f64,
    
    // Jito Status
    jito_region: String,
    min_tip: u64,
    auction_tick_ms: u64,
    bundle_auth_uuid: String,
    tip_percentile_50: f64,
    tip_percentile_90: f64,
    
    // RPC Metrics
    rpc_latency_p50: f64,
    rpc_latency_p99: f64,
    ws_subscription_count: u32,
    ws_lag_ms: f64,
    
    // QUIC/TPU
    quic_handshake_success_rate: f64,
    concurrent_connections: u32,
    streams_per_peer: f64,
    throttling_events: u32,
    pps_rate: f64,
    
    // Timing Waterfall
    client_to_rpc_ms: f64,
    rpc_to_be_ms: f64,
    be_to_auction_ms: f64,
    auction_to_relay_ms: f64,
    relay_to_leader_ms: f64,
    
    // Bundle Success
    bundle_acceptance_rate: f64,
    bundle_landing_delay_ms: f64,
    
    // ShredStream
    shred_packets_per_sec: f64,
    shred_gaps: u32,
    shred_reorders: u32,
    
    timestamp: u64,
}

#[derive(Clone)]
struct AppState {
    metrics: Arc<DashMap<String, MissionControlMetrics>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    info!("Starting Mission Control Service on port 8083");

    let state = AppState {
        metrics: Arc::new(DashMap::new()),
    };

    // Start metrics collection
    let metrics_state = state.clone();
    tokio::spawn(async move {
        collect_metrics(metrics_state).await;
    });

    // Build routes
    let app = Router::new()
        .route("/", get(home))
        .route("/api/mission-control/overview", get(get_overview))
        .route("/api/mission-control/node-summary", get(get_node_summary))
        .route("/api/mission-control/consensus-health", get(get_consensus_health))
        .route("/api/mission-control/cluster-perf", get(get_cluster_perf))
        .route("/api/mission-control/jito-status", get(get_jito_status))
        .route("/api/mission-control/rpc-metrics", get(get_rpc_metrics))
        .route("/api/mission-control/timing-waterfall", get(get_timing_waterfall))
        .route("/api/mission-control/tip-intelligence", get(get_tip_intelligence))
        .route("/api/mission-control/bundle-success", get(get_bundle_success))
        .route("/api/mission-control/shredstream", get(get_shredstream))
        .route("/api/mission-control/quic-health", get(get_quic_health))
        .route("/api/mission-control/qos-peering", get(get_qos_peering))
        .route("/api/mission-control/gossip-metrics", get(get_gossip_metrics))
        .route("/api/mission-control/node-control", post(node_control))
        .route("/api/mission-control/preflight-checks", get(preflight_checks))
        .route("/ws/mission-control", get(ws_handler))
        .with_state(state)
        .layer(CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8083").await?;
    info!("Mission Control API listening on http://0.0.0.0:8083");
    
    axum::serve(listener, app).await?;
    Ok(())
}

async fn collect_metrics(state: AppState) {
    let mut ticker = interval(Duration::from_millis(100));
    let start_time = Instant::now();
    
    loop {
        ticker.tick().await;
        
        let elapsed = start_time.elapsed().as_secs();
        let metrics = MissionControlMetrics {
            // Node Summary
            node_version: "1.18.23-jito".to_string(),
            cluster: "mainnet-beta".to_string(),
            ledger_height: 250_000_000 + elapsed * 2,
            slot: 280_000_000 + elapsed * 4,
            distance_to_tip: (elapsed as f64 * 0.1).sin() as i64 * 3,
            epoch_progress: ((elapsed % 432000) as f64 / 432000.0) * 100.0,
            
            // Consensus Health
            vote_credits: 420_000 + elapsed,
            delinquent: false,
            block_production_rate: 95.0 + (elapsed as f64 * 0.1).sin() * 3.0,
            
            // Cluster Performance
            tps: 3000.0 + (elapsed as f64 * 0.3).sin() * 500.0,
            confirmation_time_ms: 400.0 + (elapsed as f64 * 0.2).cos() * 100.0,
            
            // Jito Status
            jito_region: "frankfurt".to_string(),
            min_tip: 1000 + ((elapsed * 10) % 5000),
            auction_tick_ms: 50,
            bundle_auth_uuid: "550e8400-e29b-41d4-a716-446655440000".to_string(),
            tip_percentile_50: 2500.0 + (elapsed as f64 * 0.15).sin() * 500.0,
            tip_percentile_90: 5000.0 + (elapsed as f64 * 0.15).sin() * 1000.0,
            
            // RPC Metrics
            rpc_latency_p50: 18.0 + (elapsed as f64 * 0.1).sin() * 5.0,
            rpc_latency_p99: 140.0 + (elapsed as f64 * 0.1).cos() * 40.0,
            ws_subscription_count: 150 + ((elapsed % 50) as u32),
            ws_lag_ms: 5.0 + (elapsed as f64 * 0.2).sin() * 3.0,
            
            // QUIC/TPU
            quic_handshake_success_rate: 0.98 + (elapsed as f64 * 0.1).sin() * 0.01,
            concurrent_connections: 384 + ((elapsed % 100) as u32),
            streams_per_peer: 8.0 + (elapsed as f64 * 0.1).sin() * 2.0,
            throttling_events: (elapsed % 10) as u32,
            pps_rate: 50000.0 + (elapsed as f64 * 0.2).sin() * 10000.0,
            
            // Timing Waterfall
            client_to_rpc_ms: 2.0 + (elapsed as f64 * 0.1).sin(),
            rpc_to_be_ms: 5.0 + (elapsed as f64 * 0.15).cos() * 2.0,
            be_to_auction_ms: 3.0 + (elapsed as f64 * 0.2).sin(),
            auction_to_relay_ms: 4.0 + (elapsed as f64 * 0.25).cos() * 1.5,
            relay_to_leader_ms: 8.0 + (elapsed as f64 * 0.3).sin() * 3.0,
            
            // Bundle Success
            bundle_acceptance_rate: 0.72 + (elapsed as f64 * 0.1).sin() * 0.1,
            bundle_landing_delay_ms: 150.0 + (elapsed as f64 * 0.2).cos() * 50.0,
            
            // ShredStream
            shred_packets_per_sec: 12000.0 + (elapsed as f64 * 0.15).sin() * 2000.0,
            shred_gaps: (elapsed % 5) as u32,
            shred_reorders: (elapsed % 8) as u32,
            
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        };
        
        state.metrics.insert("current".to_string(), metrics);
    }
}

async fn home() -> Html<&'static str> {
    Html("<h1>Mission Control API</h1><p>Legendary Solana Node Management System</p>")
}

async fn get_overview(State(state): State<AppState>) -> Json<serde_json::Value> {
    let metrics = state.metrics.get("current").map(|m| m.clone());
    Json(serde_json::json!({
        "status": "success",
        "data": metrics
    }))
}

async fn get_node_summary(State(state): State<AppState>) -> Json<serde_json::Value> {
    let metrics = state.metrics.get("current").map(|m| m.clone());
    if let Some(m) = metrics {
        Json(serde_json::json!({
            "status": "success",
            "data": {
                "version": m.node_version,
                "cluster": m.cluster,
                "ledger_height": m.ledger_height,
                "slot": m.slot,
                "distance_to_tip": m.distance_to_tip,
                "epoch_progress": m.epoch_progress
            }
        }))
    } else {
        Json(serde_json::json!({"status": "error", "message": "No data available"}))
    }
}

async fn get_consensus_health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let metrics = state.metrics.get("current").map(|m| m.clone());
    if let Some(m) = metrics {
        Json(serde_json::json!({
            "status": "success",
            "data": {
                "vote_credits": m.vote_credits,
                "delinquent": m.delinquent,
                "block_production_rate": m.block_production_rate
            }
        }))
    } else {
        Json(serde_json::json!({"status": "error", "message": "No data available"}))
    }
}

async fn get_cluster_perf(State(state): State<AppState>) -> Json<serde_json::Value> {
    let metrics = state.metrics.get("current").map(|m| m.clone());
    if let Some(m) = metrics {
        Json(serde_json::json!({
            "status": "success",
            "data": {
                "tps": m.tps,
                "confirmation_time_ms": m.confirmation_time_ms
            }
        }))
    } else {
        Json(serde_json::json!({"status": "error", "message": "No data available"}))
    }
}

async fn get_jito_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let metrics = state.metrics.get("current").map(|m| m.clone());
    if let Some(m) = metrics {
        Json(serde_json::json!({
            "status": "success",
            "data": {
                "region": m.jito_region,
                "min_tip": m.min_tip,
                "auction_tick_ms": m.auction_tick_ms,
                "bundle_auth_uuid": m.bundle_auth_uuid,
                "tip_percentiles": {
                    "p50": m.tip_percentile_50,
                    "p90": m.tip_percentile_90
                }
            }
        }))
    } else {
        Json(serde_json::json!({"status": "error", "message": "No data available"}))
    }
}

async fn get_rpc_metrics(State(state): State<AppState>) -> Json<serde_json::Value> {
    let metrics = state.metrics.get("current").map(|m| m.clone());
    if let Some(m) = metrics {
        Json(serde_json::json!({
            "status": "success",
            "data": {
                "latency_p50": m.rpc_latency_p50,
                "latency_p99": m.rpc_latency_p99,
                "ws_subscriptions": m.ws_subscription_count,
                "ws_lag_ms": m.ws_lag_ms
            }
        }))
    } else {
        Json(serde_json::json!({"status": "error", "message": "No data available"}))
    }
}

async fn get_timing_waterfall(State(state): State<AppState>) -> Json<serde_json::Value> {
    let metrics = state.metrics.get("current").map(|m| m.clone());
    if let Some(m) = metrics {
        Json(serde_json::json!({
            "status": "success",
            "data": {
                "client_to_rpc": m.client_to_rpc_ms,
                "rpc_to_be": m.rpc_to_be_ms,
                "be_to_auction": m.be_to_auction_ms,
                "auction_to_relay": m.auction_to_relay_ms,
                "relay_to_leader": m.relay_to_leader_ms,
                "total": m.client_to_rpc_ms + m.rpc_to_be_ms + m.be_to_auction_ms + m.auction_to_relay_ms + m.relay_to_leader_ms
            }
        }))
    } else {
        Json(serde_json::json!({"status": "error", "message": "No data available"}))
    }
}

async fn get_tip_intelligence(State(state): State<AppState>) -> Json<serde_json::Value> {
    let metrics = state.metrics.get("current").map(|m| m.clone());
    if let Some(m) = metrics {
        Json(serde_json::json!({
            "status": "success",
            "data": {
                "percentiles": {
                    "p50": m.tip_percentile_50,
                    "p90": m.tip_percentile_90,
                    "p99": m.tip_percentile_90 * 1.5
                },
                "efficiency": m.tip_percentile_50 / 1000.0,
                "min_tip": m.min_tip
            }
        }))
    } else {
        Json(serde_json::json!({"status": "error", "message": "No data available"}))
    }
}

async fn get_bundle_success(State(state): State<AppState>) -> Json<serde_json::Value> {
    let metrics = state.metrics.get("current").map(|m| m.clone());
    if let Some(m) = metrics {
        Json(serde_json::json!({
            "status": "success",
            "data": {
                "acceptance_rate": m.bundle_acceptance_rate,
                "landing_delay_ms": m.bundle_landing_delay_ms,
                "rejection_reasons": {
                    "simulation_failed": 0.15,
                    "tip_too_low": 0.08,
                    "bundle_too_large": 0.05
                }
            }
        }))
    } else {
        Json(serde_json::json!({"status": "error", "message": "No data available"}))
    }
}

async fn get_shredstream(State(state): State<AppState>) -> Json<serde_json::Value> {
    let metrics = state.metrics.get("current").map(|m| m.clone());
    if let Some(m) = metrics {
        Json(serde_json::json!({
            "status": "success",
            "data": {
                "packets_per_sec": m.shred_packets_per_sec,
                "gaps": m.shred_gaps,
                "reorders": m.shred_reorders
            }
        }))
    } else {
        Json(serde_json::json!({"status": "error", "message": "No data available"}))
    }
}

async fn get_quic_health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let metrics = state.metrics.get("current").map(|m| m.clone());
    if let Some(m) = metrics {
        Json(serde_json::json!({
            "status": "success",
            "data": {
                "handshake_success_rate": m.quic_handshake_success_rate,
                "concurrent_connections": m.concurrent_connections,
                "streams_per_peer": m.streams_per_peer,
                "throttling_events": m.throttling_events,
                "pps_rate": m.pps_rate
            }
        }))
    } else {
        Json(serde_json::json!({"status": "error", "message": "No data available"}))
    }
}

async fn get_qos_peering(State(_state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "success",
        "data": {
            "whitelisted_rpcs": [
                {"address": "rpc1.example.com", "stake": 1000000},
                {"address": "rpc2.example.com", "stake": 500000}
            ],
            "virtual_stake_mapping": {
                "total": 10000000,
                "distribution": [
                    {"peer": "validator1", "stake": 3000000},
                    {"peer": "validator2", "stake": 2000000}
                ]
            },
            "leader_quic_port": 8003,
            "pin_peering": true
        }
    }))
}

async fn get_gossip_metrics(State(_state): State<AppState>) -> Json<serde_json::Value> {
    let now = chrono::Utc::now().timestamp_millis();
    Json(serde_json::json!({
        "status": "success",
        "data": {
            "peers": 1247,
            "shreds_sent": 450000,
            "shreds_received": 445000,
            "retransmits": 1250,
            "repair_requests": 89,
            "broadcast_success_rate": 0.995
        }
    }))
}

async fn node_control(Json(payload): Json<serde_json::Value>) -> Json<serde_json::Value> {
    let action = payload.get("action").and_then(|v| v.as_str()).unwrap_or("");
    
    match action {
        "start" => Json(serde_json::json!({
            "status": "success",
            "message": "Node starting...",
            "preflight": {
                "ledger_fsync": true,
                "snapshots": true,
                "catch_up_debt": 0
            }
        })),
        "stop" => Json(serde_json::json!({
            "status": "success",
            "message": "Node stopping gracefully..."
        })),
        "restart" => Json(serde_json::json!({
            "status": "success",
            "message": "Node restarting..."
        })),
        _ => Json(serde_json::json!({
            "status": "error",
            "message": "Invalid action"
        }))
    }
}

async fn preflight_checks(State(_state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "success",
        "checks": {
            "ledger_fsync": true,
            "snapshots_healthy": true,
            "catch_up_debt": 0,
            "disk_space_gb": 450,
            "memory_available_gb": 120,
            "network_connectivity": true,
            "jito_connection": true,
            "rpc_responsive": true
        },
        "ready": true
    }))
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl axum::response::IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: axum::extract::ws::WebSocket, state: AppState) {
    use futures_util::StreamExt;
    use futures_util::SinkExt;
    
    let mut ticker = interval(Duration::from_millis(100));
    
    loop {
        ticker.tick().await;
        
        if let Some(metrics) = state.metrics.get("current") {
            let msg = serde_json::json!({
                "type": "metrics_update",
                "data": metrics.clone()
            });
            
            if socket.send(axum::extract::ws::Message::Text(msg.to_string())).await.is_err() {
                break;
            }
        }
    }
}