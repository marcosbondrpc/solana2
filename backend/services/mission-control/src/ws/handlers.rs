use crate::cache::CacheManager;
use crate::models::*;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
};
use futures::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};
use tracing::{error, info, warn};

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WsMessage {
    Subscribe { channel: String },
    Unsubscribe { channel: String },
    Ping,
    Pong,
    Data { channel: String, payload: serde_json::Value },
    Error { message: String },
}

#[derive(Clone)]
pub struct WsState {
    pub cache: Arc<CacheManager>,
}

// WebSocket upgrade handler for /ws/mission-control
pub async fn ws_mission_control(
    ws: WebSocketUpgrade,
    State(state): State<Arc<WsState>>,
) -> Response {
    ws.on_upgrade(move |socket| handle_mission_control_socket(socket, state))
}

// WebSocket upgrade handler for /ws/jito-tips
pub async fn ws_jito_tips(
    ws: WebSocketUpgrade,
    State(state): State<Arc<WsState>>,
) -> Response {
    ws.on_upgrade(move |socket| handle_jito_tips_socket(socket, state))
}

// WebSocket upgrade handler for /ws/bundle-status
pub async fn ws_bundle_status(
    ws: WebSocketUpgrade,
    State(state): State<Arc<WsState>>,
) -> Response {
    ws.on_upgrade(move |socket| handle_bundle_status_socket(socket, state))
}

async fn handle_mission_control_socket(socket: WebSocket, state: Arc<WsState>) {
    let (mut sender, mut receiver) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<WsMessage>();
    
    // Spawn task to send messages to client
    let mut send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            let json = serde_json::to_string(&msg).unwrap();
            if sender.send(Message::Text(json)).await.is_err() {
                break;
            }
        }
    });
    
    // Spawn task to stream real-time metrics
    let tx_clone = tx.clone();
    let state_clone = state.clone();
    let mut metrics_task = tokio::spawn(async move {
        let mut ticker = interval(Duration::from_secs(1));
        
        loop {
            ticker.tick().await;
            
            // Fetch latest metrics from cache
            if let Ok(metrics) = state_clone.cache.get_all_cached_metrics().await {
                let payload = serde_json::json!({
                    "node_summary": metrics.node_summary,
                    "consensus_health": metrics.consensus_health,
                    "cluster_performance": metrics.cluster_performance,
                    "jito_status": metrics.jito_status,
                    "rpc_metrics": metrics.rpc_metrics,
                    "quic_health": metrics.quic_health,
                    "qos_peering": metrics.qos_peering,
                    "timestamp": chrono::Utc::now(),
                });
                
                let msg = WsMessage::Data {
                    channel: "mission_control".to_string(),
                    payload,
                };
                
                if tx_clone.send(msg).is_err() {
                    break;
                }
            }
        }
    });
    
    // Handle incoming messages
    let tx_clone = tx.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    if let Ok(ws_msg) = serde_json::from_str::<WsMessage>(&text) {
                        match ws_msg {
                            WsMessage::Ping => {
                                let _ = tx_clone.send(WsMessage::Pong);
                            }
                            WsMessage::Subscribe { channel } => {
                                info!("Client subscribed to channel: {}", channel);
                            }
                            WsMessage::Unsubscribe { channel } => {
                                info!("Client unsubscribed from channel: {}", channel);
                            }
                            _ => {}
                        }
                    }
                }
                Message::Close(_) => {
                    info!("WebSocket connection closed");
                    break;
                }
                _ => {}
            }
        }
    });
    
    // Wait for any task to complete
    tokio::select! {
        _ = &mut send_task => {
            recv_task.abort();
            metrics_task.abort();
        }
        _ = &mut recv_task => {
            send_task.abort();
            metrics_task.abort();
        }
        _ = &mut metrics_task => {
            send_task.abort();
            recv_task.abort();
        }
    }
    
    info!("WebSocket connection terminated");
}

async fn handle_jito_tips_socket(socket: WebSocket, state: Arc<WsState>) {
    let (mut sender, mut receiver) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<WsMessage>();
    
    // Spawn task to send messages to client
    let mut send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            let json = serde_json::to_string(&msg).unwrap();
            if sender.send(Message::Text(json)).await.is_err() {
                break;
            }
        }
    });
    
    // Spawn task to stream Jito tips
    let tx_clone = tx.clone();
    let state_clone = state.clone();
    let mut tips_task = tokio::spawn(async move {
        let mut ticker = interval(Duration::from_millis(500)); // Higher frequency for tips
        
        loop {
            ticker.tick().await;
            
            // In production, would get real-time tips from Jito collector
            let tip_data = serde_json::json!({
                "slot": 250_000_000u64 + rand::random::<u64>() % 100,
                "timestamp": chrono::Utc::now(),
                "tip_lamports": 100_000u64 + rand::random::<u64>() % 1_000_000,
                "tipper": format!("Tipper{}", rand::random::<u8>()),
                "bundle_id": format!("bundle_{}", rand::random::<u32>()),
                "region": ["frankfurt", "amsterdam", "tokyo", "ny"][rand::random::<usize>() % 4],
            });
            
            let msg = WsMessage::Data {
                channel: "jito_tips".to_string(),
                payload: tip_data,
            };
            
            if tx_clone.send(msg).is_err() {
                break;
            }
        }
    });
    
    // Handle incoming messages
    let tx_clone = tx.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    if let Ok(ws_msg) = serde_json::from_str::<WsMessage>(&text) {
                        match ws_msg {
                            WsMessage::Ping => {
                                let _ = tx_clone.send(WsMessage::Pong);
                            }
                            _ => {}
                        }
                    }
                }
                Message::Close(_) => {
                    info!("Jito tips WebSocket closed");
                    break;
                }
                _ => {}
            }
        }
    });
    
    // Wait for any task to complete
    tokio::select! {
        _ = &mut send_task => {
            recv_task.abort();
            tips_task.abort();
        }
        _ = &mut recv_task => {
            send_task.abort();
            tips_task.abort();
        }
        _ = &mut tips_task => {
            send_task.abort();
            recv_task.abort();
        }
    }
}

async fn handle_bundle_status_socket(socket: WebSocket, state: Arc<WsState>) {
    let (mut sender, mut receiver) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<WsMessage>();
    
    // Spawn task to send messages to client
    let mut send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            let json = serde_json::to_string(&msg).unwrap();
            if sender.send(Message::Text(json)).await.is_err() {
                break;
            }
        }
    });
    
    // Spawn task to stream bundle status updates
    let tx_clone = tx.clone();
    let state_clone = state.clone();
    let mut bundle_task = tokio::spawn(async move {
        let mut ticker = interval(Duration::from_secs(2));
        
        loop {
            ticker.tick().await;
            
            // In production, would get real bundle status from Jito collector
            let statuses = ["pending", "simulated", "accepted", "rejected", "landed"];
            let rejection_reasons = [
                "simulation_failed",
                "stale_blockhash",
                "duplicate_transaction",
                "insufficient_tip",
            ];
            
            let status = statuses[rand::random::<usize>() % statuses.len()];
            let mut bundle_data = serde_json::json!({
                "bundle_id": format!("bundle_{}", rand::random::<u32>()),
                "status": status,
                "timestamp": chrono::Utc::now(),
                "simulation_time_ms": if status != "pending" { Some(10.0 + rand::random::<f64>() * 20.0) } else { None },
            });
            
            if status == "rejected" {
                bundle_data["rejection_reason"] = serde_json::json!(
                    rejection_reasons[rand::random::<usize>() % rejection_reasons.len()]
                );
            }
            
            if status == "landed" {
                bundle_data["landed_slot"] = serde_json::json!(250_000_000u64 + rand::random::<u64>() % 100);
            }
            
            let msg = WsMessage::Data {
                channel: "bundle_status".to_string(),
                payload: bundle_data,
            };
            
            if tx_clone.send(msg).is_err() {
                break;
            }
        }
    });
    
    // Handle incoming messages
    let tx_clone = tx.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    if let Ok(ws_msg) = serde_json::from_str::<WsMessage>(&text) {
                        match ws_msg {
                            WsMessage::Ping => {
                                let _ = tx_clone.send(WsMessage::Pong);
                            }
                            _ => {}
                        }
                    }
                }
                Message::Close(_) => {
                    info!("Bundle status WebSocket closed");
                    break;
                }
                _ => {}
            }
        }
    });
    
    // Wait for any task to complete
    tokio::select! {
        _ = &mut send_task => {
            recv_task.abort();
            bundle_task.abort();
        }
        _ = &mut recv_task => {
            send_task.abort();
            bundle_task.abort();
        }
        _ = &mut bundle_task => {
            send_task.abort();
            recv_task.abort();
        }
    }
}

// Add rand for simulation
use rand;