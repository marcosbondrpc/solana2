use std::sync::Arc;
use std::net::SocketAddr;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, tungstenite::Message};
use futures_util::{StreamExt, SinkExt};
use tokio::sync::broadcast;
use tokio::time::{self, Duration};
use anyhow::Result;
use tracing::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use serde_json;

use crate::metrics_collector::{MetricsCollector, NodeMetrics, ConnectionHealth};
use crate::cache_manager::CacheManager;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WSMessage {
    Subscribe {
        channels: Vec<String>,
    },
    Unsubscribe {
        channels: Vec<String>,
    },
    Ping,
    Pong,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WSResponse {
    Metrics(NodeMetrics),
    Health(Vec<ConnectionHealth>),
    LatencyUpdate {
        endpoint_type: String,
        endpoint: String,
        latency_us: u64,
    },
    Error {
        message: String,
    },
    Pong,
}

pub struct WebSocketServer {
    port: u16,
    metrics_collector: Arc<MetricsCollector>,
    cache: Arc<CacheManager>,
    broadcast_tx: broadcast::Sender<WSResponse>,
}

impl WebSocketServer {
    pub fn new(
        port: u16,
        metrics_collector: Arc<MetricsCollector>,
        cache: Arc<CacheManager>,
    ) -> Self {
        let (broadcast_tx, _) = broadcast::channel(1024);
        
        Self {
            port,
            metrics_collector,
            cache,
            broadcast_tx,
        }
    }
    
    pub async fn start(self) -> Result<()> {
        let addr = format!("0.0.0.0:{}", self.port);
        let listener = TcpListener::bind(&addr).await?;
        
        info!("WebSocket server listening on ws://{}", addr);
        
        // Start metrics broadcast task
        let broadcast_tx = self.broadcast_tx.clone();
        let metrics_collector = self.metrics_collector.clone();
        tokio::spawn(async move {
            Self::broadcast_metrics(broadcast_tx, metrics_collector).await;
        });
        
        // Accept connections
        while let Ok((stream, addr)) = listener.accept().await {
            tokio::spawn(self.clone().handle_connection(stream, addr));
        }
        
        Ok(())
    }
    
    async fn broadcast_metrics(
        tx: broadcast::Sender<WSResponse>,
        metrics_collector: Arc<MetricsCollector>,
    ) {
        let mut interval = time::interval(Duration::from_millis(100));
        interval.set_missed_tick_behavior(time::MissedTickBehavior::Skip);
        
        loop {
            interval.tick().await;
            
            // Get current metrics
            let metrics = metrics_collector.get_current_metrics().await;
            
            // Broadcast to all connected clients
            let _ = tx.send(WSResponse::Metrics(metrics));
            
            // Also broadcast connection health
            let health = metrics_collector.get_connection_health().await;
            let _ = tx.send(WSResponse::Health(health));
        }
    }
    
    async fn handle_connection(self, stream: TcpStream, addr: SocketAddr) {
        debug!("New WebSocket connection from {}", addr);
        
        // Set TCP options for low latency
        if let Ok(stream_ref) = stream.try_clone() {
            let _ = stream_ref.set_nodelay(true);
        }
        
        let ws_stream = match accept_async(stream).await {
            Ok(ws) => ws,
            Err(e) => {
                error!("WebSocket handshake failed for {}: {}", addr, e);
                return;
            }
        };
        
        let (mut ws_sender, mut ws_receiver) = ws_stream.split();
        let mut broadcast_rx = self.broadcast_tx.subscribe();
        
        // Spawn task to handle incoming messages
        let addr_clone = addr.clone();
        let incoming_task = tokio::spawn(async move {
            while let Some(msg) = ws_receiver.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        if let Ok(ws_msg) = serde_json::from_str::<WSMessage>(&text) {
                            match ws_msg {
                                WSMessage::Ping => {
                                    debug!("Received ping from {}", addr_clone);
                                    // Pong will be sent in the outgoing task
                                }
                                WSMessage::Subscribe { channels } => {
                                    debug!("Client {} subscribing to channels: {:?}", addr_clone, channels);
                                }
                                WSMessage::Unsubscribe { channels } => {
                                    debug!("Client {} unsubscribing from channels: {:?}", addr_clone, channels);
                                }
                                _ => {}
                            }
                        }
                    }
                    Ok(Message::Close(_)) => {
                        debug!("Client {} disconnected", addr_clone);
                        break;
                    }
                    Ok(Message::Ping(data)) => {
                        debug!("Received WebSocket ping from {}", addr_clone);
                    }
                    Err(e) => {
                        error!("WebSocket error for {}: {}", addr_clone, e);
                        break;
                    }
                    _ => {}
                }
            }
        });
        
        // Spawn task to send broadcast messages to client
        let outgoing_task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    // Receive broadcast messages
                    Ok(msg) = broadcast_rx.recv() => {
                        let json = match serde_json::to_string(&msg) {
                            Ok(json) => json,
                            Err(e) => {
                                error!("Failed to serialize message: {}", e);
                                continue;
                            }
                        };
                        
                        if let Err(e) = ws_sender.send(Message::Text(json)).await {
                            error!("Failed to send message to {}: {}", addr, e);
                            break;
                        }
                    }
                    
                    // Send periodic pings to keep connection alive
                    _ = tokio::time::sleep(Duration::from_secs(30)) => {
                        if let Err(e) = ws_sender.send(Message::Ping(vec![])).await {
                            error!("Failed to send ping to {}: {}", addr, e);
                            break;
                        }
                    }
                }
            }
        });
        
        // Wait for either task to complete
        tokio::select! {
            _ = incoming_task => {
                debug!("Incoming task completed for {}", addr);
            }
            _ = outgoing_task => {
                debug!("Outgoing task completed for {}", addr);
            }
        }
        
        info!("WebSocket connection closed for {}", addr);
    }
}

impl Clone for WebSocketServer {
    fn clone(&self) -> Self {
        Self {
            port: self.port,
            metrics_collector: self.metrics_collector.clone(),
            cache: self.cache.clone(),
            broadcast_tx: self.broadcast_tx.clone(),
        }
    }
}