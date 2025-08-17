use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time;
use tracing::{debug, error, info, warn};
use anyhow::Result;
use failsafe::{CircuitBreaker, Config as CBConfig, Error as CBError};
use solana_client::nonblocking::rpc_client::RpcClient;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{StreamExt, SinkExt};

use crate::config::Config;
use crate::metrics_collector::{MetricsCollector, EndpointType};

pub struct HealthChecker {
    config: Config,
    metrics_collector: Arc<MetricsCollector>,
    rpc_breakers: Vec<CircuitBreaker>,
    ws_breakers: Vec<CircuitBreaker>,
    geyser_breakers: Vec<CircuitBreaker>,
    jito_breakers: Vec<CircuitBreaker>,
}

impl HealthChecker {
    pub fn new(config: Config, metrics_collector: Arc<MetricsCollector>) -> Self {
        // Create circuit breakers for each endpoint
        let breaker_config = CBConfig::new()
            .failure_rate_threshold(0.5)
            .consecutive_failures(config.circuit_breaker_threshold)
            .half_open_delay(Duration::from_millis(config.circuit_breaker_timeout_ms));
        
        let rpc_breakers = (0..config.rpc_endpoints.len())
            .map(|_| CircuitBreaker::with_config(breaker_config.clone()))
            .collect();
        
        let ws_breakers = (0..config.ws_endpoints.len())
            .map(|_| CircuitBreaker::with_config(breaker_config.clone()))
            .collect();
        
        let geyser_breakers = (0..config.geyser_endpoints.len())
            .map(|_| CircuitBreaker::with_config(breaker_config.clone()))
            .collect();
        
        let jito_breakers = (0..config.jito_endpoints.len())
            .map(|_| CircuitBreaker::with_config(breaker_config.clone()))
            .collect();
        
        Self {
            config,
            metrics_collector,
            rpc_breakers,
            ws_breakers,
            geyser_breakers,
            jito_breakers,
        }
    }
    
    pub async fn start_monitoring(&self) {
        info!("Starting health monitoring with {}ms interval", self.config.health_check_interval_ms);
        
        let mut interval = time::interval(Duration::from_millis(self.config.health_check_interval_ms));
        interval.set_missed_tick_behavior(time::MissedTickBehavior::Skip);
        
        loop {
            interval.tick().await;
            
            // Check all endpoints concurrently
            let mut tasks = Vec::new();
            
            // Check RPC endpoints
            for (idx, endpoint) in self.config.rpc_endpoints.iter().enumerate() {
                let endpoint = endpoint.clone();
                let breaker = self.rpc_breakers[idx].clone();
                let metrics = self.metrics_collector.clone();
                
                tasks.push(tokio::spawn(async move {
                    check_rpc_health(endpoint, breaker, metrics).await
                }));
            }
            
            // Check WebSocket endpoints
            for (idx, endpoint) in self.config.ws_endpoints.iter().enumerate() {
                let endpoint = endpoint.clone();
                let breaker = self.ws_breakers[idx].clone();
                let metrics = self.metrics_collector.clone();
                
                tasks.push(tokio::spawn(async move {
                    check_ws_health(endpoint, breaker, metrics).await
                }));
            }
            
            // Check Geyser endpoints
            for (idx, endpoint) in self.config.geyser_endpoints.iter().enumerate() {
                let endpoint = endpoint.clone();
                let breaker = self.geyser_breakers[idx].clone();
                let metrics = self.metrics_collector.clone();
                
                tasks.push(tokio::spawn(async move {
                    check_geyser_health(endpoint, breaker, metrics).await
                }));
            }
            
            // Check Jito endpoints
            for (idx, endpoint) in self.config.jito_endpoints.iter().enumerate() {
                let endpoint = endpoint.clone();
                let breaker = self.jito_breakers[idx].clone();
                let metrics = self.metrics_collector.clone();
                
                tasks.push(tokio::spawn(async move {
                    check_jito_health(endpoint, breaker, metrics).await
                }));
            }
            
            // Wait for all health checks to complete
            for task in tasks {
                if let Err(e) = task.await {
                    error!("Health check task failed: {}", e);
                }
            }
        }
    }
}

async fn check_rpc_health(
    endpoint: String,
    breaker: CircuitBreaker,
    metrics: Arc<MetricsCollector>,
) -> Result<()> {
    let start = Instant::now();
    
    let result = breaker.call(async {
        let client = RpcClient::new(endpoint.clone());
        
        // Perform health check with timeout
        tokio::time::timeout(
            Duration::from_millis(1000),
            client.get_health()
        ).await
        .map_err(|_| CBError::Inner("Timeout".to_string()))?
        .map_err(|e| CBError::Inner(e.to_string()))
    }).await;
    
    let latency_us = start.elapsed().as_micros() as u64;
    let is_healthy = result.is_ok();
    
    metrics.update_connection_health(
        endpoint.clone(),
        EndpointType::RPC,
        is_healthy,
        latency_us,
    ).await;
    
    if !is_healthy {
        warn!("RPC endpoint {} is unhealthy: {:?}", endpoint, result);
    }
    
    Ok(())
}

async fn check_ws_health(
    endpoint: String,
    breaker: CircuitBreaker,
    metrics: Arc<MetricsCollector>,
) -> Result<()> {
    let start = Instant::now();
    
    let result = breaker.call(async {
        // Connect to WebSocket
        let (ws_stream, _) = connect_async(&endpoint)
            .await
            .map_err(|e| CBError::Inner(e.to_string()))?;
        
        let (mut write, mut read) = ws_stream.split();
        
        // Send subscription request
        let subscribe_msg = r#"{"jsonrpc":"2.0","id":1,"method":"slotSubscribe"}"#;
        write.send(Message::Text(subscribe_msg.to_string()))
            .await
            .map_err(|e| CBError::Inner(e.to_string()))?;
        
        // Wait for response with timeout
        tokio::time::timeout(
            Duration::from_millis(1000),
            read.next()
        ).await
        .map_err(|_| CBError::Inner("Timeout".to_string()))?
        .ok_or_else(|| CBError::Inner("No response".to_string()))?
        .map_err(|e| CBError::Inner(e.to_string()))?;
        
        // Close connection
        let _ = write.send(Message::Close(None)).await;
        
        Ok::<(), CBError>(())
    }).await;
    
    let latency_us = start.elapsed().as_micros() as u64;
    let is_healthy = result.is_ok();
    
    metrics.update_connection_health(
        endpoint.clone(),
        EndpointType::WebSocket,
        is_healthy,
        latency_us,
    ).await;
    
    if !is_healthy {
        warn!("WebSocket endpoint {} is unhealthy: {:?}", endpoint, result);
    }
    
    Ok(())
}

async fn check_geyser_health(
    endpoint: String,
    breaker: CircuitBreaker,
    metrics: Arc<MetricsCollector>,
) -> Result<()> {
    let start = Instant::now();
    
    // For Geyser gRPC endpoints, we'll do a simple TCP connect check
    // In production, you'd want to make an actual gRPC health check call
    let result = breaker.call(async {
        let addr = endpoint
            .strip_prefix("grpc://")
            .unwrap_or(&endpoint)
            .to_string();
        
        tokio::time::timeout(
            Duration::from_millis(1000),
            tokio::net::TcpStream::connect(&addr)
        ).await
        .map_err(|_| CBError::Inner("Timeout".to_string()))?
        .map_err(|e| CBError::Inner(e.to_string()))?;
        
        Ok::<(), CBError>(())
    }).await;
    
    let latency_us = start.elapsed().as_micros() as u64;
    let is_healthy = result.is_ok();
    
    metrics.update_connection_health(
        endpoint.clone(),
        EndpointType::Geyser,
        is_healthy,
        latency_us,
    ).await;
    
    if !is_healthy {
        warn!("Geyser endpoint {} is unhealthy: {:?}", endpoint, result);
    }
    
    Ok(())
}

async fn check_jito_health(
    endpoint: String,
    breaker: CircuitBreaker,
    metrics: Arc<MetricsCollector>,
) -> Result<()> {
    let start = Instant::now();
    
    // For Jito gRPC endpoints, similar TCP health check
    // In production, implement proper Jito block engine health check
    let result = breaker.call(async {
        let addr = endpoint
            .strip_prefix("grpc://")
            .unwrap_or(&endpoint)
            .to_string();
        
        tokio::time::timeout(
            Duration::from_millis(1000),
            tokio::net::TcpStream::connect(&addr)
        ).await
        .map_err(|_| CBError::Inner("Timeout".to_string()))?
        .map_err(|e| CBError::Inner(e.to_string()))?;
        
        Ok::<(), CBError>(())
    }).await;
    
    let latency_us = start.elapsed().as_micros() as u64;
    let is_healthy = result.is_ok();
    
    metrics.update_connection_health(
        endpoint.clone(),
        EndpointType::Jito,
        is_healthy,
        latency_us,
    ).await;
    
    if !is_healthy {
        warn!("Jito endpoint {} is unhealthy: {:?}", endpoint, result);
    }
    
    Ok(())
}