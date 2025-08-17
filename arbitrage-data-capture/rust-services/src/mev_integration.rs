/// MEV Backend Integration Module
/// Connects Rust services with FastAPI backend for ultra-low-latency operations

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use reqwest::Client;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{StreamExt, SinkExt};
use tracing::{info, warn, error, debug};
use dashmap::DashMap;

const API_BASE: &str = "http://localhost:8000";
const WS_BASE: &str = "ws://localhost:8000";

/// MEV opportunity from backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MEVOpportunity {
    pub id: String,
    #[serde(rename = "type")]
    pub opportunity_type: String,
    pub profit_estimate: f64,
    pub confidence: f64,
    pub gas_estimate: f64,
    pub deadline_ms: i64,
    pub route: Vec<RouteStep>,
    pub risk_score: f64,
    pub dna_fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteStep {
    pub from: String,
    pub to: String,
    pub pool: String,
    pub amount: Option<u64>,
}

/// Execution request
#[derive(Debug, Serialize)]
pub struct ExecutionRequest {
    pub opportunity_id: String,
    pub max_slippage: f64,
    pub priority_fee: f64,
    pub use_jito: bool,
}

/// Execution result
#[derive(Debug, Deserialize)]
pub struct ExecutionResult {
    pub id: String,
    pub opportunity_id: String,
    pub status: String,
    pub profit_actual: Option<f64>,
    pub gas_used: Option<f64>,
    pub latency_ms: f64,
    pub strategy: String,
}

/// Bundle submission request
#[derive(Debug, Serialize)]
pub struct BundleSubmitRequest {
    pub transactions: Vec<String>,
    pub tip_lamports: u64,
    pub region: String,
}

/// MEV backend client with connection pooling
pub struct MEVBackendClient {
    http_client: Client,
    api_base: String,
    auth_token: Option<String>,
    opportunity_cache: Arc<DashMap<String, MEVOpportunity>>,
    metrics: Arc<RwLock<ClientMetrics>>,
}

#[derive(Debug, Default)]
struct ClientMetrics {
    requests_sent: u64,
    requests_successful: u64,
    requests_failed: u64,
    total_latency_us: u64,
    p50_latency_us: u64,
    p99_latency_us: u64,
}

impl MEVBackendClient {
    /// Create new MEV backend client
    pub fn new(api_base: Option<String>, auth_token: Option<String>) -> Self {
        let http_client = Client::builder()
            .pool_idle_timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(100)
            .timeout(Duration::from_millis(5000))
            .tcp_nodelay(true)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            http_client,
            api_base: api_base.unwrap_or_else(|| API_BASE.to_string()),
            auth_token,
            opportunity_cache: Arc::new(DashMap::new()),
            metrics: Arc::new(RwLock::new(ClientMetrics::default())),
        }
    }

    /// Scan for MEV opportunities
    pub async fn scan_opportunities(
        &self,
        scan_type: &str,
        min_profit: f64,
    ) -> Result<Vec<MEVOpportunity>, Box<dyn std::error::Error>> {
        let start = Instant::now();
        
        let url = format!("{}/api/mev/scan", self.api_base);
        let mut request = self.http_client
            .post(&url)
            .json(&serde_json::json!({
                "scan_type": scan_type,
                "min_profit": min_profit,
                "max_gas_price": 0.01,
                "include_pending": true
            }));

        if let Some(token) = &self.auth_token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await?;
        let latency_us = start.elapsed().as_micros() as u64;
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.requests_sent += 1;
            metrics.total_latency_us += latency_us;
            
            if response.status().is_success() {
                metrics.requests_successful += 1;
            } else {
                metrics.requests_failed += 1;
            }
        }

        let data: serde_json::Value = response.json().await?;
        let opportunities: Vec<MEVOpportunity> = serde_json::from_value(
            data["opportunities"].clone()
        )?;

        // Cache opportunities
        for opp in &opportunities {
            self.opportunity_cache.insert(opp.id.clone(), opp.clone());
        }

        debug!("Scanned {} opportunities in {}us", opportunities.len(), latency_us);
        Ok(opportunities)
    }

    /// Execute MEV opportunity
    pub async fn execute_opportunity(
        &self,
        opportunity_id: &str,
        max_slippage: f64,
    ) -> Result<ExecutionResult, Box<dyn std::error::Error>> {
        let start = Instant::now();
        
        let url = format!("{}/api/mev/execute/{}", self.api_base, opportunity_id);
        let mut request = self.http_client
            .post(&url)
            .json(&ExecutionRequest {
                opportunity_id: opportunity_id.to_string(),
                max_slippage,
                priority_fee: 0.001,
                use_jito: true,
            });

        if let Some(token) = &self.auth_token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await?;
        let latency_us = start.elapsed().as_micros() as u64;
        
        let result: ExecutionResult = response.json().await?;
        
        info!(
            "Executed opportunity {} with status: {} in {}us",
            opportunity_id, result.status, latency_us
        );
        
        Ok(result)
    }

    /// Submit bundle to Jito
    pub async fn submit_bundle(
        &self,
        transactions: Vec<String>,
        tip_lamports: u64,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let start = Instant::now();
        
        let url = format!("{}/api/mev/bundle/submit", self.api_base);
        let mut request = self.http_client
            .post(&url)
            .json(&BundleSubmitRequest {
                transactions,
                tip_lamports,
                region: "amsterdam".to_string(),
            });

        if let Some(token) = &self.auth_token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await?;
        let latency_us = start.elapsed().as_micros() as u64;
        
        let result: serde_json::Value = response.json().await?;
        
        info!("Bundle submitted in {}us: {:?}", latency_us, result);
        
        Ok(result)
    }

    /// Get current stats
    pub async fn get_stats(&self) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let url = format!("{}/api/mev/stats", self.api_base);
        let mut request = self.http_client.get(&url);

        if let Some(token) = &self.auth_token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await?;
        Ok(response.json().await?)
    }

    /// Get client metrics
    pub async fn get_metrics(&self) -> ClientMetrics {
        self.metrics.read().await.clone()
    }
}

/// WebSocket client for real-time streams
pub struct MEVWebSocketClient {
    ws_base: String,
    auth_token: Option<String>,
}

impl MEVWebSocketClient {
    pub fn new(ws_base: Option<String>, auth_token: Option<String>) -> Self {
        Self {
            ws_base: ws_base.unwrap_or_else(|| WS_BASE.to_string()),
            auth_token,
        }
    }

    /// Connect to opportunities WebSocket stream
    pub async fn connect_opportunities_stream<F>(
        &self,
        mut handler: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(MEVOpportunity) + Send + 'static,
    {
        let url = format!("{}/api/mev/ws/opportunities", self.ws_base);
        let (ws_stream, _) = connect_async(&url).await?;
        let (_, mut read) = ws_stream.split();

        info!("Connected to opportunities WebSocket stream");

        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(data) = serde_json::from_str::<serde_json::Value>(&text) {
                        if let Ok(opportunities) = serde_json::from_value::<Vec<MEVOpportunity>>(
                            data["data"].clone()
                        ) {
                            for opp in opportunities {
                                handler(opp);
                            }
                        }
                    }
                }
                Ok(Message::Close(_)) => {
                    info!("WebSocket connection closed");
                    break;
                }
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Connect to metrics WebSocket stream
    pub async fn connect_metrics_stream<F>(
        &self,
        mut handler: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(serde_json::Value) + Send + 'static,
    {
        let url = format!("{}/api/mev/ws/metrics", self.ws_base);
        let (ws_stream, _) = connect_async(&url).await?;
        let (_, mut read) = ws_stream.split();

        info!("Connected to metrics WebSocket stream");

        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(data) = serde_json::from_str::<serde_json::Value>(&text) {
                        handler(data);
                    }
                }
                Ok(Message::Close(_)) => break,
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// High-performance opportunity processor
pub struct OpportunityProcessor {
    client: Arc<MEVBackendClient>,
    min_profit_threshold: f64,
    max_concurrent_executions: usize,
}

impl OpportunityProcessor {
    pub fn new(client: Arc<MEVBackendClient>, min_profit_threshold: f64) -> Self {
        Self {
            client,
            min_profit_threshold,
            max_concurrent_executions: 10,
        }
    }

    /// Process opportunities in parallel
    pub async fn process_opportunities(&self, opportunities: Vec<MEVOpportunity>) {
        let mut handles = vec![];
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.max_concurrent_executions));

        for opp in opportunities {
            if opp.profit_estimate < self.min_profit_threshold {
                continue;
            }

            let client = self.client.clone();
            let sem = semaphore.clone();
            
            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                
                match client.execute_opportunity(&opp.id, 0.01).await {
                    Ok(result) => {
                        info!("Executed {}: {:?}", opp.id, result.status);
                    }
                    Err(e) => {
                        error!("Failed to execute {}: {}", opp.id, e);
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all executions to complete
        for handle in handles {
            let _ = handle.await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mev_client() {
        let client = MEVBackendClient::new(None, None);
        
        // Test scanning
        match client.scan_opportunities("all", 0.1).await {
            Ok(opportunities) => {
                println!("Found {} opportunities", opportunities.len());
                for opp in opportunities.iter().take(3) {
                    println!("Opportunity: {} - Profit: {} SOL", opp.id, opp.profit_estimate);
                }
            }
            Err(e) => {
                println!("Failed to scan opportunities: {}", e);
            }
        }

        // Test stats
        match client.get_stats().await {
            Ok(stats) => {
                println!("Stats: {}", serde_json::to_string_pretty(&stats).unwrap());
            }
            Err(e) => {
                println!("Failed to get stats: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_websocket_client() {
        let ws_client = MEVWebSocketClient::new(None, None);
        
        // Test opportunities stream
        let handle = tokio::spawn(async move {
            let _ = ws_client.connect_opportunities_stream(|opp| {
                println!("New opportunity: {} - {}", opp.id, opp.profit_estimate);
            }).await;
        });

        // Let it run for a bit
        tokio::time::sleep(Duration::from_secs(5)).await;
        handle.abort();
    }
}