use crate::error::{MissionControlError, Result};
use crate::metrics::MetricsRecorder;
use dashmap::DashMap;
use parking_lot::RwLock;
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_rpc_client_api::config::RpcBlockProductionConfig;
use solana_sdk::commitment_config::CommitmentConfig;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::interval;
use tracing::{error, info, warn};

const RPC_TIMEOUT: Duration = Duration::from_secs(5);
const COLLECTION_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Clone)]
pub struct RpcCollector {
    clients: Arc<Vec<Arc<RpcClient>>>,
    metrics_recorder: Arc<MetricsRecorder>,
    method_latencies: Arc<DashMap<String, Vec<f64>>>,
    circuit_breakers: Arc<DashMap<String, CircuitBreaker>>,
}

#[derive(Clone)]
struct CircuitBreaker {
    failures: Arc<RwLock<u32>>,
    last_failure: Arc<RwLock<Option<Instant>>>,
    state: Arc<RwLock<CircuitState>>,
}

#[derive(Clone, Copy, PartialEq)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitBreaker {
    fn new() -> Self {
        Self {
            failures: Arc::new(RwLock::new(0)),
            last_failure: Arc::new(RwLock::new(None)),
            state: Arc::new(RwLock::new(CircuitState::Closed)),
        }
    }

    fn record_success(&self) {
        let mut failures = self.failures.write();
        *failures = 0;
        *self.state.write() = CircuitState::Closed;
    }

    fn record_failure(&self) {
        let mut failures = self.failures.write();
        *failures += 1;
        *self.last_failure.write() = Some(Instant::now());
        
        if *failures >= 5 {
            *self.state.write() = CircuitState::Open;
        }
    }

    fn can_attempt(&self) -> bool {
        let state = *self.state.read();
        match state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                if let Some(last) = *self.last_failure.read() {
                    if last.elapsed() > Duration::from_secs(30) {
                        *self.state.write() = CircuitState::HalfOpen;
                        true
                    } else {
                        false
                    }
                } else {
                    true
                }
            }
            CircuitState::HalfOpen => true,
        }
    }
}

impl RpcCollector {
    pub fn new(rpc_endpoints: Vec<String>, metrics_recorder: Arc<MetricsRecorder>) -> Result<Self> {
        let clients: Vec<Arc<RpcClient>> = rpc_endpoints
            .into_iter()
            .map(|endpoint| {
                Arc::new(RpcClient::new_with_timeout_and_commitment(
                    endpoint.clone(),
                    RPC_TIMEOUT,
                    CommitmentConfig::confirmed(),
                ))
            })
            .collect();

        if clients.is_empty() {
            return Err(MissionControlError::ConfigError(
                "No RPC endpoints configured".into(),
            ));
        }

        Ok(Self {
            clients: Arc::new(clients),
            metrics_recorder,
            method_latencies: Arc::new(DashMap::new()),
            circuit_breakers: Arc::new(DashMap::new()),
        })
    }

    pub async fn start_collection(&self) {
        let mut ticker = interval(COLLECTION_INTERVAL);
        
        loop {
            ticker.tick().await;
            
            // Spawn parallel collection tasks for each client
            let mut tasks = vec![];
            
            for (idx, client) in self.clients.iter().enumerate() {
                let client = Arc::clone(client);
                let metrics = Arc::clone(&self.metrics_recorder);
                let latencies = Arc::clone(&self.method_latencies);
                let breakers = Arc::clone(&self.circuit_breakers);
                
                tasks.push(tokio::spawn(async move {
                    Self::collect_metrics_from_client(
                        client,
                        metrics,
                        latencies,
                        breakers,
                        idx,
                    ).await
                }));
            }
            
            // Wait for all collections to complete
            for task in tasks {
                if let Err(e) = task.await {
                    error!("RPC collection task failed: {}", e);
                }
            }
            
            // Calculate and record aggregate metrics
            self.calculate_percentiles();
        }
    }

    async fn collect_metrics_from_client(
        client: Arc<RpcClient>,
        metrics: Arc<MetricsRecorder>,
        latencies: Arc<DashMap<String, Vec<f64>>>,
        breakers: Arc<DashMap<String, CircuitBreaker>>,
        client_idx: usize,
    ) {
        let endpoint_key = format!("rpc_{}", client_idx);
        
        // Get or create circuit breaker for this endpoint
        let breaker = breakers
            .entry(endpoint_key.clone())
            .or_insert_with(CircuitBreaker::new)
            .clone();
        
        if !breaker.can_attempt() {
            warn!("Circuit breaker open for {}", endpoint_key);
            return;
        }
        
        // Collect version info
        if let Err(e) = Self::collect_version(&client, &metrics, &latencies, &breaker).await {
            error!("Failed to collect version: {}", e);
        }
        
        // Collect cluster nodes
        if let Err(e) = Self::collect_cluster_nodes(&client, &metrics, &latencies, &breaker).await {
            error!("Failed to collect cluster nodes: {}", e);
        }
        
        // Collect epoch info
        if let Err(e) = Self::collect_epoch_info(&client, &metrics, &latencies, &breaker).await {
            error!("Failed to collect epoch info: {}", e);
        }
        
        // Collect slot info
        if let Err(e) = Self::collect_slot_info(&client, &metrics, &latencies, &breaker).await {
            error!("Failed to collect slot info: {}", e);
        }
        
        // Collect block production
        if let Err(e) = Self::collect_block_production(&client, &metrics, &latencies, &breaker).await {
            error!("Failed to collect block production: {}", e);
        }
        
        // Collect performance samples for TPS
        if let Err(e) = Self::collect_performance_samples(&client, &metrics, &latencies, &breaker).await {
            error!("Failed to collect performance samples: {}", e);
        }
        
        // Collect prioritization fees
        if let Err(e) = Self::collect_prioritization_fees(&client, &metrics, &latencies, &breaker).await {
            error!("Failed to collect prioritization fees: {}", e);
        }
        
        // Collect vote accounts
        if let Err(e) = Self::collect_vote_accounts(&client, &metrics, &latencies, &breaker).await {
            error!("Failed to collect vote accounts: {}", e);
        }
    }

    async fn collect_version(
        client: &RpcClient,
        metrics: &MetricsRecorder,
        latencies: &DashMap<String, Vec<f64>>,
        breaker: &CircuitBreaker,
    ) -> Result<()> {
        let start = Instant::now();
        
        match client.get_version().await {
            Ok(version) => {
                let latency = start.elapsed().as_secs_f64() * 1000.0;
                latencies.entry("getVersion".into())
                    .or_insert_with(Vec::new)
                    .push(latency);
                
                metrics.record_rpc_latency("getVersion", latency);
                metrics.set_node_version(&version.solana_core);
                
                breaker.record_success();
                Ok(())
            }
            Err(e) => {
                breaker.record_failure();
                metrics.increment_rpc_errors("getVersion");
                Err(MissionControlError::RpcError(e.to_string()))
            }
        }
    }

    async fn collect_cluster_nodes(
        client: &RpcClient,
        metrics: &MetricsRecorder,
        latencies: &DashMap<String, Vec<f64>>,
        breaker: &CircuitBreaker,
    ) -> Result<()> {
        let start = Instant::now();
        
        match client.get_cluster_nodes().await {
            Ok(nodes) => {
                let latency = start.elapsed().as_secs_f64() * 1000.0;
                latencies.entry("getClusterNodes".into())
                    .or_insert_with(Vec::new)
                    .push(latency);
                
                metrics.record_rpc_latency("getClusterNodes", latency);
                metrics.set_cluster_node_count(nodes.len() as u64);
                
                breaker.record_success();
                Ok(())
            }
            Err(e) => {
                breaker.record_failure();
                metrics.increment_rpc_errors("getClusterNodes");
                Err(MissionControlError::RpcError(e.to_string()))
            }
        }
    }

    async fn collect_epoch_info(
        client: &RpcClient,
        metrics: &MetricsRecorder,
        latencies: &DashMap<String, Vec<f64>>,
        breaker: &CircuitBreaker,
    ) -> Result<()> {
        let start = Instant::now();
        
        match client.get_epoch_info().await {
            Ok(epoch_info) => {
                let latency = start.elapsed().as_secs_f64() * 1000.0;
                latencies.entry("getEpochInfo".into())
                    .or_insert_with(Vec::new)
                    .push(latency);
                
                metrics.record_rpc_latency("getEpochInfo", latency);
                metrics.set_epoch(epoch_info.epoch);
                metrics.set_slot_index(epoch_info.slot_index);
                metrics.set_slots_per_epoch(epoch_info.slots_in_epoch);
                
                breaker.record_success();
                Ok(())
            }
            Err(e) => {
                breaker.record_failure();
                metrics.increment_rpc_errors("getEpochInfo");
                Err(MissionControlError::RpcError(e.to_string()))
            }
        }
    }

    async fn collect_slot_info(
        client: &RpcClient,
        metrics: &MetricsRecorder,
        latencies: &DashMap<String, Vec<f64>>,
        breaker: &CircuitBreaker,
    ) -> Result<()> {
        let start = Instant::now();
        
        match client.get_slot().await {
            Ok(slot) => {
                let latency = start.elapsed().as_secs_f64() * 1000.0;
                latencies.entry("getSlot".into())
                    .or_insert_with(Vec::new)
                    .push(latency);
                
                metrics.record_rpc_latency("getSlot", latency);
                metrics.set_current_slot(slot);
                
                breaker.record_success();
            }
            Err(e) => {
                breaker.record_failure();
                metrics.increment_rpc_errors("getSlot");
                return Err(MissionControlError::RpcError(e.to_string()));
            }
        }
        
        // Also get block height
        let start = Instant::now();
        match client.get_block_height().await {
            Ok(height) => {
                let latency = start.elapsed().as_secs_f64() * 1000.0;
                latencies.entry("getBlockHeight".into())
                    .or_insert_with(Vec::new)
                    .push(latency);
                
                metrics.record_rpc_latency("getBlockHeight", latency);
                metrics.set_block_height(height);
                
                breaker.record_success();
                Ok(())
            }
            Err(e) => {
                breaker.record_failure();
                metrics.increment_rpc_errors("getBlockHeight");
                Err(MissionControlError::RpcError(e.to_string()))
            }
        }
    }

    async fn collect_block_production(
        client: &RpcClient,
        metrics: &MetricsRecorder,
        latencies: &DashMap<String, Vec<f64>>,
        breaker: &CircuitBreaker,
    ) -> Result<()> {
        let start = Instant::now();
        
        let config = RpcBlockProductionConfig {
            identity: None,
            range: None,
            commitment: Some(CommitmentConfig::confirmed()),
        };
        
        match client.get_block_production_with_config(config).await {
            Ok(production) => {
                let latency = start.elapsed().as_secs_f64() * 1000.0;
                latencies.entry("getBlockProduction".into())
                    .or_insert_with(Vec::new)
                    .push(latency);
                
                metrics.record_rpc_latency("getBlockProduction", latency);
                
                let total_slots = production.value.range.as_ref()
                    .map(|r| r.last_slot - r.first_slot)
                    .unwrap_or(0);
                    
                let produced_blocks: u64 = production.value.by_identity
                    .values()
                    .map(|v| v.0.len() as u64)
                    .sum();
                
                if total_slots > 0 {
                    let production_rate = produced_blocks as f64 / total_slots as f64;
                    metrics.set_block_production_rate(production_rate);
                }
                
                breaker.record_success();
                Ok(())
            }
            Err(e) => {
                breaker.record_failure();
                metrics.increment_rpc_errors("getBlockProduction");
                Err(MissionControlError::RpcError(e.to_string()))
            }
        }
    }

    async fn collect_performance_samples(
        client: &RpcClient,
        metrics: &MetricsRecorder,
        latencies: &DashMap<String, Vec<f64>>,
        breaker: &CircuitBreaker,
    ) -> Result<()> {
        let start = Instant::now();
        
        match client.get_recent_performance_samples(Some(60)).await {
            Ok(samples) => {
                let latency = start.elapsed().as_secs_f64() * 1000.0;
                latencies.entry("getRecentPerformanceSamples".into())
                    .or_insert_with(Vec::new)
                    .push(latency);
                
                metrics.record_rpc_latency("getRecentPerformanceSamples", latency);
                
                // Calculate TPS from samples
                let mut tps_values = Vec::new();
                for sample in samples.iter() {
                    if sample.num_slots > 0 {
                        let tps = sample.num_non_vote_transactions as f64 / sample.num_slots as f64;
                        tps_values.push(tps);
                    }
                }
                
                if !tps_values.is_empty() {
                    let avg_tps = tps_values.iter().sum::<f64>() / tps_values.len() as f64;
                    metrics.set_tps(avg_tps);
                    
                    // Calculate 1min and 5min averages
                    let tps_1min = tps_values.iter().take(12).sum::<f64>() 
                        / tps_values.len().min(12) as f64;
                    let tps_5min = tps_values.iter().sum::<f64>() / tps_values.len() as f64;
                    
                    metrics.set_tps_1min(tps_1min);
                    metrics.set_tps_5min(tps_5min);
                }
                
                breaker.record_success();
                Ok(())
            }
            Err(e) => {
                breaker.record_failure();
                metrics.increment_rpc_errors("getRecentPerformanceSamples");
                Err(MissionControlError::RpcError(e.to_string()))
            }
        }
    }

    async fn collect_prioritization_fees(
        client: &RpcClient,
        metrics: &MetricsRecorder,
        latencies: &DashMap<String, Vec<f64>>,
        breaker: &CircuitBreaker,
    ) -> Result<()> {
        let start = Instant::now();
        
        match client.get_recent_prioritization_fees(&[]).await {
            Ok(fees) => {
                let latency = start.elapsed().as_secs_f64() * 1000.0;
                latencies.entry("getRecentPrioritizationFees".into())
                    .or_insert_with(Vec::new)
                    .push(latency);
                
                metrics.record_rpc_latency("getRecentPrioritizationFees", latency);
                
                if !fees.is_empty() {
                    let mut priority_fees: Vec<u64> = fees.iter()
                        .map(|f| f.prioritization_fee)
                        .collect();
                    priority_fees.sort_unstable();
                    
                    let median = priority_fees[priority_fees.len() / 2];
                    let p75 = priority_fees[priority_fees.len() * 3 / 4];
                    let p95 = priority_fees[priority_fees.len() * 95 / 100];
                    
                    metrics.set_priority_fee_median(median);
                    metrics.set_priority_fee_p75(p75);
                    metrics.set_priority_fee_p95(p95);
                }
                
                breaker.record_success();
                Ok(())
            }
            Err(e) => {
                breaker.record_failure();
                metrics.increment_rpc_errors("getRecentPrioritizationFees");
                Err(MissionControlError::RpcError(e.to_string()))
            }
        }
    }

    async fn collect_vote_accounts(
        client: &RpcClient,
        metrics: &MetricsRecorder,
        latencies: &DashMap<String, Vec<f64>>,
        breaker: &CircuitBreaker,
    ) -> Result<()> {
        let start = Instant::now();
        
        match client.get_vote_accounts().await {
            Ok(accounts) => {
                let latency = start.elapsed().as_secs_f64() * 1000.0;
                latencies.entry("getVoteAccounts".into())
                    .or_insert_with(Vec::new)
                    .push(latency);
                
                metrics.record_rpc_latency("getVoteAccounts", latency);
                
                let active_stake: u64 = accounts.current.iter()
                    .map(|v| v.activated_stake)
                    .sum();
                let delinquent_stake: u64 = accounts.delinquent.iter()
                    .map(|v| v.activated_stake)
                    .sum();
                let total_stake = active_stake + delinquent_stake;
                
                metrics.set_active_stake(active_stake);
                metrics.set_delinquent_stake(delinquent_stake);
                metrics.set_total_stake(total_stake);
                
                if total_stake > 0 {
                    let participation = active_stake as f64 / total_stake as f64;
                    metrics.set_consensus_participation(participation);
                }
                
                metrics.set_active_validators(accounts.current.len() as u64);
                metrics.set_delinquent_validators(accounts.delinquent.len() as u64);
                
                breaker.record_success();
                Ok(())
            }
            Err(e) => {
                breaker.record_failure();
                metrics.increment_rpc_errors("getVoteAccounts");
                Err(MissionControlError::RpcError(e.to_string()))
            }
        }
    }

    fn calculate_percentiles(&self) {
        for entry in self.method_latencies.iter() {
            let method = entry.key();
            let mut latencies = entry.value().clone();
            
            if latencies.is_empty() {
                continue;
            }
            
            latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let p50 = latencies[latencies.len() / 2];
            let p95 = latencies[latencies.len() * 95 / 100];
            let p99 = latencies[latencies.len() * 99 / 100];
            
            self.metrics_recorder.set_method_percentiles(method, p50, p95, p99);
            
            // Clear old latencies to prevent memory growth
            if latencies.len() > 10000 {
                entry.value().clear();
            }
        }
    }
    
    pub async fn get_method_latencies(&self) -> HashMap<String, (f64, f64, f64)> {
        use std::collections::HashMap;
        let mut result = HashMap::new();
        
        for entry in self.method_latencies.iter() {
            let latencies = entry.value();
            if !latencies.is_empty() {
                let mut sorted = latencies.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                
                let p50 = sorted[sorted.len() / 2];
                let p95 = sorted[sorted.len() * 95 / 100];
                let p99 = sorted[sorted.len() * 99 / 100];
                
                result.insert(entry.key().clone(), (p50, p95, p99));
            }
        }
        
        result
    }
}