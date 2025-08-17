use anyhow::Result;
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::signature::Signature;
use std::sync::Arc;
use tracing::{debug, info};

use crate::bundle::Bundle;
use crate::config::Config;

// Route selection and Jito engine integration
// Temporarily stub RouteSelector until crate is wired
pub struct RouteSelector {
    _budget: f64,
}

impl RouteSelector {
    pub fn new(budget: f64) -> Self { Self { _budget: budget } }
    pub async fn fetch_lamports_per_cu(_redis: &mut redis::aio::ConnectionManager, _key: &str) -> anyhow::Result<f64> {
        Ok(100.0)
    }
    pub fn estimate_compute_units(path_len: usize) -> f64 { (path_len as f64) * 100_000.0 }
    pub async fn select_route(&self, _ev: f64, _lamports_per_cu: f64, _expected_cu: f64, _congestion: f64) -> Option<RouteCfg> {
        Some(RouteCfg { submission_type: "Jito".to_string() })
    }
}

#[derive(Debug, Clone)]
pub struct RouteCfg { pub submission_type: String }
use redis::aio::ConnectionManager as RedisConnectionManager;
use backend_shared_strategy::bandit::ts::{RouteArm, ThompsonSelector};

pub struct SubmissionEngine {
    config: Arc<Config>,
    rpc_client: Arc<RpcClient>,
    ts_selector: ThompsonSelector,
    redis: Option<RedisConnectionManager>,
}

impl SubmissionEngine {
    pub async fn new(config: Arc<Config>, rpc_client: Arc<RpcClient>) -> Result<Self> {
        // Initialize Thompson sampler selector
        let ts_selector = ThompsonSelector::new_default();
        // Best-effort Redis connection for fee snapshots
        let redis = if let Ok(client) = redis::Client::open(config.redis_url.clone()) {
            match client.get_tokio_connection_manager().await {
                Ok(cm) => Some(cm),
                Err(_) => None,
            }
        } else { None };

        Ok(Self { config, rpc_client, ts_selector, redis })
    }
    
    pub async fn submit(&self, bundle: Bundle) -> Result<()> {
        debug!("Submitting bundle with {} transactions", bundle.transactions.len());
        // Pull fee percentile and estimate CU
        let lamports_per_cu = if let Some(redis) = &self.redis {
            let mut r = redis.clone();
            RouteSelector::fetch_lamports_per_cu(&mut r, "p90").await.unwrap_or(100.0)
        } else { 100.0 };

        let expected_cu = RouteSelector::estimate_compute_units(2);

        // Select route based on EV
        let arm = self.ts_selector.select();
        info!("Selected arm: {:?}", arm);
        if matches!(arm, RouteArm::JitoNY | RouteArm::JitoAMS | RouteArm::JitoZRH) {
            // TODO: Implement Direct vs Jito vs Hedged submission paths
        } else {
            info!("No route selected within budget; skipping submission");
        }
        Ok(())
    }
    
    pub async fn check_status(&self, signature: &Signature) -> Result<bool> {
        // Placeholder for status checking logic
        Ok(false)
    }
}