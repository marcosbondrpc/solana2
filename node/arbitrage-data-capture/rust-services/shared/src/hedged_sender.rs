use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::{
    signature::Signature,
    transaction::Transaction,
    commitment_config::CommitmentConfig,
};
use anyhow::Result;

/// W-shape hedged transaction sender with multi-armed bandit routing
/// Sends through multiple paths with intelligent tip escalation
pub struct HedgedSender {
    /// Primary RPC endpoints
    rpc_clients: Vec<Arc<RpcClient>>,
    /// Jito block engine endpoints
    jito_clients: Vec<Arc<RpcClient>>,
    /// Bandit state for endpoint selection
    bandit: Arc<RwLock<BanditState>>,
    /// Tip escalation strategy
    tip_strategy: TipStrategy,
}

/// Multi-armed bandit for endpoint selection
struct BanditState {
    /// Success rates per endpoint
    success_rates: Vec<f64>,
    /// Latency estimates (exponential moving average)
    latencies: Vec<Duration>,
    /// Exploration count
    exploration_count: Vec<usize>,
    /// Exploitation count
    exploitation_count: Vec<usize>,
    /// UCB1 temperature
    temperature: f64,
}

/// Tip escalation strategy
#[derive(Clone)]
pub struct TipStrategy {
    /// Base tip in lamports
    pub base_tip: u64,
    /// Escalation multiplier
    pub escalation_factor: f64,
    /// Maximum tip in lamports
    pub max_tip: u64,
    /// Time between escalations
    pub escalation_interval: Duration,
}

impl Default for TipStrategy {
    fn default() -> Self {
        Self {
            base_tip: 10_000, // 0.00001 SOL
            escalation_factor: 2.0,
            max_tip: 1_000_000, // 0.001 SOL
            escalation_interval: Duration::from_millis(50),
        }
    }
}

/// Result of hedged send
pub struct HedgedResult {
    pub signature: Signature,
    pub latency: Duration,
    pub path: SendPath,
    pub tip_used: u64,
}

#[derive(Debug, Clone)]
pub enum SendPath {
    DirectRpc(usize),
    JitoBundle(usize),
    DualShot(usize, usize),
    TripleShot,
}

impl HedgedSender {
    pub fn new(
        rpc_endpoints: Vec<String>,
        jito_endpoints: Vec<String>,
        tip_strategy: TipStrategy,
    ) -> Result<Self> {
        let rpc_clients: Vec<_> = rpc_endpoints
            .into_iter()
            .map(|url| Arc::new(RpcClient::new(url)))
            .collect();
            
        let jito_clients: Vec<_> = jito_endpoints
            .into_iter()
            .map(|url| Arc::new(RpcClient::new(url)))
            .collect();
            
        let n_endpoints = rpc_clients.len() + jito_clients.len();
        
        let bandit = Arc::new(RwLock::new(BanditState {
            success_rates: vec![0.5; n_endpoints],
            latencies: vec![Duration::from_millis(100); n_endpoints],
            exploration_count: vec![0; n_endpoints],
            exploitation_count: vec![0; n_endpoints],
            temperature: 1.0,
        }));
        
        Ok(Self {
            rpc_clients,
            jito_clients,
            bandit,
            tip_strategy,
        })
    }
    
    /// Send transaction with W-shape hedging pattern
    pub async fn send_hedged(&self, tx: Transaction, urgency: Urgency) -> Result<HedgedResult> {
        let start = Instant::now();
        
        match urgency {
            Urgency::Low => self.send_single_shot(tx).await,
            Urgency::Medium => self.send_dual_shot(tx).await,
            Urgency::High => self.send_triple_shot(tx).await,
            Urgency::Critical => self.send_w_shape(tx).await,
        }
    }
    
    /// Single shot through best endpoint
    async fn send_single_shot(&self, tx: Transaction) -> Result<HedgedResult> {
        let endpoint_idx = self.select_best_endpoint().await;
        let start = Instant::now();
        
        if endpoint_idx < self.rpc_clients.len() {
            let client = &self.rpc_clients[endpoint_idx];
            let sig = client.send_transaction(&tx).await?;
            
            self.update_bandit_success(endpoint_idx, start.elapsed()).await;
            
            Ok(HedgedResult {
                signature: sig,
                latency: start.elapsed(),
                path: SendPath::DirectRpc(endpoint_idx),
                tip_used: 0,
            })
        } else {
            let jito_idx = endpoint_idx - self.rpc_clients.len();
            let sig = self.send_jito_bundle(&tx, jito_idx, self.tip_strategy.base_tip).await?;
            
            self.update_bandit_success(endpoint_idx, start.elapsed()).await;
            
            Ok(HedgedResult {
                signature: sig,
                latency: start.elapsed(),
                path: SendPath::JitoBundle(jito_idx),
                tip_used: self.tip_strategy.base_tip,
            })
        }
    }
    
    /// Dual shot through two best endpoints
    async fn send_dual_shot(&self, tx: Transaction) -> Result<HedgedResult> {
        let (first, second) = self.select_top_two_endpoints().await;
        let start = Instant::now();
        
        // Send through both endpoints simultaneously
        let tx1 = tx.clone();
        let tx2 = tx;
        
        let handle1 = {
            let client = self.get_client(first).clone();
            tokio::spawn(async move {
                client.send_transaction(&tx1).await
            })
        };
        
        let handle2 = {
            let client = self.get_client(second).clone();
            tokio::spawn(async move {
                client.send_transaction(&tx2).await
            })
        };
        
        // Race for first success
        tokio::select! {
            res1 = handle1 => {
                if let Ok(Ok(sig)) = res1 {
                    self.update_bandit_success(first, start.elapsed()).await;
                    return Ok(HedgedResult {
                        signature: sig,
                        latency: start.elapsed(),
                        path: SendPath::DualShot(first, second),
                        tip_used: 0,
                    });
                }
            }
            res2 = handle2 => {
                if let Ok(Ok(sig)) = res2 {
                    self.update_bandit_success(second, start.elapsed()).await;
                    return Ok(HedgedResult {
                        signature: sig,
                        latency: start.elapsed(),
                        path: SendPath::DualShot(first, second),
                        tip_used: 0,
                    });
                }
            }
        }
        
        anyhow::bail!("Dual shot failed")
    }
    
    /// Triple shot with escalating tips
    async fn send_triple_shot(&self, tx: Transaction) -> Result<HedgedResult> {
        let endpoints = self.select_top_three_endpoints().await;
        let start = Instant::now();
        let mut tip = self.tip_strategy.base_tip;
        
        for (i, endpoint) in endpoints.iter().enumerate() {
            let tx_clone = tx.clone();
            let current_tip = if i > 0 {
                tip = self.escalate_tip(tip);
                tip
            } else {
                0
            };
            
            let client = self.get_client(*endpoint).clone();
            let handle = tokio::spawn(async move {
                if current_tip > 0 {
                    // Send as Jito bundle with tip
                    // Implementation depends on Jito API
                    client.send_transaction(&tx_clone).await
                } else {
                    client.send_transaction(&tx_clone).await
                }
            });
            
            // Wait briefly between shots
            if i < endpoints.len() - 1 {
                tokio::time::sleep(self.tip_strategy.escalation_interval).await;
            }
            
            if let Ok(Ok(sig)) = handle.await {
                self.update_bandit_success(*endpoint, start.elapsed()).await;
                return Ok(HedgedResult {
                    signature: sig,
                    latency: start.elapsed(),
                    path: SendPath::TripleShot,
                    tip_used: current_tip,
                });
            }
        }
        
        anyhow::bail!("Triple shot failed")
    }
    
    /// W-shape pattern: initial burst, wait, secondary burst
    async fn send_w_shape(&self, tx: Transaction) -> Result<HedgedResult> {
        let start = Instant::now();
        
        // First wave: send through top 2 endpoints
        let (first, second) = self.select_top_two_endpoints().await;
        
        let tx1 = tx.clone();
        let handle1 = {
            let client = self.get_client(first).clone();
            tokio::spawn(async move {
                client.send_transaction(&tx1).await
            })
        };
        
        // Wait for trough
        tokio::time::sleep(Duration::from_millis(30)).await;
        
        // Second wave: send with escalated tip through Jito
        let tip = self.escalate_tip(self.tip_strategy.base_tip);
        let tx2 = tx.clone();
        let handle2 = self.send_jito_bundle_async(tx2, 0, tip);
        
        // Wait for second trough
        tokio::time::sleep(Duration::from_millis(30)).await;
        
        // Third wave: maximum tip through all Jito endpoints
        let max_tip = self.tip_strategy.max_tip;
        for (i, client) in self.jito_clients.iter().enumerate() {
            let tx_clone = tx.clone();
            let _ = self.send_jito_bundle_async(tx_clone, i, max_tip);
        }
        
        // Collect results
        tokio::select! {
            res1 = handle1 => {
                if let Ok(Ok(sig)) = res1 {
                    return Ok(HedgedResult {
                        signature: sig,
                        latency: start.elapsed(),
                        path: SendPath::TripleShot,
                        tip_used: 0,
                    });
                }
            }
            res2 = handle2 => {
                if let Ok(sig) = res2 {
                    return Ok(HedgedResult {
                        signature: sig,
                        latency: start.elapsed(),
                        path: SendPath::TripleShot,
                        tip_used: tip,
                    });
                }
            }
        }
        
        anyhow::bail!("W-shape send failed")
    }
    
    /// Select best endpoint using UCB1 algorithm
    async fn select_best_endpoint(&self) -> usize {
        let bandit = self.bandit.read().await;
        let total_pulls: usize = bandit.exploration_count.iter().sum();
        
        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = 0;
        
        for i in 0..bandit.success_rates.len() {
            let exploration_bonus = if bandit.exploration_count[i] == 0 {
                f64::INFINITY
            } else {
                bandit.temperature * (2.0 * (total_pulls as f64).ln() / bandit.exploration_count[i] as f64).sqrt()
            };
            
            let latency_penalty = bandit.latencies[i].as_millis() as f64 / 1000.0;
            let score = bandit.success_rates[i] + exploration_bonus - latency_penalty;
            
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        
        best_idx
    }
    
    /// Select top two endpoints
    async fn select_top_two_endpoints(&self) -> (usize, usize) {
        let bandit = self.bandit.read().await;
        let mut scores: Vec<_> = (0..bandit.success_rates.len())
            .map(|i| {
                let latency_penalty = bandit.latencies[i].as_millis() as f64 / 1000.0;
                (i, bandit.success_rates[i] - latency_penalty)
            })
            .collect();
            
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        (scores[0].0, scores[1].0)
    }
    
    /// Select top three endpoints
    async fn select_top_three_endpoints(&self) -> Vec<usize> {
        let bandit = self.bandit.read().await;
        let mut scores: Vec<_> = (0..bandit.success_rates.len())
            .map(|i| {
                let latency_penalty = bandit.latencies[i].as_millis() as f64 / 1000.0;
                (i, bandit.success_rates[i] - latency_penalty)
            })
            .collect();
            
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        scores.iter().take(3).map(|(i, _)| *i).collect()
    }
    
    /// Update bandit state after successful send
    async fn update_bandit_success(&self, endpoint: usize, latency: Duration) {
        let mut bandit = self.bandit.write().await;
        
        // Update success rate (exponential moving average)
        let alpha = 0.1;
        bandit.success_rates[endpoint] = 
            alpha * 1.0 + (1.0 - alpha) * bandit.success_rates[endpoint];
            
        // Update latency estimate
        bandit.latencies[endpoint] = Duration::from_millis(
            (alpha * latency.as_millis() as f64 + 
             (1.0 - alpha) * bandit.latencies[endpoint].as_millis() as f64) as u64
        );
        
        // Update counts
        bandit.exploration_count[endpoint] += 1;
        
        // Decay temperature
        bandit.temperature *= 0.999;
        bandit.temperature = bandit.temperature.max(0.1);
    }
    
    /// Get client by global index
    fn get_client(&self, idx: usize) -> &Arc<RpcClient> {
        if idx < self.rpc_clients.len() {
            &self.rpc_clients[idx]
        } else {
            &self.jito_clients[idx - self.rpc_clients.len()]
        }
    }
    
    /// Escalate tip amount
    fn escalate_tip(&self, current_tip: u64) -> u64 {
        let new_tip = (current_tip as f64 * self.tip_strategy.escalation_factor) as u64;
        new_tip.min(self.tip_strategy.max_tip)
    }
    
    /// Send through Jito with tip
    async fn send_jito_bundle(&self, tx: &Transaction, jito_idx: usize, tip: u64) -> Result<Signature> {
        // Simplified - actual implementation would build proper Jito bundle
        let client = &self.jito_clients[jito_idx];
        // Add tip transaction to bundle
        // Send bundle through Jito API
        client.send_transaction(tx).await.map_err(Into::into)
    }
    
    /// Async Jito bundle send
    fn send_jito_bundle_async(&self, tx: Transaction, jito_idx: usize, tip: u64) 
        -> tokio::task::JoinHandle<Result<Signature>> 
    {
        let client = self.jito_clients[jito_idx].clone();
        tokio::spawn(async move {
            // Simplified - actual implementation would build proper Jito bundle
            client.send_transaction(&tx).await.map_err(Into::into)
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Urgency {
    Low,      // Single shot
    Medium,   // Dual shot
    High,     // Triple shot with tips
    Critical, // W-shape with max tips
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_bandit_selection() {
        let sender = HedgedSender::new(
            vec!["http://localhost:8899".to_string()],
            vec![],
            TipStrategy::default(),
        ).unwrap();
        
        let best = sender.select_best_endpoint().await;
        assert_eq!(best, 0); // Only one endpoint
    }
}