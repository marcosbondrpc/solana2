// Budgeted Thompson Sampling for Route & Fee Policy Selection
// Optimized for MEV/Arbitrage with budget constraints

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use rand::prelude::*;
use rand_distr::{Beta, Distribution};
use serde::{Serialize, Deserialize};

/// Arm in the multi-armed bandit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditArm {
    pub arm_id: String,
    pub successes: f64,
    pub failures: f64,
    pub total_reward: f64,
    pub total_cost: f64,
    pub total_pulls: u64,
    pub last_selected: u64,  // Timestamp
    pub metadata: HashMap<String, String>,
}

impl BanditArm {
    pub fn new(arm_id: String) -> Self {
        Self {
            arm_id,
            successes: 1.0,  // Beta prior
            failures: 1.0,   // Beta prior
            total_reward: 0.0,
            total_cost: 0.0,
            total_pulls: 0,
            last_selected: 0,
            metadata: HashMap::new(),
        }
    }
    
    pub fn sample_theta(&self) -> f64 {
        let beta = Beta::new(self.successes, self.failures).unwrap();
        let mut rng = thread_rng();
        beta.sample(&mut rng)
    }
    
    pub fn update(&mut self, reward: f64, cost: f64, success: bool) {
        if success {
            self.successes += 1.0;
        } else {
            self.failures += 1.0;
        }
        self.total_reward += reward;
        self.total_cost += cost;
        self.total_pulls += 1;
        self.last_selected = chrono::Utc::now().timestamp_nanos() as u64;
    }
    
    pub fn avg_reward(&self) -> f64 {
        if self.total_pulls > 0 {
            self.total_reward / self.total_pulls as f64
        } else {
            0.0
        }
    }
    
    pub fn avg_cost(&self) -> f64 {
        if self.total_pulls > 0 {
            self.total_cost / self.total_pulls as f64
        } else {
            0.0
        }
    }
    
    pub fn net_value(&self) -> f64 {
        self.avg_reward() - self.avg_cost()
    }
}

/// Budget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    pub total_budget: f64,
    pub remaining_budget: f64,
    pub min_cost_threshold: f64,
    pub budget_reset_interval_ms: u64,
    pub last_reset: u64,
}

impl BudgetConfig {
    pub fn new(total_budget: f64) -> Self {
        Self {
            total_budget,
            remaining_budget: total_budget,
            min_cost_threshold: 0.01,
            budget_reset_interval_ms: 60_000,  // 1 minute
            last_reset: chrono::Utc::now().timestamp_nanos() as u64,
        }
    }
    
    pub fn consume(&mut self, cost: f64) -> bool {
        if self.remaining_budget >= cost {
            self.remaining_budget -= cost;
            true
        } else {
            false
        }
    }
    
    pub fn reset_if_needed(&mut self) {
        let now = chrono::Utc::now().timestamp_nanos() as u64;
        if now - self.last_reset > self.budget_reset_interval_ms * 1_000_000 {
            self.remaining_budget = self.total_budget;
            self.last_reset = now;
        }
    }
    
    pub fn utilization(&self) -> f64 {
        (self.total_budget - self.remaining_budget) / self.total_budget
    }
}

/// Budgeted Thompson Sampling Bandit
pub struct BudgetedThompsonSampling {
    arms: Arc<RwLock<HashMap<String, BanditArm>>>,
    budget: Arc<RwLock<BudgetConfig>>,
    exploration_rate: f64,
    decay_factor: f64,
    context_weights: HashMap<String, f64>,
}

impl BudgetedThompsonSampling {
    pub fn new(total_budget: f64, exploration_rate: f64) -> Self {
        Self {
            arms: Arc::new(RwLock::new(HashMap::new())),
            budget: Arc::new(RwLock::new(BudgetConfig::new(total_budget))),
            exploration_rate,
            decay_factor: 0.99,
            context_weights: HashMap::new(),
        }
    }
    
    /// Select an arm based on Thompson Sampling with budget constraints
    pub async fn select_arm(
        &self,
        available_arms: Vec<String>,
        costs: HashMap<String, f64>,
        context: Option<HashMap<String, f64>>,
    ) -> Option<(String, f64)> {
        // Reset budget if needed
        {
            let mut budget = self.budget.write().await;
            budget.reset_if_needed();
        }
        
        // Get or create arms
        let mut arms = self.arms.write().await;
        for arm_id in &available_arms {
            if !arms.contains_key(arm_id) {
                arms.insert(arm_id.clone(), BanditArm::new(arm_id.clone()));
            }
        }
        
        // Thompson Sampling with budget awareness
        let mut best_arm = None;
        let mut best_score = f64::NEG_INFINITY;
        
        let budget_remaining = self.budget.read().await.remaining_budget;
        
        for arm_id in available_arms {
            let arm = arms.get(&arm_id).unwrap();
            let cost = costs.get(&arm_id).copied().unwrap_or(0.0);
            
            // Skip if over budget
            if cost > budget_remaining {
                continue;
            }
            
            // Sample from posterior
            let theta = arm.sample_theta();
            
            // Add exploration bonus
            let exploration_bonus = self.calculate_exploration_bonus(arm);
            
            // Add context score if available
            let context_score = if let Some(ref ctx) = context {
                self.calculate_context_score(ctx, &arm_id)
            } else {
                0.0
            };
            
            // Budget-aware scoring
            let budget_factor = (budget_remaining / self.budget.read().await.total_budget).sqrt();
            let score = theta + exploration_bonus + context_score * budget_factor;
            
            if score > best_score {
                best_score = score;
                best_arm = Some((arm_id.clone(), cost));
            }
        }
        
        // Consume budget if arm selected
        if let Some((ref arm_id, cost)) = best_arm {
            let mut budget = self.budget.write().await;
            if !budget.consume(cost) {
                return None;
            }
        }
        
        best_arm
    }
    
    /// Update arm with observed reward
    pub async fn update_arm(
        &self,
        arm_id: &str,
        reward: f64,
        cost: f64,
        success: bool,
    ) {
        let mut arms = self.arms.write().await;
        if let Some(arm) = arms.get_mut(arm_id) {
            arm.update(reward, cost, success);
        }
    }
    
    /// Calculate exploration bonus using UCB-style formula
    fn calculate_exploration_bonus(&self, arm: &BanditArm) -> f64 {
        if arm.total_pulls == 0 {
            return self.exploration_rate;
        }
        
        let confidence = (2.0 * (arm.total_pulls as f64).ln() / arm.total_pulls as f64).sqrt();
        self.exploration_rate * confidence
    }
    
    /// Calculate context score
    fn calculate_context_score(&self, context: &HashMap<String, f64>, arm_id: &str) -> f64 {
        let mut score = 0.0;
        
        // Simple linear combination of context features
        for (key, value) in context {
            if let Some(weight) = self.context_weights.get(key) {
                score += weight * value;
            }
        }
        
        // Arm-specific adjustments based on context
        match arm_id {
            "jito" if context.get("network_congestion").copied().unwrap_or(0.0) > 0.7 => {
                score *= 1.5;  // Prefer Jito during high congestion
            }
            "direct" if context.get("gas_price").copied().unwrap_or(0.0) < 10.0 => {
                score *= 1.2;  // Prefer direct during low gas
            }
            _ => {}
        }
        
        score
    }
    
    /// Get current statistics
    pub async fn get_stats(&self) -> BanditStats {
        let arms = self.arms.read().await;
        let budget = self.budget.read().await;
        
        let mut arm_stats = Vec::new();
        for (_, arm) in arms.iter() {
            arm_stats.push(ArmStats {
                arm_id: arm.arm_id.clone(),
                total_pulls: arm.total_pulls,
                avg_reward: arm.avg_reward(),
                avg_cost: arm.avg_cost(),
                net_value: arm.net_value(),
                success_rate: arm.successes / (arm.successes + arm.failures),
            });
        }
        
        BanditStats {
            total_arms: arms.len(),
            total_pulls: arms.values().map(|a| a.total_pulls).sum(),
            total_reward: arms.values().map(|a| a.total_reward).sum(),
            total_cost: arms.values().map(|a| a.total_cost).sum(),
            budget_utilization: budget.utilization(),
            budget_remaining: budget.remaining_budget,
            arm_stats,
        }
    }
    
    /// Apply decay to older observations
    pub async fn apply_decay(&self) {
        let mut arms = self.arms.write().await;
        for arm in arms.values_mut() {
            arm.successes = 1.0 + (arm.successes - 1.0) * self.decay_factor;
            arm.failures = 1.0 + (arm.failures - 1.0) * self.decay_factor;
        }
    }
    
    /// Save state for persistence
    pub async fn save_state(&self) -> BanditState {
        let arms = self.arms.read().await;
        let budget = self.budget.read().await;
        
        BanditState {
            arms: arms.clone(),
            budget: budget.clone(),
            exploration_rate: self.exploration_rate,
            decay_factor: self.decay_factor,
            context_weights: self.context_weights.clone(),
        }
    }
    
    /// Load state from persistence
    pub async fn load_state(&self, state: BanditState) {
        *self.arms.write().await = state.arms;
        *self.budget.write().await = state.budget;
    }
}

/// Statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditStats {
    pub total_arms: usize,
    pub total_pulls: u64,
    pub total_reward: f64,
    pub total_cost: f64,
    pub budget_utilization: f64,
    pub budget_remaining: f64,
    pub arm_stats: Vec<ArmStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmStats {
    pub arm_id: String,
    pub total_pulls: u64,
    pub avg_reward: f64,
    pub avg_cost: f64,
    pub net_value: f64,
    pub success_rate: f64,
}

/// Persistent state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditState {
    pub arms: HashMap<String, BanditArm>,
    pub budget: BudgetConfig,
    pub exploration_rate: f64,
    pub decay_factor: f64,
    pub context_weights: HashMap<String, f64>,
}

/// Route selection specific implementation
pub struct RouteSelector {
    bandit: BudgetedThompsonSampling,
    routes: HashMap<String, RouteConfig>,
}

#[derive(Debug, Clone)]
pub struct RouteConfig {
    pub route_id: String,
    pub submission_type: SubmissionType,
    pub priority_fee_multiplier: f64,
    pub max_retries: u32,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone)]
pub enum SubmissionType {
    Direct,           // Direct to leader
    Jito,            // Via Jito bundle
    Hedged,          // Both simultaneously
    Canary,          // Test transaction
}

impl RouteSelector {
    pub fn new(budget: f64) -> Self {
        let mut routes = HashMap::new();
        
        routes.insert("direct".to_string(), RouteConfig {
            route_id: "direct".to_string(),
            submission_type: SubmissionType::Direct,
            priority_fee_multiplier: 1.0,
            max_retries: 2,
            timeout_ms: 500,
        });
        
        routes.insert("jito".to_string(), RouteConfig {
            route_id: "jito".to_string(),
            submission_type: SubmissionType::Jito,
            priority_fee_multiplier: 1.5,
            max_retries: 1,
            timeout_ms: 750,
        });
        
        routes.insert("hedged".to_string(), RouteConfig {
            route_id: "hedged".to_string(),
            submission_type: SubmissionType::Hedged,
            priority_fee_multiplier: 2.0,
            max_retries: 1,
            timeout_ms: 500,
        });
        
        Self {
            bandit: BudgetedThompsonSampling::new(budget, 0.1),
            routes,
        }
    }
    
    /// Fetch live lamports-per-CU percentile (e.g., P90) from Redis.
    /// Returns None if unavailable.
    pub async fn fetch_lamports_per_cu(redis: &mut redis::aio::ConnectionManager, percentile: &str) -> Option<f64> {
        let key = format!("priority-fees:current:{}", percentile);
        match redis.get::<_, String>(&key).await {
            Ok(val) => val.parse::<f64>().ok(),
            Err(_) => None,
        }
    }

    /// Crude compute unit estimator for a given strategy path.
    /// Replace with model-based estimator once per-strategy profiles are collected.
    pub fn estimate_compute_units(path_len: usize) -> f64 {
        // Base 120k CU plus 40k per leg as a heuristic.
        let base = 120_000.0;
        let per_leg = 40_000.0 * (path_len as f64).max(1.0);
        base + per_leg
    }

    pub async fn select_route(
        &self,
        opportunity_value: f64,
        lamports_per_cu: f64,
        expected_cu: f64,
        network_congestion: f64,
    ) -> Option<RouteConfig> {
        let available_arms: Vec<String> = self.routes.keys().cloned().collect();
        
        // Calculate costs for each route
        let mut costs = HashMap::new();
        for (route_id, config) in &self.routes {
            let cost = lamports_per_cu * expected_cu * config.priority_fee_multiplier;
            costs.insert(route_id.clone(), cost);
        }
        
        // Create context
        let mut context = HashMap::new();
        context.insert("opportunity_value".to_string(), opportunity_value);
        context.insert("lamports_per_cu".to_string(), lamports_per_cu);
        context.insert("expected_cu".to_string(), expected_cu);
        context.insert("network_congestion".to_string(), network_congestion);
        
        // Select arm
        if let Some((arm_id, _)) = self.bandit.select_arm(available_arms, costs, Some(context)).await {
            self.routes.get(&arm_id).cloned()
        } else {
            None
        }
    }
    
    pub async fn update_route_performance(
        &self,
        route_id: &str,
        profit: f64,
        gas_cost: f64,
        landed: bool,
    ) {
        self.bandit.update_arm(route_id, profit, gas_cost, landed).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_bandit_selection() {
        let bandit = BudgetedThompsonSampling::new(1000.0, 0.1);
        
        let arms = vec!["arm1".to_string(), "arm2".to_string(), "arm3".to_string()];
        let mut costs = HashMap::new();
        costs.insert("arm1".to_string(), 10.0);
        costs.insert("arm2".to_string(), 20.0);
        costs.insert("arm3".to_string(), 30.0);
        
        // Should select an arm
        let selected = bandit.select_arm(arms.clone(), costs.clone(), None).await;
        assert!(selected.is_some());
        
        // Update with reward
        if let Some((arm_id, cost)) = selected {
            bandit.update_arm(&arm_id, 100.0, cost, true).await;
        }
        
        // Check stats
        let stats = bandit.get_stats().await;
        assert_eq!(stats.total_pulls, 1);
        assert!(stats.budget_utilization > 0.0);
    }
    
    #[tokio::test]
    async fn test_budget_constraints() {
        let bandit = BudgetedThompsonSampling::new(50.0, 0.1);
        
        let arms = vec!["expensive".to_string()];
        let mut costs = HashMap::new();
        costs.insert("expensive".to_string(), 100.0);  // Over budget
        
        let selected = bandit.select_arm(arms, costs, None).await;
        assert!(selected.is_none());  // Should not select over-budget arm
    }
    
    #[tokio::test]
    async fn test_route_selector() {
        let selector = RouteSelector::new(1000.0);
        
        // Example values: lamports_per_cu = 100, expected_cu = 150_000
        let route = selector.select_route(100.0, 100.0, 150_000.0, 0.5).await;
        assert!(route.is_some());
        
        if let Some(config) = route {
            selector.update_route_performance(
                &config.route_id,
                50.0,  // profit
                10.0,  // gas cost
                true,  // landed
            ).await;
        }
    }
}