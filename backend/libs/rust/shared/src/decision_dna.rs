// LEGENDARY Decision DNA Cryptographic Fingerprinting System
// Creates unique, verifiable fingerprints for every MEV/ARB decision
// Enables forensic analysis and decision replay for optimization

use blake3::{Hash, Hasher};
use serde::{Serialize, Deserialize};
use std::time::{SystemTime, UNIX_EPOCH};
use bincode;

/// Decision context that gets fingerprinted
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionContext {
    pub timestamp_ns: u64,
    pub block_height: u64,
    pub slot: u64,
    pub leader_pubkey: [u8; 32],
    pub mempool_hash: [u8; 32],
    pub market_state: MarketSnapshot,
    pub features: Vec<f32>,
    pub model_version: String,
}

/// Market state snapshot at decision time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSnapshot {
    pub pool_reserves: Vec<(String, u64, u64)>,  // (pool_id, token_a, token_b)
    pub oracle_prices: Vec<(String, f64)>,        // (token, price)
    pub gas_price: u64,
    pub priority_fee: u64,
    pub jito_tip: u64,
}

/// Decision outcome that gets recorded
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOutcome {
    pub action: DecisionAction,
    pub confidence: f32,
    pub expected_value: f64,
    pub risk_score: f32,
    pub route: Vec<String>,
    pub amounts: Vec<u64>,
    pub slippage_tolerance: f32,
}

/// Types of decisions made
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionAction {
    Execute { tx_type: String },
    Skip { reason: String },
    Defer { until_slot: u64 },
    Hedge { primary: Box<DecisionOutcome>, secondary: Box<DecisionOutcome> },
}

/// Complete decision DNA record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionDNA {
    pub id: [u8; 32],                    // Unique fingerprint
    pub parent_id: Option<[u8; 32]>,     // Previous decision in chain
    pub context: DecisionContext,
    pub outcome: DecisionOutcome,
    pub metadata: DecisionMetadata,
}

/// Metadata for decision tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionMetadata {
    pub agent_id: String,
    pub strategy: String,
    pub latency_us: u64,
    pub cpu_cycles: u64,
    pub memory_bytes: u64,
    pub network_bytes: u64,
}

/// Decision result after execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionResult {
    pub dna_id: [u8; 32],
    pub success: bool,
    pub actual_value: f64,
    pub actual_slippage: f32,
    pub execution_time_ms: u64,
    pub gas_used: u64,
    pub signature: Option<String>,
    pub error: Option<String>,
}

/// Decision DNA builder and verifier
pub struct DecisionDNAEngine {
    agent_id: String,
    chain: Vec<[u8; 32]>,
    last_dna: Option<DecisionDNA>,
}

impl DecisionDNAEngine {
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            chain: Vec::new(),
            last_dna: None,
        }
    }

    /// Create a new decision DNA record
    pub fn create_dna(
        &mut self,
        context: DecisionContext,
        outcome: DecisionOutcome,
        strategy: impl Into<String>,
        latency_us: u64,
    ) -> DecisionDNA {
        // Get parent ID from chain
        let parent_id = self.chain.last().cloned();

        // Create metadata
        let metadata = DecisionMetadata {
            agent_id: self.agent_id.clone(),
            strategy: strategy.into(),
            latency_us,
            cpu_cycles: Self::estimate_cpu_cycles(latency_us),
            memory_bytes: Self::estimate_memory(&context, &outcome),
            network_bytes: Self::estimate_network(&outcome),
        };

        // Compute fingerprint
        let id = Self::compute_fingerprint(&context, &outcome, &metadata, parent_id);

        let dna = DecisionDNA {
            id,
            parent_id,
            context,
            outcome,
            metadata,
        };

        // Update chain
        self.chain.push(id);
        if self.chain.len() > 1000 {
            self.chain.drain(0..100);  // Keep last 900 decisions
        }
        self.last_dna = Some(dna.clone());

        dna
    }

    /// Compute cryptographic fingerprint for decision
    fn compute_fingerprint(
        context: &DecisionContext,
        outcome: &DecisionOutcome,
        metadata: &DecisionMetadata,
        parent_id: Option<[u8; 32]>,
    ) -> [u8; 32] {
        let mut hasher = Hasher::new();

        // Hash parent for chain integrity
        if let Some(parent) = parent_id {
            hasher.update(&parent);
        }

        // Hash context
        hasher.update(&context.timestamp_ns.to_le_bytes());
        hasher.update(&context.block_height.to_le_bytes());
        hasher.update(&context.slot.to_le_bytes());
        hasher.update(&context.leader_pubkey);
        hasher.update(&context.mempool_hash);
        
        // Hash market state
        let market_bytes = bincode::serialize(&context.market_state).unwrap_or_default();
        hasher.update(&market_bytes);

        // Hash features (quantized for stability)
        for feature in &context.features {
            let quantized = (*feature * 10000.0).round() as i32;
            hasher.update(&quantized.to_le_bytes());
        }

        // Hash outcome
        let outcome_bytes = bincode::serialize(outcome).unwrap_or_default();
        hasher.update(&outcome_bytes);

        // Hash metadata
        hasher.update(metadata.agent_id.as_bytes());
        hasher.update(metadata.strategy.as_bytes());
        hasher.update(&metadata.latency_us.to_le_bytes());

        *hasher.finalize().as_bytes()
    }

    /// Verify decision DNA integrity
    pub fn verify_dna(dna: &DecisionDNA) -> bool {
        let computed = Self::compute_fingerprint(
            &dna.context,
            &dna.outcome,
            &dna.metadata,
            dna.parent_id,
        );
        computed == dna.id
    }

    /// Verify chain integrity
    pub fn verify_chain(decisions: &[DecisionDNA]) -> bool {
        if decisions.is_empty() {
            return true;
        }

        let mut expected_parent: Option<[u8; 32]> = None;

        for dna in decisions {
            // Check parent linkage
            if dna.parent_id != expected_parent {
                return false;
            }

            // Verify fingerprint
            if !Self::verify_dna(dna) {
                return false;
            }

            expected_parent = Some(dna.id);
        }

        true
    }

    /// Replay decision from DNA
    pub fn replay_decision(dna: &DecisionDNA) -> Result<DecisionReplay, String> {
        // Verify integrity first
        if !Self::verify_dna(dna) {
            return Err("Invalid DNA fingerprint".to_string());
        }

        Ok(DecisionReplay {
            dna_id: dna.id,
            context: dna.context.clone(),
            outcome: dna.outcome.clone(),
            can_execute: Self::can_replay_context(&dna.context),
            estimated_current_value: Self::estimate_current_value(&dna.context, &dna.outcome),
        })
    }

    /// Check if decision context can be replayed
    fn can_replay_context(context: &DecisionContext) -> bool {
        // Check if market conditions are similar enough
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let age_ns = now.saturating_sub(context.timestamp_ns);
        age_ns < 60_000_000_000  // Less than 60 seconds old
    }

    /// Estimate current value of decision
    fn estimate_current_value(context: &DecisionContext, outcome: &DecisionOutcome) -> f64 {
        // This would use current market state to re-evaluate
        // For now, decay original estimate by time
        let age_seconds = (SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64 - context.timestamp_ns) / 1_000_000_000;
        
        let decay_factor = 0.99_f64.powi(age_seconds as i32);
        outcome.expected_value * decay_factor
    }

    // Estimation helpers
    fn estimate_cpu_cycles(latency_us: u64) -> u64 {
        latency_us * 3000  // Rough estimate: 3GHz CPU
    }

    fn estimate_memory(context: &DecisionContext, outcome: &DecisionOutcome) -> u64 {
        let context_size = std::mem::size_of_val(context) + 
                          context.features.len() * std::mem::size_of::<f32>();
        let outcome_size = bincode::serialize(outcome).map(|v| v.len()).unwrap_or(0);
        (context_size + outcome_size) as u64
    }

    fn estimate_network(outcome: &DecisionOutcome) -> u64 {
        match &outcome.action {
            DecisionAction::Execute { .. } => 1200,  // Typical transaction size
            DecisionAction::Hedge { .. } => 2400,     // Two transactions
            _ => 0,
        }
    }

    /// Export decision chain for analysis
    pub fn export_chain(&self) -> Vec<[u8; 32]> {
        self.chain.clone()
    }

    /// Get chain statistics
    pub fn get_stats(&self) -> ChainStats {
        ChainStats {
            total_decisions: self.chain.len(),
            unique_strategies: self.count_unique_strategies(),
            chain_head: self.chain.last().cloned(),
            last_decision_time: self.last_dna.as_ref()
                .map(|d| d.context.timestamp_ns),
        }
    }

    fn count_unique_strategies(&self) -> usize {
        // In production, track this incrementally
        1  // Placeholder
    }
}

/// Decision replay information
#[derive(Debug, Clone)]
pub struct DecisionReplay {
    pub dna_id: [u8; 32],
    pub context: DecisionContext,
    pub outcome: DecisionOutcome,
    pub can_execute: bool,
    pub estimated_current_value: f64,
}

/// Chain statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainStats {
    pub total_decisions: usize,
    pub unique_strategies: usize,
    pub chain_head: Option<[u8; 32]>,
    pub last_decision_time: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_dna_creation() {
        let mut engine = DecisionDNAEngine::new("test-agent");

        let context = DecisionContext {
            timestamp_ns: 1234567890,
            block_height: 1000,
            slot: 5000,
            leader_pubkey: [0u8; 32],
            mempool_hash: [1u8; 32],
            market_state: MarketSnapshot {
                pool_reserves: vec![],
                oracle_prices: vec![],
                gas_price: 100,
                priority_fee: 10,
                jito_tip: 5,
            },
            features: vec![0.1, 0.2, 0.3],
            model_version: "v1.0".to_string(),
        };

        let outcome = DecisionOutcome {
            action: DecisionAction::Execute { 
                tx_type: "arbitrage".to_string() 
            },
            confidence: 0.95,
            expected_value: 100.0,
            risk_score: 0.1,
            route: vec!["pool1".to_string(), "pool2".to_string()],
            amounts: vec![1000, 1100],
            slippage_tolerance: 0.01,
        };

        let dna = engine.create_dna(context, outcome, "test-strategy", 100);

        assert!(DecisionDNAEngine::verify_dna(&dna));
        assert_eq!(engine.chain.len(), 1);
        assert_eq!(engine.chain[0], dna.id);
    }

    #[test]
    fn test_chain_integrity() {
        let mut engine = DecisionDNAEngine::new("test-agent");
        let mut decisions = Vec::new();

        for i in 0..5 {
            let context = DecisionContext {
                timestamp_ns: 1234567890 + i,
                block_height: 1000 + i,
                slot: 5000 + i,
                leader_pubkey: [0u8; 32],
                mempool_hash: [i as u8; 32],
                market_state: MarketSnapshot {
                    pool_reserves: vec![],
                    oracle_prices: vec![],
                    gas_price: 100,
                    priority_fee: 10,
                    jito_tip: 5,
                },
                features: vec![i as f32 * 0.1],
                model_version: "v1.0".to_string(),
            };

            let outcome = DecisionOutcome {
                action: DecisionAction::Skip { 
                    reason: format!("test-{}", i) 
                },
                confidence: 0.5,
                expected_value: 0.0,
                risk_score: 0.5,
                route: vec![],
                amounts: vec![],
                slippage_tolerance: 0.0,
            };

            let dna = engine.create_dna(context, outcome, "test", 100);
            decisions.push(dna);
        }

        assert!(DecisionDNAEngine::verify_chain(&decisions));

        // Tamper with one decision
        decisions[2].metadata.latency_us = 999999;
        assert!(!DecisionDNAEngine::verify_chain(&decisions));
    }
}