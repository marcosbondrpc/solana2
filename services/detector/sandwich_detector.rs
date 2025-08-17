//! Sandwich Attack Detection Engine
//! DEFENSIVE-ONLY: Pure analysis, no execution
//! Implements slot-local bracket heuristic with multi-pattern detection

use ahash::{AHashMap, AHashSet};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clickhouse::{Client, Row};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tracing::{debug, info, warn};

// Detection constants
const MAX_SLOT_DISTANCE: u64 = 3;
const MIN_PRICE_IMPACT: f64 = 0.001; // 0.1%
const MIN_CONFIDENCE: f32 = 0.75;
const ADJACENCY_THRESHOLD_MS: f64 = 50.0;

/// Sandwich pattern detector using slot-local bracket heuristic
#[derive(Debug, Clone)]
pub struct SandwichDetector {
    client: Client,
    slot_cache: Arc<DashMap<u64, SlotTransactions>>,
    detected_patterns: Arc<DashMap<String, SandwichCandidate>>,
    config: DetectorConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DetectorConfig {
    pub min_confidence: f32,
    pub max_slot_distance: u64,
    pub min_price_impact: f64,
    pub adjacency_threshold_ms: f64,
    pub enable_obfuscation_detection: bool,
    pub enable_multi_address_detection: bool,
}

/// Slot-local transaction cache for pattern detection
#[derive(Debug, Clone)]
pub struct SlotTransactions {
    pub slot: u64,
    pub transactions: Vec<Transaction>,
    pub swap_map: AHashMap<String, Vec<usize>>, // pool -> tx indices
    pub payer_map: AHashMap<String, Vec<usize>>, // payer -> tx indices
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub sig: String,
    pub slot: u64,
    pub ts: DateTime<Utc>,
    pub payer: String,
    pub pool: String,
    pub token_in: String,
    pub token_out: String,
    pub amount_in: f64,
    pub amount_out: f64,
    pub fee: u64,
    pub priority_fee: u64,
    pub programs: Vec<String>,
    pub position_in_block: Option<u32>,
    pub bundle_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Row)]
pub struct SandwichCandidate {
    #[serde(with = "clickhouse::serde::time::datetime64::nanos")]
    pub detection_ts: DateTime<Utc>,
    pub slot: u64,
    pub victim_sig: String,
    pub attacker_a_sig: String,
    pub attacker_b_sig: String,
    pub attacker_addr: String,
    pub victim_addr: String,
    pub pool: String,
    pub d_ms: f64,
    pub d_slots: u16,
    pub slippage_victim: f64,
    pub price_reversion: f64,
    pub evidence: String,
    pub score_rule: f32,
    pub score_gnn: f32,
    pub score_transformer: f32,
    pub ensemble_score: f32,
    pub attack_style: String,
    pub victim_selection: String,
    pub victim_loss_sol: f64,
    pub attacker_profit_sol: f64,
    pub fee_burn_sol: f64,
    pub dna_fingerprint: String,
    pub model_version: String,
}

/// Evidence types for sandwich detection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Evidence {
    Bracket,       // Clear A-V-B pattern in same slot
    SlipRebound,   // Price impact and reversion detected
    Both,          // Both bracket and slip patterns
    Weak,          // Low confidence detection
}

impl SandwichDetector {
    pub async fn new(config: DetectorConfig, clickhouse: Client) -> Result<Self> {
        Ok(Self {
            client: clickhouse,
            slot_cache: Arc::new(DashMap::new()),
            detected_patterns: Arc::new(DashMap::new()),
            config,
        })
    }
    
    /// Process incoming transaction for sandwich detection
    pub async fn process_transaction(&self, tx: Transaction) -> Result<Option<SandwichCandidate>> {
        // Add to slot cache
        self.update_slot_cache(tx.clone()).await?;
        
        // Check for sandwich patterns
        if let Some(candidate) = self.detect_sandwich_pattern(&tx).await? {
            // Validate candidate
            if self.validate_candidate(&candidate).await? {
                // Store detection
                self.store_detection(&candidate).await?;
                return Ok(Some(candidate));
            }
        }
        
        Ok(None)
    }
    
    /// Update slot-local transaction cache
    async fn update_slot_cache(&self, tx: Transaction) -> Result<()> {
        let mut slot_txs = self.slot_cache.entry(tx.slot).or_insert_with(|| {
            SlotTransactions {
                slot: tx.slot,
                transactions: Vec::new(),
                swap_map: AHashMap::new(),
                payer_map: AHashMap::new(),
            }
        });
        
        let tx_index = slot_txs.transactions.len();
        
        // Update pool mapping
        slot_txs.swap_map
            .entry(tx.pool.clone())
            .or_insert_with(Vec::new)
            .push(tx_index);
        
        // Update payer mapping
        slot_txs.payer_map
            .entry(tx.payer.clone())
            .or_insert_with(Vec::new)
            .push(tx_index);
        
        slot_txs.transactions.push(tx);
        
        // Clean old slots
        self.cleanup_old_slots().await?;
        
        Ok(())
    }
    
    /// Core sandwich detection logic using bracket heuristic
    async fn detect_sandwich_pattern(&self, victim_tx: &Transaction) -> Result<Option<SandwichCandidate>> {
        let slot_range = (
            victim_tx.slot.saturating_sub(self.config.max_slot_distance),
            victim_tx.slot + self.config.max_slot_distance,
        );
        
        // Look for potential attackers in nearby slots
        for slot in slot_range.0..=slot_range.1 {
            if let Some(slot_data) = self.slot_cache.get(&slot) {
                // Get all transactions on the same pool
                if let Some(pool_txs) = slot_data.swap_map.get(&victim_tx.pool) {
                    // Check for bracket pattern
                    if let Some(candidate) = self.check_bracket_pattern(
                        victim_tx,
                        &slot_data.transactions,
                        pool_txs,
                    ).await? {
                        return Ok(Some(candidate));
                    }
                }
                
                // Check for multi-address variants if enabled
                if self.config.enable_multi_address_detection {
                    if let Some(candidate) = self.check_multi_address_pattern(
                        victim_tx,
                        &slot_data.transactions,
                    ).await? {
                        return Ok(Some(candidate));
                    }
                }
            }
        }
        
        Ok(None)
    }
    
    /// Check for classic A-V-B bracket pattern
    async fn check_bracket_pattern(
        &self,
        victim: &Transaction,
        slot_txs: &[Transaction],
        pool_indices: &[usize],
    ) -> Result<Option<SandwichCandidate>> {
        // Need at least 3 transactions for sandwich
        if pool_indices.len() < 3 {
            return Ok(None);
        }
        
        // Find victim position
        let victim_pos = pool_indices.iter()
            .position(|&i| slot_txs[i].sig == victim.sig);
        
        if let Some(v_pos) = victim_pos {
            // Check for attacker before and after
            if v_pos > 0 && v_pos < pool_indices.len() - 1 {
                let before_idx = pool_indices[v_pos - 1];
                let after_idx = pool_indices[v_pos + 1];
                
                let before_tx = &slot_txs[before_idx];
                let after_tx = &slot_txs[after_idx];
                
                // Check if same attacker
                if before_tx.payer == after_tx.payer {
                    // Check swap directions (opposite to victim)
                    if self.is_sandwich_pattern(before_tx, victim, after_tx) {
                        return Ok(Some(self.build_candidate(
                            before_tx,
                            victim,
                            after_tx,
                            Evidence::Bracket,
                        ).await?));
                    }
                }
            }
        }
        
        Ok(None)
    }
    
    /// Check for obfuscated multi-address patterns
    async fn check_multi_address_pattern(
        &self,
        victim: &Transaction,
        slot_txs: &[Transaction],
    ) -> Result<Option<SandwichCandidate>> {
        // Look for coordinated attacks from different addresses
        let mut potential_attackers: AHashMap<String, Vec<&Transaction>> = AHashMap::new();
        
        for tx in slot_txs {
            if tx.pool == victim.pool && tx.sig != victim.sig {
                // Group by similar patterns
                let pattern_key = self.get_pattern_key(tx);
                potential_attackers.entry(pattern_key)
                    .or_insert_with(Vec::new)
                    .push(tx);
            }
        }
        
        // Check for sandwich patterns in grouped transactions
        for (_, group) in potential_attackers {
            if group.len() >= 2 {
                // Find transactions before and after victim
                let before: Vec<_> = group.iter()
                    .filter(|tx| tx.ts < victim.ts)
                    .collect();
                let after: Vec<_> = group.iter()
                    .filter(|tx| tx.ts > victim.ts)
                    .collect();
                
                if !before.is_empty() && !after.is_empty() {
                    // Check for sandwich pattern
                    for b_tx in &before {
                        for a_tx in &after {
                            if self.is_sandwich_pattern(b_tx, victim, a_tx) {
                                return Ok(Some(self.build_candidate(
                                    b_tx,
                                    victim,
                                    a_tx,
                                    Evidence::Both,
                                ).await?));
                            }
                        }
                    }
                }
            }
        }
        
        Ok(None)
    }
    
    /// Generate pattern key for grouping similar transactions
    fn get_pattern_key(&self, tx: &Transaction) -> String {
        // Group by similar fee patterns and timing
        format!(
            "{}_{}_{}_{}",
            tx.priority_fee / 100000, // Round to nearest 0.1 SOL
            tx.programs.join(","),
            tx.token_in,
            tx.token_out,
        )
    }
    
    /// Check if transactions form a sandwich pattern
    fn is_sandwich_pattern(&self, before: &Transaction, victim: &Transaction, after: &Transaction) -> bool {
        // Before: Buy what victim is buying (push price up)
        // Victim: Buys at inflated price
        // After: Sell what was bought (capture profit)
        
        let before_buys = before.token_out == victim.token_out;
        let after_sells = after.token_in == before.token_out;
        let opposite_direction = before.token_in == after.token_out;
        
        // Check price impact
        let price_impact = self.calculate_price_impact(before, victim, after);
        
        before_buys && after_sells && opposite_direction && price_impact > self.config.min_price_impact
    }
    
    /// Calculate price impact of sandwich
    fn calculate_price_impact(&self, before: &Transaction, victim: &Transaction, after: &Transaction) -> f64 {
        // Calculate victim's effective price vs expected
        let victim_price = victim.amount_in / victim.amount_out;
        let before_price = before.amount_in / before.amount_out;
        let after_price = after.amount_out / after.amount_in;
        
        // Price impact = difference from fair price
        let fair_price = (before_price + after_price) / 2.0;
        (victim_price - fair_price).abs() / fair_price
    }
    
    /// Build sandwich candidate from detected pattern
    async fn build_candidate(
        &self,
        before: &Transaction,
        victim: &Transaction,
        after: &Transaction,
        evidence: Evidence,
    ) -> Result<SandwichCandidate> {
        let d_ms = (after.ts.timestamp_millis() - before.ts.timestamp_millis()) as f64;
        let d_slots = (after.slot - before.slot) as u16;
        
        // Calculate economic impact
        let victim_loss = self.calculate_victim_loss(before, victim, after);
        let attacker_profit = self.calculate_attacker_profit(before, after);
        let fee_burn = (before.fee + after.fee) as f64 / 1e9; // Convert to SOL
        
        // Generate DNA fingerprint
        let dna_fingerprint = self.generate_dna_fingerprint(before, victim, after);
        
        Ok(SandwichCandidate {
            detection_ts: Utc::now(),
            slot: victim.slot,
            victim_sig: victim.sig.clone(),
            attacker_a_sig: before.sig.clone(),
            attacker_b_sig: after.sig.clone(),
            attacker_addr: before.payer.clone(),
            victim_addr: victim.payer.clone(),
            pool: victim.pool.clone(),
            d_ms,
            d_slots,
            slippage_victim: self.calculate_price_impact(before, victim, after),
            price_reversion: self.calculate_price_reversion(before, after),
            evidence: format!("{:?}", evidence),
            score_rule: self.calculate_rule_score(before, victim, after),
            score_gnn: 0.0, // Placeholder for GNN model
            score_transformer: 0.0, // Placeholder for transformer model
            ensemble_score: self.calculate_ensemble_score(evidence),
            attack_style: self.classify_attack_style(before, victim, after),
            victim_selection: self.classify_victim(victim),
            victim_loss_sol: victim_loss,
            attacker_profit_sol: attacker_profit,
            fee_burn_sol: fee_burn,
            dna_fingerprint,
            model_version: "sandwich_v1".to_string(),
        })
    }
    
    fn calculate_victim_loss(&self, before: &Transaction, victim: &Transaction, after: &Transaction) -> f64 {
        let expected_out = victim.amount_in * (before.amount_out / before.amount_in);
        let actual_out = victim.amount_out;
        let loss_tokens = expected_out - actual_out;
        
        // Convert to SOL value (simplified)
        loss_tokens * 0.001 // Placeholder conversion
    }
    
    fn calculate_attacker_profit(&self, before: &Transaction, after: &Transaction) -> f64 {
        let spent = before.amount_in;
        let received = after.amount_out;
        let profit = received - spent;
        
        // Convert to SOL value
        profit * 0.001 // Placeholder conversion
    }
    
    fn calculate_price_reversion(&self, before: &Transaction, after: &Transaction) -> f64 {
        let before_price = before.amount_in / before.amount_out;
        let after_price = after.amount_out / after.amount_in;
        (after_price - before_price).abs() / before_price
    }
    
    fn calculate_rule_score(&self, before: &Transaction, victim: &Transaction, after: &Transaction) -> f32 {
        let mut score = 0.0;
        
        // Timing proximity
        let time_diff = (after.ts.timestamp_millis() - before.ts.timestamp_millis()) as f64;
        if time_diff < 100.0 {
            score += 0.3;
        }
        
        // Same attacker
        if before.payer == after.payer {
            score += 0.3;
        }
        
        // Clear opposite trades
        if before.token_out == after.token_in {
            score += 0.2;
        }
        
        // High priority fees
        if before.priority_fee > 100000 && after.priority_fee > 100000 {
            score += 0.2;
        }
        
        score.min(1.0)
    }
    
    fn calculate_ensemble_score(&self, evidence: Evidence) -> f32 {
        match evidence {
            Evidence::Both => 0.95,
            Evidence::Bracket => 0.85,
            Evidence::SlipRebound => 0.75,
            Evidence::Weak => 0.60,
        }
    }
    
    fn classify_attack_style(&self, before: &Transaction, victim: &Transaction, after: &Transaction) -> String {
        let fee_ratio = (before.priority_fee + after.priority_fee) as f64 / victim.fee as f64;
        let timing_precision = (after.ts.timestamp_millis() - before.ts.timestamp_millis()) as f64;
        
        if fee_ratio > 10.0 && timing_precision < 50.0 {
            "surgical".to_string()
        } else if fee_ratio > 5.0 {
            "adaptive".to_string()
        } else {
            "shotgun".to_string()
        }
    }
    
    fn classify_victim(&self, victim: &Transaction) -> String {
        // Classify based on transaction patterns
        if victim.amount_in > 10000.0 {
            "whale".to_string()
        } else if victim.amount_in < 100.0 {
            "retail".to_string()
        } else {
            "unknown".to_string()
        }
    }
    
    fn generate_dna_fingerprint(&self, before: &Transaction, victim: &Transaction, after: &Transaction) -> String {
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        hasher.update(before.sig.as_bytes());
        hasher.update(victim.sig.as_bytes());
        hasher.update(after.sig.as_bytes());
        hasher.update(&victim.slot.to_le_bytes());
        format!("{}", hasher.finalize())
    }
    
    /// Validate detected candidate
    async fn validate_candidate(&self, candidate: &SandwichCandidate) -> Result<bool> {
        // Check minimum confidence
        if candidate.ensemble_score < self.config.min_confidence {
            return Ok(false);
        }
        
        // Check for duplicate detection
        if self.detected_patterns.contains_key(&candidate.victim_sig) {
            return Ok(false);
        }
        
        // Additional validation rules
        if candidate.d_slots > self.config.max_slot_distance as u16 {
            return Ok(false);
        }
        
        if candidate.slippage_victim < self.config.min_price_impact {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Store detection in ClickHouse
    async fn store_detection(&self, candidate: &SandwichCandidate) -> Result<()> {
        let mut insert = self.client.insert("candidates")?;
        insert.write(candidate).await?;
        insert.end().await?;
        
        // Cache detection
        self.detected_patterns.insert(
            candidate.victim_sig.clone(),
            candidate.clone(),
        );
        
        info!(
            "Detected sandwich: victim={}, attacker={}, profit={:.4} SOL, confidence={:.2}",
            &candidate.victim_sig[..8],
            &candidate.attacker_addr[..8],
            candidate.attacker_profit_sol,
            candidate.ensemble_score,
        );
        
        Ok(())
    }
    
    /// Clean up old slot data
    async fn cleanup_old_slots(&self) -> Result<()> {
        let current_slot = self.get_current_slot().await?;
        let cutoff = current_slot.saturating_sub(self.config.max_slot_distance * 2);
        
        self.slot_cache.retain(|&slot, _| slot >= cutoff);
        
        Ok(())
    }
    
    async fn get_current_slot(&self) -> Result<u64> {
        // Get current slot from cache or RPC
        Ok(0) // Placeholder
    }
}