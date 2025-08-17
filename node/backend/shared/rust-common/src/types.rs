use serde::{Deserialize, Serialize};
use solana_sdk::pubkey::Pubkey;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MevOpportunity {
    pub id: String,
    pub profit_lamports: u64,
    pub source_pool: Pubkey,
    pub target_pool: Pubkey,
    pub timestamp: i64,
}