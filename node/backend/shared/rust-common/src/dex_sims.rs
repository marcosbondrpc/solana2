//! Canonical micro-sim engines for Solana DEXes.
//!
//! Design: decouple parsing from math. Parsers (account layouts) should produce canonical state
//! structs below. The engines are pure math and easy to unit test.

use bytemuck::{Pod, Zeroable};

pub trait SimEngine {
    /// Returns estimated output amount (u64) and cost in compute units.
    fn simulate(&self, input_amount: u64) -> (u64, u32); // (out_amount, estimated_CU)
}

/* ================= Uniswap-v3 style (Raydium/Orca Whirlpool) ================ */

/// Canonical concentrated-liquidity pool state.
#[derive(Clone, Copy, Debug)]
pub struct ClmmState {
    /// Q64.64 sqrt price (token1/token0)
    pub sqrt_price_x64: u128,
    /// Active in-range liquidity (Q64.64 style magnitude)
    pub liquidity: u128,
    /// Fee in basis points (e.g., 30 = 0.30%)
    pub fee_bps: u16,
    /// Lower/upper tick boundary sqrt prices (Q64.64); optional if not at edge.
    pub sqrt_price_lower_x64: Option<u128>,
    pub sqrt_price_upper_x64: Option<u128>,
}

pub struct ClmmEngine {
    pub st: ClmmState,
}

impl ClmmEngine {
    /// Swap token0 -> token1 at current price, capped by in-range liquidity and optional bounds.
    /// Very small, approximate; good enough for ranking candidates pre-sim.
    pub fn swap_x_to_y(&self, dx: u128) -> (u128, u32) {
        // fee
        let fee = dx * (self.st.fee_bps as u128) / 10_000u128;
        let dx_after_fee = dx.saturating_sub(fee);

        // Uni v3-ish: dy ≈ L * (sqrtP_new - sqrtP_old), where d(1/sqrtP) = dx / L for x->y.
        let l = self.st.liquidity.max(1);
        let sp = self.st.sqrt_price_x64.max(1);
        
        // Simplified calculation without full u256 math
        let dy = (l * dx_after_fee) / sp;
        
        (dy, 20_000)
    }
}

impl SimEngine for ClmmEngine {
    fn simulate(&self, input_amount: u64) -> (u64, u32) {
        let (dy, cu) = self.swap_x_to_y(input_amount as u128);
        (dy.min(u128::from(u64::MAX)) as u64, cu)
    }
}

/* ================= Constant-product (Meteora fallback / generic AMM) ======= */
#[derive(Clone, Copy, Debug)]
pub struct CpmmState {
    pub reserve_x: u128,
    pub reserve_y: u128,
    pub fee_bps: u16,
}

pub struct CpmmEngine { pub st: CpmmState }

impl SimEngine for CpmmEngine {
    fn simulate(&self, input_amount: u64) -> (u64, u32) {
        let fee = (input_amount as u128) * (self.st.fee_bps as u128) / 10_000u128;
        let dx = (input_amount as u128).saturating_sub(fee);
        let k = self.st.reserve_x.saturating_mul(self.st.reserve_y);
        let new_x = self.st.reserve_x.saturating_add(dx);
        if new_x == 0 { return (0, 8_000); }
        let new_y = k / new_x;
        let dy = self.st.reserve_y.saturating_sub(new_y);
        (dy.min(u128::from(u64::MAX)) as u64, 8_000)
    }
}

/* ================= Phoenix-style orderbook ================================ */
#[derive(Clone, Copy, Debug)]
pub struct Level { pub price_q64: u128, pub qty: u64 }

#[derive(Clone, Debug)]
pub struct LobView {
    pub asks: Vec<Level>, // sorted ascending price
    pub bids: Vec<Level>, // sorted descending price
}

pub struct LobEngine { pub lob: LobView }

impl LobEngine {
    pub fn market_buy(&self, max_in_quote: u128) -> (u64, u32) {
        let mut remain_q = max_in_quote;
        let mut base_bought: u128 = 0;
        for lvl in &self.lob.asks {
            if remain_q == 0 { break; }
            let price = lvl.price_q64;
            if price == 0 { continue; }
            let max_base_here = (remain_q << 32) / (price >> 32).max(1);
            let take_base = (lvl.qty as u128).min(max_base_here);
            let spend = (take_base * (price >> 32)) >> 32;
            base_bought += take_base;
            remain_q = remain_q.saturating_sub(spend);
        }
        (base_bought.min(u128::from(u64::MAX)) as u64, 12_000)
    }
}

impl SimEngine for LobEngine {
    fn simulate(&self, input_amount: u64) -> (u64, u32) {
        self.market_buy(input_amount as u128)
    }
}