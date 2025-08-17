use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use rust_common::dedupe::Dedupe;
use rust_common::ring::MpmcRing;
use backend_shared_strategy::bandit::ts::ThompsonSelector;
mod arbiter;
mod ev;
mod sims;
mod router;
mod lineage_writer;

fn main() {
    // Placeholders to show wiring; the real event loop lives in service runtime.
    let _ring: MpmcRing<Vec<u8>> = MpmcRing::with_capacity_pow2(1 << 14);
    let _dedupe = Dedupe::new(Duration::from_millis(400));
    let _router = ThompsonSelector::new_default();
    let _arb = arbiter::CreditArbiter::new();

    println!("Arbitrage Engine initialized");
}
