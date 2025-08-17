use backend_shared_strategy::bandit::ts::{RouteArm, ThompsonSelector};
use std::collections::HashMap;

pub struct Router {
    pub ts: ThompsonSelector,
}

impl Router {
    pub fn new() -> Self {
        Self { ts: ThompsonSelector::new_default() }
    }

    /// Select a route using Thompson sampling weighted by provided EV estimates per arm.
    pub fn pick(&mut self, ev_by_arm: &HashMap<RouteArm, f64>) -> RouteArm {
        for (arm, ev) in ev_by_arm.iter() {
            if let Some(stats) = self.ts.arms.get_mut(arm) {
                stats.mean_ev = *ev;
            }
        }
        self.ts.select()
    }

    /// Update arm statistics with realized outcome.
    pub fn update(&mut self, arm: RouteArm, landed: bool, realized_ev: f64) {
        if let Some(s) = self.ts.arms.get_mut(&arm) {
            s.update(landed, realized_ev);
        }
    }
}


