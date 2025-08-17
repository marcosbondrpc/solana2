use rand_distr::{Distribution, Beta};
use rand::thread_rng;
use std::collections::HashMap;
use redis::aio::ConnectionManager;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum RouteArm { DirectTpu, JitoNY, JitoAMS, JitoZRH }

#[derive(Default, Clone)]
pub struct ArmStats {
    pub alpha: f64,
    pub beta: f64,
    pub mean_ev: f64,
    pub m2: f64,
    pub n: f64,
}

impl ArmStats {
    pub fn update(&mut self, landed: bool, realized_ev: f64) {
        if landed { self.alpha += 1.0; } else { self.beta += 1.0; }
        self.n += 1.0;
        let delta = realized_ev - self.mean_ev;
        self.mean_ev += delta / self.n;
        self.m2 += delta * (realized_ev - self.mean_ev);
    }
    
    pub fn sample_score(&self) -> f64 {
        let a = self.alpha.max(1e-3);
        let b = self.beta.max(1e-3);
        let beta = Beta::new(a, b).unwrap();
        let p_land = beta.sample(&mut thread_rng());
        p_land * self.mean_ev.max(0.0)
    }
}

pub struct ThompsonSelector {
    pub arms: HashMap<RouteArm, ArmStats>,
}

impl ThompsonSelector {
    pub fn new_default() -> Self {
        let mut arms = HashMap::new();
        for arm in [RouteArm::DirectTpu, RouteArm::JitoNY, RouteArm::JitoAMS, RouteArm::JitoZRH] {
            arms.insert(arm, ArmStats { alpha: 3.0, beta: 3.0, ..Default::default() });
        }
        Self { arms }
    }

    /// Seed priors from Redis HGETALL keys of the form:
    /// { p50_lamports_per_cu, expected_cu, p_land_prior }
    pub async fn with_redis(mut self, client: &redis::Client, keys: &[(RouteArm, &str)]) -> Self {
        let mut con: Option<ConnectionManager> = client.get_connection_manager().await.ok();
        for (arm, key) in keys {
            if let Some(ref mut c) = con {
                if let Ok(redis::Value::Array(items)) = redis::cmd("HGETALL").arg(key)
                    .query_async(c).await {
                    let mut map = HashMap::new();
                    let mut it = items.into_iter();
                    while let (Some(k), Some(v)) = (it.next(), it.next()) {
                        if let (redis::Value::BulkString(kb), redis::Value::BulkString(vb)) = (k, v) {
                            map.insert(String::from_utf8_lossy(&kb).to_string(), String::from_utf8_lossy(&vb).to_string());
                        }
                    }
                    let e = self.arms.entry(arm.clone()).or_default();
                    if let Some(pr) = map.get("p_land_prior") {
                        let p: f64 = pr.parse().unwrap_or(0.5);
                        let k = 10.0; e.alpha = 1.0 + p * k; e.beta = 1.0 + (1.0 - p) * k;
                    }
                    if let (Some(p50), Some(exp_cu)) = (map.get("p50_lamports_per_cu"), map.get("expected_cu")) {
                        let lpcu: f64 = p50.parse().unwrap_or(0.0);
                        let ecu: f64  = exp_cu.parse().unwrap_or(0.0);
                        e.mean_ev = -(lpcu * ecu);
                    }
                }
            }
        }
        self
    }

    pub fn select(&self) -> RouteArm {
        self.arms.iter()
            .max_by(|a,b| a.1.sample_score().partial_cmp(&b.1.sample_score()).unwrap())
            .map(|(k, _)| k.clone())
            .unwrap_or(RouteArm::DirectTpu)
    }
}