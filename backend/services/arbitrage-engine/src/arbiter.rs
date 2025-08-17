use dashmap::DashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Eq)]
pub struct Key(pub [u8; 32]);

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}
impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) { self.0.hash(state) }
}

struct Lease { until: Instant }

#[derive(Clone)]
pub struct CreditArbiter(Arc<DashMap<Key, Lease>>);

impl CreditArbiter {
    pub fn new() -> Self { Self(Arc::new(DashMap::new())) }

    pub fn try_acquire(&self, k: Key, ttl: Duration) -> bool {
        let now = Instant::now();
        match self.0.entry(k) {
            dashmap::mapref::entry::Entry::Occupied(mut e) => {
                if e.get().until > now { return false; }
                e.insert(Lease { until: now + ttl });
                true
            }
            dashmap::mapref::entry::Entry::Vacant(v) => { v.insert(Lease { until: now + ttl }); true }
        }
    }

    pub fn release(&self, k: &Key) { let _ = self.0.remove(k); }
}



