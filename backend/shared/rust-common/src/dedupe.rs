use blake3::Hasher;
use dashmap::DashMap;
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct Dedupe {
    seen: DashMap<[u8; 32], Instant>,
    ttl: Duration,
}

impl Dedupe {
    pub fn new(ttl: Duration) -> Self {
        Self {
            seen: DashMap::new(),
            ttl,
        }
    }

    #[inline]
    pub fn hash_packets<'a, I: IntoIterator<Item = &'a [u8]>>(iter: I) -> [u8; 32] {
        let mut h = Hasher::new();
        for pkt in iter {
            h.update(pkt);
        }
        *h.finalize().as_bytes()
    }

    /// Returns true if this key is newly observed.
    pub fn first_seen(&self, key: [u8; 32]) -> bool {
        self.gc();
        self.seen.insert(key, Instant::now()).is_none()
    }

    fn gc(&self) {
        let ttl = self.ttl;
        let now = Instant::now();
        self.seen.retain(|_, t| now.duration_since(*t) < ttl);
    }
}


