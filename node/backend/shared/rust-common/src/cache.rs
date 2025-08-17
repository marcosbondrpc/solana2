use dashmap::DashMap;
use std::sync::Arc;

pub struct Cache<K, V> {
    inner: Arc<DashMap<K, V>>,
}

impl<K: Eq + std::hash::Hash, V: Clone> Cache<K, V> {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(DashMap::new()),
        }
    }
    
    pub fn get(&self, key: &K) -> Option<V> {
        self.inner.get(key).map(|v| v.clone())
    }
    
    pub fn insert(&self, key: K, value: V) {
        self.inner.insert(key, value);
    }
}