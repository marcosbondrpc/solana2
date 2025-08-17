use anyhow::Result;
use moka::future::Cache as MokaCache;
use std::sync::Arc;
use std::time::Duration;
use rust_decimal::Decimal;
use solana_sdk::pubkey::Pubkey;

#[derive(Debug, Clone)]
pub struct Cache {
    pool_cache: Arc<MokaCache<Pubkey, PoolCacheEntry>>,
    price_cache: Arc<MokaCache<String, PriceCacheEntry>>,
    opportunity_cache: Arc<MokaCache<String, OpportunityCacheEntry>>,
}

#[derive(Debug, Clone)]
pub struct PoolCacheEntry {
    pub pool_address: Pubkey,
    pub token_a: Pubkey,
    pub token_b: Pubkey,
    pub reserves: (u64, u64),
    pub last_update: i64,
}

#[derive(Debug, Clone)]
pub struct PriceCacheEntry {
    pub price: Decimal,
    pub volume: Decimal,
    pub timestamp: i64,
}

#[derive(Debug, Clone)]
pub struct OpportunityCacheEntry {
    pub id: String,
    pub profit: Decimal,
    pub executed: bool,
    pub timestamp: i64,
}

impl Cache {
    pub fn new() -> Self {
        Self {
            pool_cache: Arc::new(
                MokaCache::builder()
                    .max_capacity(10000)
                    .time_to_live(Duration::from_secs(60))
                    .build()
            ),
            price_cache: Arc::new(
                MokaCache::builder()
                    .max_capacity(5000)
                    .time_to_live(Duration::from_secs(30))
                    .build()
            ),
            opportunity_cache: Arc::new(
                MokaCache::builder()
                    .max_capacity(1000)
                    .time_to_live(Duration::from_secs(300))
                    .build()
            ),
        }
    }

    pub async fn get_pool(&self, pool_address: &Pubkey) -> Option<PoolCacheEntry> {
        self.pool_cache.get(pool_address).await
    }

    pub async fn set_pool(&self, pool_address: Pubkey, entry: PoolCacheEntry) {
        self.pool_cache.insert(pool_address, entry).await;
    }

    pub async fn get_price(&self, pair: &str) -> Option<PriceCacheEntry> {
        self.price_cache.get(pair).await
    }

    pub async fn set_price(&self, pair: String, entry: PriceCacheEntry) {
        self.price_cache.insert(pair, entry).await;
    }

    pub async fn get_opportunity(&self, id: &str) -> Option<OpportunityCacheEntry> {
        self.opportunity_cache.get(id).await
    }

    pub async fn set_opportunity(&self, id: String, entry: OpportunityCacheEntry) {
        self.opportunity_cache.insert(id, entry).await;
    }

    pub async fn clear_all(&self) {
        self.pool_cache.invalidate_all();
        self.price_cache.invalidate_all();
        self.opportunity_cache.invalidate_all();
    }

    pub async fn stats(&self) -> CacheStats {
        CacheStats {
            pool_entries: self.pool_cache.entry_count(),
            price_entries: self.price_cache.entry_count(),
            opportunity_entries: self.opportunity_cache.entry_count(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub pool_entries: u64,
    pub price_entries: u64,
    pub opportunity_entries: u64,
}