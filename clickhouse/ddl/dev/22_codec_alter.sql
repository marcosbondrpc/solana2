-- ClickHouse Codec Optimizations for MEV Data
-- Reduces storage by 60-80% while maintaining query performance

-- ============================================================================
-- CODEC SELECTION GUIDE:
-- DoubleDelta: Best for timestamps and monotonic sequences
-- Delta: Good for incrementing counters and IDs
-- Gorilla: Excellent for floating-point time series (prices, rates)
-- T64: Optimal for integers with small range
-- ZSTD: General purpose, good for strings and low-cardinality data
-- LZ4HC: Fast decompression, moderate compression
-- ============================================================================

-- Optimize main arbitrage detection table
ALTER TABLE arbitrage_opportunities MODIFY COLUMN 
    timestamp DateTime64(3) CODEC(DoubleDelta, LZ4),
    slot UInt64 CODEC(Delta, ZSTD(1)),
    profit_lamports Int64 CODEC(T64),
    gas_estimate UInt32 CODEC(T64),
    price_impact Float32 CODEC(Gorilla),
    pool_a_reserves UInt64 CODEC(Gorilla, ZSTD(1)),
    pool_b_reserves UInt64 CODEC(Gorilla, ZSTD(1)),
    sqrt_price_a UInt128 CODEC(Gorilla),
    sqrt_price_b UInt128 CODEC(Gorilla);

-- Optimize transaction submission table
ALTER TABLE mev_transactions MODIFY COLUMN
    submitted_at DateTime64(6) CODEC(DoubleDelta),
    confirmed_at Nullable(DateTime64(6)) CODEC(DoubleDelta),
    slot UInt64 CODEC(Delta),
    signature FixedString(88) CODEC(ZSTD(3)),
    sender_pubkey FixedString(44) CODEC(ZSTD(3)),
    priority_fee UInt64 CODEC(T64),
    compute_units UInt32 CODEC(T64),
    lamports_transferred UInt64 CODEC(T64),
    status Enum8('pending' = 1, 'confirmed' = 2, 'failed' = 3) CODEC(ZSTD(1));

-- Optimize pool state tracking
ALTER TABLE pool_states MODIFY COLUMN
    observed_at DateTime64(3) CODEC(DoubleDelta),
    slot UInt64 CODEC(Delta),
    pool_address FixedString(44) CODEC(ZSTD(3)),
    token_a FixedString(44) CODEC(ZSTD(3)),
    token_b FixedString(44) CODEC(ZSTD(3)),
    reserve_a UInt128 CODEC(Gorilla),
    reserve_b UInt128 CODEC(Gorilla),
    fee_rate UInt16 CODEC(T64),
    liquidity UInt128 CODEC(Gorilla),
    virtual_price Float64 CODEC(Gorilla);

-- Create specialized table for high-frequency price data
CREATE TABLE IF NOT EXISTS price_ticks (
    timestamp DateTime64(9) CODEC(DoubleDelta),  -- Nanosecond precision
    pool_address FixedString(44) CODEC(ZSTD(3)),
    price Float64 CODEC(Gorilla),
    volume UInt64 CODEC(T64),
    liquidity UInt128 CODEC(Gorilla),
    
    -- Delta encoding for tick data
    price_delta Float32 CODEC(Gorilla),
    volume_delta Int64 CODEC(T64),
    
    INDEX idx_pool pool_address TYPE bloom_filter(0.001) GRANULARITY 1,
    INDEX idx_time timestamp TYPE minmax GRANULARITY 8192
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (pool_address, timestamp)
TTL timestamp + INTERVAL 7 DAY TO DISK 'cold_storage'
SETTINGS 
    index_granularity = 8192,
    min_bytes_for_wide_part = 10485760,  -- 10MB
    compress_marks = true,
    compress_primary_key = true;

-- Optimize leader schedule tracking
ALTER TABLE leader_slots MODIFY COLUMN
    slot UInt64 CODEC(Delta),
    leader_pubkey FixedString(44) CODEC(ZSTD(3)),
    slot_time_ms UInt16 CODEC(T64),
    transactions_count UInt16 CODEC(T64),
    compute_units_used UInt32 CODEC(T64),
    priority_fees_collected UInt64 CODEC(T64);

-- Create projection for MEV profit analysis
ALTER TABLE mev_transactions ADD PROJECTION profit_analysis (
    SELECT 
        toStartOfHour(submitted_at) as hour,
        sum(profit_lamports) as total_profit,
        count() as transaction_count,
        avg(priority_fee) as avg_priority_fee,
        quantilesTDigest(0.5, 0.9, 0.99)(latency_ms) as latency_percentiles
    GROUP BY hour
);

-- Specialized table for bundle tracking with optimal codecs
CREATE TABLE IF NOT EXISTS jito_bundles (
    bundle_id FixedString(44) CODEC(ZSTD(3)),
    created_at DateTime64(3) CODEC(DoubleDelta),
    submitted_at DateTime64(3) CODEC(DoubleDelta),
    landed_slot Nullable(UInt64) CODEC(Delta),
    
    -- Bundle composition
    transaction_count UInt8 CODEC(T64),
    total_compute_units UInt32 CODEC(T64),
    tip_lamports UInt64 CODEC(T64),
    
    -- Auction data
    auction_slot UInt64 CODEC(Delta),
    winning_bid UInt64 CODEC(T64),
    our_bid UInt64 CODEC(T64),
    bid_percentile Float32 CODEC(Gorilla),
    
    -- Performance metrics
    simulation_success Bool CODEC(ZSTD(1)),
    landing_success Bool CODEC(ZSTD(1)),
    profit_realized Int64 CODEC(T64),
    
    -- Compressed bundle data
    bundle_data String CODEC(ZSTD(5)),  -- Higher compression for large data
    
    INDEX idx_bundle bundle_id TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_slot landed_slot TYPE minmax GRANULARITY 4096
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(created_at)
ORDER BY (created_at, bundle_id)
TTL created_at + INTERVAL 30 DAY;

-- Network topology table with geographic data
CREATE TABLE IF NOT EXISTS rpc_endpoints (
    endpoint_id String CODEC(ZSTD(3)),
    url String CODEC(ZSTD(3)),
    region LowCardinality(String),  -- Automatic dictionary encoding
    provider LowCardinality(String),
    
    -- Performance metrics (updated frequently)
    last_check DateTime CODEC(DoubleDelta),
    latency_ms Float32 CODEC(Gorilla),
    success_rate Float32 CODEC(Gorilla),
    rate_limit UInt32 CODEC(T64),
    
    -- Geographic coordinates
    latitude Float32 CODEC(Gorilla),
    longitude Float32 CODEC(Gorilla),
    
    INDEX idx_region region TYPE set(100) GRANULARITY 1
) ENGINE = ReplacingMergeTree(last_check)
ORDER BY endpoint_id;

-- Create dictionary for frequent lookups
CREATE DICTIONARY IF NOT EXISTS token_metadata (
    token_address FixedString(44),
    symbol String,
    decimals UInt8,
    name String
) PRIMARY KEY token_address
SOURCE(CLICKHOUSE(
    HOST 'localhost'
    PORT 9000
    DB 'default'
    TABLE 'token_info'
))
LIFETIME(MIN 3600 MAX 7200)
LAYOUT(HASHED());

-- Optimize existing tables
OPTIMIZE TABLE arbitrage_opportunities FINAL;
OPTIMIZE TABLE mev_transactions FINAL;
OPTIMIZE TABLE pool_states FINAL;

-- Create compression statistics view
CREATE OR REPLACE VIEW compression_stats AS
SELECT 
    table,
    formatReadableSize(sum(bytes)) AS size,
    formatReadableSize(sum(data_uncompressed_bytes)) AS uncompressed_size,
    round(sum(data_compressed_bytes) / sum(data_uncompressed_bytes) * 100, 2) AS compression_ratio,
    sum(rows) AS total_rows,
    round(sum(bytes) / sum(rows), 2) AS bytes_per_row
FROM system.parts
WHERE active
GROUP BY table
ORDER BY sum(bytes) DESC;

-- Add sampling for large tables
ALTER TABLE price_ticks MODIFY SETTING sampling_key = xxHash64(pool_address);

-- Enable adaptive index granularity
ALTER TABLE mev_transactions MODIFY SETTING enable_adaptive_granularity = 1;
ALTER TABLE pool_states MODIFY SETTING enable_adaptive_granularity = 1;