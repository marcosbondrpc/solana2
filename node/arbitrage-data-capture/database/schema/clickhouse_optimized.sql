-- Elite ClickHouse Schema for Arbitrage Data Capture
-- Optimized for 100k+ TPS with sub-millisecond query performance

-- Create database with atomic engine for transactional consistency
CREATE DATABASE IF NOT EXISTS arbitrage_mainnet
ENGINE = Atomic;

USE arbitrage_mainnet;

-- Core transaction table with MergeTree for time-series optimization
CREATE TABLE IF NOT EXISTS transactions (
    -- Primary identifiers
    signature String CODEC(ZSTD(3)),
    block_height UInt64 CODEC(Delta, ZSTD(1)),
    block_timestamp DateTime64(3) CODEC(Delta, ZSTD(1)),
    slot UInt64 CODEC(Delta, ZSTD(1)),
    
    -- Transaction metadata
    fee UInt64 CODEC(ZSTD(1)),
    compute_units_used UInt32 CODEC(ZSTD(1)),
    priority_fee UInt64 CODEC(ZSTD(1)),
    lamports_per_signature UInt32,
    
    -- MEV specific fields
    is_mev_transaction Bool,
    mev_type Enum8('arbitrage' = 1, 'liquidation' = 2, 'sandwich' = 3, 'jit' = 4, 'cex_dex' = 5),
    bundle_id String CODEC(ZSTD(3)),
    searcher_address String CODEC(ZSTD(3)),
    
    -- Arbitrage details
    profit_amount Int64,
    profit_token String CODEC(ZSTD(3)),
    gas_cost UInt64,
    net_profit Int64,
    roi_percentage Float32,
    
    -- Path information
    dex_count UInt8,
    hop_count UInt8,
    path_hash String CODEC(ZSTD(3)),
    dexes Array(String) CODEC(ZSTD(3)),
    tokens Array(String) CODEC(ZSTD(3)),
    amounts Array(UInt64) CODEC(ZSTD(1)),
    
    -- Risk metrics
    slippage_percentage Float32,
    impermanent_loss Float32,
    max_drawdown Float32,
    sharpe_ratio Float32,
    volatility_score Float32,
    
    -- Market conditions
    market_volatility Float32,
    liquidity_depth UInt64,
    spread_basis_points UInt16,
    volume_24h UInt64,
    
    -- Performance metrics
    execution_time_ms UInt32,
    simulation_time_ms UInt32,
    mempool_time_ms UInt32,
    confirmation_time_ms UInt32,
    
    -- ML features (pre-computed)
    price_momentum Float32,
    volume_ratio Float32,
    liquidity_score Float32,
    market_impact Float32,
    cross_dex_correlation Float32,
    
    -- Additional metadata
    program_ids Array(String) CODEC(ZSTD(3)),
    instruction_count UInt16,
    cross_program_invocations UInt16,
    error_code Nullable(String),
    status Enum8('success' = 1, 'failed' = 2, 'partial' = 3),
    
    -- Indexing timestamp
    indexed_at DateTime64(3) DEFAULT now64(3)
) 
ENGINE = MergeTree()
PARTITION BY toYYYYMM(block_timestamp)
ORDER BY (block_timestamp, signature, mev_type)
PRIMARY KEY (block_timestamp, signature)
TTL block_timestamp + INTERVAL 2 YEAR
SETTINGS 
    index_granularity = 8192,
    merge_with_ttl_timeout = 86400,
    min_bytes_for_wide_part = 10485760,
    enable_mixed_granularity_parts = 1;

-- Create indices for fast lookups
ALTER TABLE transactions ADD INDEX idx_searcher searcher_address TYPE bloom_filter() GRANULARITY 4;
ALTER TABLE transactions ADD INDEX idx_profit net_profit TYPE minmax GRANULARITY 4;
ALTER TABLE transactions ADD INDEX idx_mev_type mev_type TYPE set(100) GRANULARITY 2;
ALTER TABLE transactions ADD INDEX idx_dexes dexes TYPE bloom_filter() GRANULARITY 4;
ALTER TABLE transactions ADD INDEX idx_tokens tokens TYPE bloom_filter() GRANULARITY 4;

-- Arbitrage opportunities table
CREATE TABLE IF NOT EXISTS arbitrage_opportunities (
    opportunity_id String CODEC(ZSTD(3)),
    detected_at DateTime64(6) CODEC(Delta, ZSTD(1)),
    block_height UInt64 CODEC(Delta, ZSTD(1)),
    
    -- Opportunity details
    opportunity_type Enum8('spot' = 1, 'triangle' = 2, 'multi_hop' = 3, 'cross_chain' = 4),
    input_token String CODEC(ZSTD(3)),
    output_token String CODEC(ZSTD(3)),
    input_amount UInt64,
    expected_output UInt64,
    minimum_profit UInt64,
    
    -- Path details
    path_json String CODEC(ZSTD(5)),
    dex_sequence Array(String) CODEC(ZSTD(3)),
    pool_addresses Array(String) CODEC(ZSTD(3)),
    
    -- Market data snapshot
    pool_reserves Array(Tuple(UInt64, UInt64)) CODEC(ZSTD(1)),
    pool_fees Array(UInt16),
    price_impacts Array(Float32),
    
    -- Execution details
    executed Bool DEFAULT 0,
    execution_tx String CODEC(ZSTD(3)),
    actual_profit Nullable(Int64),
    execution_latency_ms Nullable(UInt32),
    
    -- Competition metrics
    competing_txs UInt16,
    frontrun_attempts UInt8,
    backrun_success Bool,
    
    -- Risk assessment
    confidence_score Float32,
    risk_score Float32,
    profitability_score Float32,
    
    INDEX idx_opportunity_type opportunity_type TYPE set(10) GRANULARITY 2,
    INDEX idx_executed executed TYPE set(2) GRANULARITY 1
) 
ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(detected_at)
ORDER BY (detected_at, opportunity_id)
TTL detected_at + INTERVAL 6 MONTH;

-- High-frequency market snapshots
CREATE TABLE IF NOT EXISTS market_snapshots (
    snapshot_time DateTime64(3) CODEC(Delta, ZSTD(1)),
    dex String CODEC(ZSTD(3)),
    pool_address String CODEC(ZSTD(3)),
    
    -- Liquidity metrics
    reserve0 UInt64 CODEC(ZSTD(1)),
    reserve1 UInt64 CODEC(ZSTD(1)),
    total_liquidity UInt64 CODEC(ZSTD(1)),
    
    -- Price data
    price Float64,
    price_change_1m Float32,
    price_change_5m Float32,
    price_change_1h Float32,
    
    -- Volume metrics
    volume_1m UInt64,
    volume_5m UInt64,
    volume_1h UInt64,
    trade_count_1m UInt32,
    
    -- Market depth
    bid_liquidity Array(Tuple(Float64, UInt64)) CODEC(ZSTD(3)),
    ask_liquidity Array(Tuple(Float64, UInt64)) CODEC(ZSTD(3)),
    spread_bps UInt16,
    
    INDEX idx_pool pool_address TYPE bloom_filter() GRANULARITY 4
) 
ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(snapshot_time)
ORDER BY (snapshot_time, dex, pool_address)
TTL snapshot_time + INTERVAL 30 DAY;

-- Performance metrics aggregation
CREATE TABLE IF NOT EXISTS performance_metrics (
    metric_date Date,
    metric_hour DateTime,
    
    -- Transaction metrics
    total_transactions UInt64,
    successful_arbitrages UInt64,
    failed_arbitrages UInt64,
    
    -- Profit metrics
    total_profit_usd Float64,
    average_profit_usd Float32,
    median_profit_usd Float32,
    max_profit_usd Float32,
    
    -- Gas metrics
    total_gas_used UInt64,
    average_gas_price Float32,
    gas_efficiency_ratio Float32,
    
    -- Latency metrics
    avg_execution_time_ms Float32,
    p50_execution_time_ms UInt32,
    p95_execution_time_ms UInt32,
    p99_execution_time_ms UInt32,
    
    -- Competition metrics
    frontrun_rate Float32,
    success_rate Float32,
    bundle_inclusion_rate Float32
) 
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(metric_date)
ORDER BY (metric_date, metric_hour);

-- Materialized view for real-time aggregations
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_hourly_stats
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(hour_bucket)
ORDER BY (hour_bucket, mev_type)
AS SELECT
    toStartOfHour(block_timestamp) as hour_bucket,
    mev_type,
    count() as tx_count,
    sum(net_profit) as total_profit,
    avg(net_profit) as avg_profit,
    max(net_profit) as max_profit,
    avg(execution_time_ms) as avg_execution_ms,
    sum(gas_cost) as total_gas,
    avg(roi_percentage) as avg_roi,
    uniq(searcher_address) as unique_searchers
FROM transactions
WHERE status = 'success'
GROUP BY hour_bucket, mev_type;

-- Distributed table for horizontal scaling
CREATE TABLE IF NOT EXISTS transactions_distributed AS transactions
ENGINE = Distributed('arbitrage_cluster', 'arbitrage_mainnet', 'transactions', rand());

-- Buffer table for high-throughput inserts
CREATE TABLE IF NOT EXISTS transactions_buffer AS transactions
ENGINE = Buffer('arbitrage_mainnet', 'transactions', 16, 10, 100, 10000, 1000000, 10000000, 100000000);

-- Create projections for common queries
ALTER TABLE transactions ADD PROJECTION projection_by_searcher (
    SELECT 
        searcher_address,
        block_timestamp,
        signature,
        net_profit,
        mev_type,
        roi_percentage
    ORDER BY searcher_address, block_timestamp
);

ALTER TABLE transactions ADD PROJECTION projection_by_profit (
    SELECT 
        block_timestamp,
        signature,
        net_profit,
        roi_percentage,
        mev_type,
        searcher_address
    ORDER BY net_profit DESC, block_timestamp
);

-- Settings for optimal performance
ALTER TABLE transactions MODIFY SETTING 
    max_parts_in_total = 100000,
    parts_to_delay_insert = 5000,
    parts_to_throw_insert = 10000;

-- Create dictionary for token metadata
CREATE DICTIONARY IF NOT EXISTS token_metadata (
    token_address String,
    symbol String,
    decimals UInt8,
    name String,
    price_usd Float64
)
PRIMARY KEY token_address
SOURCE(HTTP(url 'http://api-service:8080/tokens' format 'JSONEachRow'))
LIFETIME(MIN 300 MAX 600)
LAYOUT(HASHED());