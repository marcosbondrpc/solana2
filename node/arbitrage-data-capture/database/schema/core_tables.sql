-- ClickHouse Core Tables for Arbitrage Data Capture
-- Optimized for time-series data with MergeTree engine

-- Main transactions table with comprehensive fields
CREATE TABLE IF NOT EXISTS arbitrage.transactions
(
    -- Primary identifiers
    transaction_hash String,
    block_number UInt64,
    block_timestamp DateTime64(9),
    
    -- Transaction details
    from_address String,
    to_address String,
    value Decimal128(18),
    gas_price UInt256,
    gas_used UInt64,
    max_fee_per_gas Nullable(UInt256),
    max_priority_fee_per_gas Nullable(UInt256),
    transaction_index UInt32,
    nonce UInt64,
    
    -- DEX and protocol information
    dex_name LowCardinality(String),
    protocol_version LowCardinality(String),
    pool_address String,
    token_in String,
    token_out String,
    amount_in Decimal128(18),
    amount_out Decimal128(18),
    
    -- Arbitrage specific fields
    is_arbitrage Bool,
    arbitrage_type LowCardinality(String),
    profit_usd Decimal64(8),
    profit_percentage Decimal32(4),
    path Array(String),
    path_length UInt8,
    
    -- MEV and competition metrics
    is_mev Bool,
    mev_type LowCardinality(String),
    bundle_index Nullable(UInt32),
    searcher_address Nullable(String),
    builder_address Nullable(String),
    validator_index Nullable(UInt32),
    
    -- Risk and performance metrics
    slippage_percentage Decimal32(4),
    price_impact Decimal32(4),
    gas_efficiency_score Decimal32(2),
    execution_time_ms UInt32,
    revert_probability Decimal32(4),
    
    -- Market conditions
    market_volatility Decimal32(4),
    liquidity_depth Decimal64(8),
    volume_24h Decimal64(8),
    
    -- Additional metadata
    input_data String,
    logs Array(String),
    receipt_status UInt8,
    cumulative_gas_used UInt64,
    effective_gas_price UInt256,
    
    -- ML features (pre-calculated)
    feature_vector Array(Float32),
    anomaly_score Float32,
    confidence_score Float32,
    
    -- Indexing and partitioning
    inserted_at DateTime DEFAULT now(),
    processing_time_ms UInt32,
    data_version UInt8 DEFAULT 1
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(block_timestamp)
ORDER BY (block_timestamp, transaction_hash)
PRIMARY KEY (block_timestamp, transaction_hash)
SETTINGS 
    index_granularity = 8192,
    compression_method = 'zstd',
    min_bytes_for_wide_part = 10485760,
    ttl_only_drop_parts = 1
TTL block_timestamp + INTERVAL 2 YEAR;

-- Arbitrage opportunities table
CREATE TABLE IF NOT EXISTS arbitrage.opportunities
(
    opportunity_id UUID DEFAULT generateUUIDv4(),
    detected_at DateTime64(9),
    block_number UInt64,
    
    -- Opportunity details
    type LowCardinality(String),
    status LowCardinality(String),
    
    -- Path information
    path_tokens Array(String),
    path_pools Array(String),
    path_dexes Array(String),
    
    -- Financial metrics
    expected_profit_usd Decimal64(8),
    actual_profit_usd Nullable(Decimal64(8)),
    required_capital Decimal64(8),
    roi_percentage Decimal32(4),
    
    -- Execution details
    gas_estimate UInt64,
    gas_price_gwei Decimal32(2),
    total_gas_cost_usd Decimal64(8),
    net_profit_usd Decimal64(8),
    
    -- Competition analysis
    competitors_detected UInt16,
    win_probability Decimal32(4),
    optimal_gas_multiplier Decimal32(2),
    
    -- Risk metrics
    impermanent_loss_risk Decimal32(4),
    slippage_risk Decimal32(4),
    frontrun_risk Decimal32(4),
    
    -- Execution result
    executed Bool DEFAULT false,
    execution_tx_hash Nullable(String),
    execution_timestamp Nullable(DateTime64(9)),
    execution_success Nullable(Bool),
    failure_reason Nullable(String)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(detected_at)
ORDER BY (detected_at, opportunity_id)
SETTINGS compression_method = 'zstd';

-- Risk metrics time series
CREATE TABLE IF NOT EXISTS arbitrage.risk_metrics
(
    timestamp DateTime64(9),
    metric_type LowCardinality(String),
    
    -- Market risk indicators
    volatility_1h Decimal32(4),
    volatility_24h Decimal32(4),
    correlation_matrix Array(Array(Float32)),
    
    -- Liquidity metrics
    total_liquidity_usd Decimal64(8),
    liquidity_concentration Decimal32(4),
    depth_imbalance Decimal32(4),
    
    -- MEV competition metrics
    avg_gas_price_gwei Decimal32(2),
    mev_competition_score Decimal32(2),
    bundle_success_rate Decimal32(4),
    
    -- System performance
    latency_p50_ms UInt32,
    latency_p95_ms UInt32,
    latency_p99_ms UInt32,
    
    -- Profit metrics
    total_profit_1h Decimal64(8),
    total_profit_24h Decimal64(8),
    profit_volatility Decimal32(4),
    sharpe_ratio Decimal32(4)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, metric_type)
TTL timestamp + INTERVAL 1 YEAR;

-- Market snapshots for ML training
CREATE TABLE IF NOT EXISTS arbitrage.market_snapshots
(
    snapshot_id UUID DEFAULT generateUUIDv4(),
    timestamp DateTime64(9),
    block_number UInt64,
    
    -- Price data
    prices Map(String, Decimal64(8)),
    
    -- Volume data
    volumes_1h Map(String, Decimal64(8)),
    volumes_24h Map(String, Decimal64(8)),
    
    -- Liquidity data
    liquidity_by_pool Map(String, Decimal64(8)),
    liquidity_by_token Map(String, Decimal64(8)),
    
    -- Order book data (aggregated)
    bid_depth Map(String, Array(Tuple(Decimal64(8), Decimal64(8)))),
    ask_depth Map(String, Array(Tuple(Decimal64(8), Decimal64(8)))),
    
    -- Network state
    gas_price UInt256,
    base_fee UInt256,
    pending_tx_count UInt32,
    mempool_size_mb Float32
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, snapshot_id)
SETTINGS compression_method = 'zstd';

-- Performance metrics aggregated
CREATE TABLE IF NOT EXISTS arbitrage.performance_metrics
(
    period_start DateTime,
    period_end DateTime,
    metric_type LowCardinality(String),
    
    -- Transaction metrics
    total_transactions UInt64,
    successful_transactions UInt64,
    failed_transactions UInt64,
    reverted_transactions UInt64,
    
    -- Profit metrics
    gross_profit_usd Decimal64(8),
    gas_costs_usd Decimal64(8),
    net_profit_usd Decimal64(8),
    roi_percentage Decimal32(4),
    
    -- Efficiency metrics
    avg_gas_used UInt64,
    avg_execution_time_ms UInt32,
    success_rate Decimal32(4),
    
    -- Competition metrics
    frontrun_count UInt32,
    backrun_count UInt32,
    sandwich_count UInt32,
    
    -- Statistical metrics
    profit_mean Decimal64(8),
    profit_std Decimal64(8),
    profit_min Decimal64(8),
    profit_max Decimal64(8),
    profit_percentiles Array(Decimal64(8))
)
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(period_start)
ORDER BY (period_start, metric_type);

-- Create materialized views for real-time aggregations
CREATE MATERIALIZED VIEW IF NOT EXISTS arbitrage.mv_hourly_stats
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(hour)
ORDER BY (hour, dex_name)
AS SELECT
    toStartOfHour(block_timestamp) AS hour,
    dex_name,
    count() AS tx_count,
    sum(profit_usd) AS total_profit,
    avg(profit_percentage) AS avg_profit_pct,
    max(profit_usd) AS max_profit,
    sum(gas_used * effective_gas_price) / 1e18 AS total_gas_eth
FROM arbitrage.transactions
WHERE is_arbitrage = true
GROUP BY hour, dex_name;

-- Create distributed tables for scaling
CREATE TABLE IF NOT EXISTS arbitrage.transactions_distributed AS arbitrage.transactions
ENGINE = Distributed('arbitrage_cluster', 'arbitrage', 'transactions', cityHash64(transaction_hash));

-- Indexes for high-speed lookups
ALTER TABLE arbitrage.transactions 
    ADD INDEX idx_arbitrage (is_arbitrage) TYPE minmax GRANULARITY 4,
    ADD INDEX idx_profit (profit_usd) TYPE minmax GRANULARITY 4,
    ADD INDEX idx_dex (dex_name) TYPE bloom_filter(0.01) GRANULARITY 1,
    ADD INDEX idx_tokens (token_in, token_out) TYPE bloom_filter(0.01) GRANULARITY 1;

-- Create buffer tables for high-throughput inserts
CREATE TABLE IF NOT EXISTS arbitrage.transactions_buffer AS arbitrage.transactions
ENGINE = Buffer('arbitrage', 'transactions', 16, 10, 100, 10000, 1000000, 10000000, 100000000);