-- ClickHouse Schema for Solana Arbitrage Data Capture
-- Optimized for minimal disk usage with ZSTD compression

-- Create database for arbitrage data
CREATE DATABASE IF NOT EXISTS solana_arbitrage;

USE solana_arbitrage;

-- Main transactions table with optimal compression
CREATE TABLE IF NOT EXISTS transactions
(
    -- Primary identifiers
    tx_signature String CODEC(ZSTD(3)),
    block_time DateTime CODEC(DoubleDelta, ZSTD(1)),
    slot UInt64 CODEC(DoubleDelta, ZSTD(1)),
    block_height UInt64 CODEC(DoubleDelta, ZSTD(1)),
    
    -- Transaction details
    signer String CODEC(ZSTD(3)),
    program String CODEC(ZSTD(3)),
    instruction_type String CODEC(ZSTD(3)),
    
    -- Arbitrage path information
    path Array(String) CODEC(ZSTD(3)),
    dex_sequence Array(String) CODEC(ZSTD(3)),
    token_in String CODEC(ZSTD(3)),
    token_out String CODEC(ZSTD(3)),
    amount_in Decimal64(8) CODEC(Gorilla, ZSTD(1)),
    amount_out Decimal64(8) CODEC(Gorilla, ZSTD(1)),
    
    -- Financial metrics
    revenue_sol Decimal64(9) CODEC(Gorilla, ZSTD(1)),
    costs_gas_sol Decimal64(9) CODEC(Gorilla, ZSTD(1)),
    costs_priority_fee_sol Decimal64(9) CODEC(Gorilla, ZSTD(1)),
    costs_total_sol Decimal64(9) CODEC(Gorilla, ZSTD(1)),
    net_profit_sol Decimal64(9) CODEC(Gorilla, ZSTD(1)),
    roi Float32 CODEC(Gorilla, ZSTD(1)),
    
    -- MEV specific fields
    mev_type Enum8('arbitrage' = 1, 'sandwich' = 2, 'jit' = 3, 'liquidation' = 4, 'other' = 5),
    bundle_id String CODEC(ZSTD(3)),
    bundle_landed UInt8,
    jito_tip Decimal64(9) CODEC(Gorilla, ZSTD(1)),
    
    -- Market conditions
    market_volatility Float32 CODEC(Gorilla, ZSTD(1)),
    liquidity_depth Decimal64(8) CODEC(Gorilla, ZSTD(1)),
    slippage_pct Float32 CODEC(Gorilla, ZSTD(1)),
    
    -- Performance metrics
    latency_ms UInt32 CODEC(DoubleDelta, ZSTD(1)),
    parse_time_us UInt32 CODEC(DoubleDelta, ZSTD(1)),
    execution_time_ms UInt32 CODEC(DoubleDelta, ZSTD(1)),
    
    -- Risk metrics (stored as JSON for flexibility)
    risk_score Float32 CODEC(Gorilla, ZSTD(1)),
    risk_factors String CODEC(ZSTD(3)), -- JSON string
    
    -- Classification
    label_is_arb UInt8,
    confidence_score Float32 CODEC(Gorilla, ZSTD(1)),
    strategy_type String CODEC(ZSTD(3)),
    
    -- Metadata
    inserted_at DateTime DEFAULT now(),
    data_version UInt8 DEFAULT 1
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(block_time)
ORDER BY (block_time, slot, tx_signature)
SETTINGS 
    index_granularity = 8192,
    min_bytes_for_wide_part = 10485760,
    min_rows_for_wide_part = 10000,
    merge_max_block_size = 8192;

-- Create materialized view for hourly aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_stats
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(hour)
ORDER BY (hour, mev_type, strategy_type)
AS SELECT
    toStartOfHour(block_time) as hour,
    mev_type,
    strategy_type,
    count() as tx_count,
    sum(net_profit_sol) as total_profit,
    avg(roi) as avg_roi,
    max(net_profit_sol) as max_profit,
    min(net_profit_sol) as min_profit,
    avg(latency_ms) as avg_latency,
    sum(jito_tip) as total_tips,
    sumIf(1, bundle_landed = 1) as bundles_landed,
    avgIf(net_profit_sol, label_is_arb = 1) as avg_arb_profit
FROM transactions
GROUP BY hour, mev_type, strategy_type;

-- Create table for DEX pool states
CREATE TABLE IF NOT EXISTS dex_pool_states
(
    timestamp DateTime CODEC(DoubleDelta, ZSTD(1)),
    pool_address String CODEC(ZSTD(3)),
    dex_name String CODEC(ZSTD(3)),
    token_a String CODEC(ZSTD(3)),
    token_b String CODEC(ZSTD(3)),
    reserve_a Decimal128(18) CODEC(Gorilla, ZSTD(1)),
    reserve_b Decimal128(18) CODEC(Gorilla, ZSTD(1)),
    price_a_b Float64 CODEC(Gorilla, ZSTD(1)),
    liquidity_usd Decimal64(2) CODEC(Gorilla, ZSTD(1)),
    volume_24h Decimal64(2) CODEC(Gorilla, ZSTD(1)),
    fee_tier UInt16 CODEC(DoubleDelta, ZSTD(1))
)
ENGINE = ReplacingMergeTree(timestamp)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (pool_address, timestamp)
SETTINGS
    index_granularity = 8192;

-- Create table for signer reputation tracking
CREATE TABLE IF NOT EXISTS signer_reputation
(
    signer String CODEC(ZSTD(3)),
    date Date,
    total_txs UInt64,
    successful_arbs UInt64,
    failed_arbs UInt64,
    total_profit Decimal64(9),
    avg_roi Float32,
    reputation_score Float32
)
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, signer);

-- Create Kafka engine table for real-time ingestion
CREATE TABLE IF NOT EXISTS kafka_transactions_queue
(
    data String
)
ENGINE = Kafka()
SETTINGS
    kafka_broker_list = 'localhost:9092',
    kafka_topic_list = 'solana-transactions',
    kafka_group_name = 'clickhouse-consumer',
    kafka_format = 'JSONAsString',
    kafka_num_consumers = 2,
    kafka_max_block_size = 1048576;

-- Create materialized view to parse Kafka data and insert into main table
CREATE MATERIALIZED VIEW IF NOT EXISTS kafka_transactions_consumer TO transactions AS
SELECT
    JSONExtractString(data, 'tx_signature') as tx_signature,
    parseDateTimeBestEffort(JSONExtractString(data, 'block_time')) as block_time,
    JSONExtractUInt(data, 'slot') as slot,
    JSONExtractUInt(data, 'block_height') as block_height,
    JSONExtractString(data, 'signer') as signer,
    JSONExtractString(data, 'program') as program,
    JSONExtractString(data, 'instruction_type') as instruction_type,
    JSONExtractArrayRaw(data, 'path') as path,
    JSONExtractArrayRaw(data, 'dex_sequence') as dex_sequence,
    JSONExtractString(data, 'token_in') as token_in,
    JSONExtractString(data, 'token_out') as token_out,
    JSONExtractFloat(data, 'amount_in') as amount_in,
    JSONExtractFloat(data, 'amount_out') as amount_out,
    JSONExtractFloat(data, 'revenue_sol') as revenue_sol,
    JSONExtractFloat(data, 'costs_gas_sol') as costs_gas_sol,
    JSONExtractFloat(data, 'costs_priority_fee_sol') as costs_priority_fee_sol,
    JSONExtractFloat(data, 'costs_total_sol') as costs_total_sol,
    JSONExtractFloat(data, 'net_profit_sol') as net_profit_sol,
    JSONExtractFloat(data, 'roi') as roi,
    JSONExtractUInt(data, 'mev_type') as mev_type,
    JSONExtractString(data, 'bundle_id') as bundle_id,
    JSONExtractUInt(data, 'bundle_landed') as bundle_landed,
    JSONExtractFloat(data, 'jito_tip') as jito_tip,
    JSONExtractFloat(data, 'market_volatility') as market_volatility,
    JSONExtractFloat(data, 'liquidity_depth') as liquidity_depth,
    JSONExtractFloat(data, 'slippage_pct') as slippage_pct,
    JSONExtractUInt(data, 'latency_ms') as latency_ms,
    JSONExtractUInt(data, 'parse_time_us') as parse_time_us,
    JSONExtractUInt(data, 'execution_time_ms') as execution_time_ms,
    JSONExtractFloat(data, 'risk_score') as risk_score,
    JSONExtractString(data, 'risk_factors') as risk_factors,
    JSONExtractUInt(data, 'label_is_arb') as label_is_arb,
    JSONExtractFloat(data, 'confidence_score') as confidence_score,
    JSONExtractString(data, 'strategy_type') as strategy_type
FROM kafka_transactions_queue;

-- Create indexes for optimal query performance
ALTER TABLE transactions ADD INDEX idx_signer (signer) TYPE bloom_filter(0.01) GRANULARITY 4;
ALTER TABLE transactions ADD INDEX idx_mev_type (mev_type) TYPE set(100) GRANULARITY 4;
ALTER TABLE transactions ADD INDEX idx_profit (net_profit_sol) TYPE minmax GRANULARITY 4;
ALTER TABLE transactions ADD INDEX idx_strategy (strategy_type) TYPE bloom_filter(0.01) GRANULARITY 4;

-- Create projection for fast arbitrage analysis
ALTER TABLE transactions ADD PROJECTION arb_analysis
(
    SELECT 
        block_time,
        tx_signature,
        path,
        net_profit_sol,
        roi,
        latency_ms
    ORDER BY net_profit_sol DESC
);

-- Optimize table settings for compression
ALTER TABLE transactions MODIFY SETTING 
    min_compress_block_size = 65536,
    max_compress_block_size = 1048576;

-- Create a view for real-time dashboard
CREATE VIEW IF NOT EXISTS dashboard_realtime AS
SELECT
    toStartOfMinute(block_time) as minute,
    count() as tx_count,
    sum(net_profit_sol) as total_profit,
    avg(roi) * 100 as avg_roi_pct,
    max(net_profit_sol) as best_arb,
    avg(latency_ms) as avg_latency,
    sumIf(1, net_profit_sol > 0) as profitable_txs,
    sumIf(1, net_profit_sol <= 0) as unprofitable_txs
FROM transactions
WHERE block_time >= now() - INTERVAL 1 HOUR
GROUP BY minute
ORDER BY minute DESC;

-- Compression statistics query (run after data is loaded)
-- SELECT 
--     table,
--     formatReadableSize(sum(data_compressed_bytes)) as compressed_size,
--     formatReadableSize(sum(data_uncompressed_bytes)) as uncompressed_size,
--     sum(data_uncompressed_bytes) / sum(data_compressed_bytes) as compression_ratio
-- FROM system.parts
-- WHERE database = 'solana_arbitrage'
-- GROUP BY table;