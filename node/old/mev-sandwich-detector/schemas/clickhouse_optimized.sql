-- MEV Sandwich Detector ClickHouse Schema
-- Optimized for 200k+ rows/second throughput with compression

-- Create database
CREATE DATABASE IF NOT EXISTS mev_sandwich
ENGINE = Atomic;

USE mev_sandwich;

-- Main decisions table with optimal compression
CREATE TABLE IF NOT EXISTS mev_sandwich (
    timestamp DateTime64(9) CODEC(DoubleDelta, ZSTD(3)),
    decision_time_us UInt32 CODEC(T64, ZSTD(3)),
    
    -- Transaction identifiers
    target_tx String CODEC(ZSTD(3)),
    front_tx String CODEC(ZSTD(3)),
    back_tx String CODEC(ZSTD(3)),
    
    -- Financial metrics
    expected_profit UInt64 CODEC(T64, ZSTD(3)),
    gas_cost UInt64 CODEC(T64, ZSTD(3)),
    net_profit Int64 CODEC(T64, ZSTD(3)),
    tip_amount UInt64 CODEC(T64, ZSTD(3)),
    
    -- ML metrics
    confidence Float32 CODEC(Gorilla, ZSTD(3)),
    features String CODEC(ZSTD(5)), -- JSON features
    
    -- Submission metadata
    submission_path Enum8('tpu' = 1, 'jito' = 2, 'both' = 3) CODEC(T64),
    priority UInt8 CODEC(T64, ZSTD(3)),
    
    -- Outcome tracking
    landed Bool DEFAULT 0,
    actual_profit Nullable(Int64) CODEC(T64, ZSTD(3)),
    landing_slot Nullable(UInt64) CODEC(T64, ZSTD(3)),
    
    -- Partitioning helper
    date Date DEFAULT toDate(timestamp) CODEC(DoubleDelta, ZSTD(3))
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, timestamp, target_tx)
TTL date + INTERVAL 90 DAY
SETTINGS 
    index_granularity = 8192,
    index_granularity_bytes = 10485760,
    min_bytes_for_wide_part = 10485760,
    min_rows_for_wide_part = 0,
    compress_on_write = 1,
    enable_mixed_granularity_parts = 1;

-- Bundle outcomes table for tracking landed bundles
CREATE TABLE IF NOT EXISTS bundle_outcomes (
    timestamp DateTime64(9) CODEC(DoubleDelta, ZSTD(3)),
    tx_hash String CODEC(ZSTD(3)),
    landed Bool,
    actual_profit Nullable(UInt64) CODEC(T64, ZSTD(3)),
    landing_slot UInt64 CODEC(T64, ZSTD(3)),
    submission_path Enum8('tpu' = 1, 'jito' = 2) CODEC(T64),
    tip_paid UInt64 CODEC(T64, ZSTD(3)),
    gas_used UInt64 CODEC(T64, ZSTD(3)),
    
    date Date DEFAULT toDate(timestamp) CODEC(DoubleDelta, ZSTD(3))
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, timestamp, tx_hash)
TTL date + INTERVAL 30 DAY
SETTINGS index_granularity = 8192;

-- Kafka streaming table for real-time decisions
CREATE TABLE IF NOT EXISTS kafka_decisions (
    timestamp UInt64,
    target_tx String,
    expected_profit UInt64,
    gas_cost UInt64,
    confidence Float32,
    tip_amount UInt64,
    features String
)
ENGINE = Kafka()
SETTINGS
    kafka_broker_list = 'localhost:9092',
    kafka_topic_list = 'sandwich-decisions',
    kafka_group_name = 'clickhouse-consumer',
    kafka_format = 'JSONEachRow',
    kafka_num_consumers = 4,
    kafka_max_block_size = 1048576;

-- Materialized view for Kafka streaming
CREATE MATERIALIZED VIEW IF NOT EXISTS kafka_decisions_mv TO mev_sandwich AS
SELECT
    toDateTime64(timestamp / 1000000, 9) AS timestamp,
    toUInt32(timestamp % 1000000) AS decision_time_us,
    target_tx,
    '' AS front_tx,  -- Will be updated later
    '' AS back_tx,   -- Will be updated later
    expected_profit,
    gas_cost,
    toInt64(expected_profit) - toInt64(gas_cost) AS net_profit,
    tip_amount,
    confidence,
    features,
    'both' AS submission_path,
    toUInt8(confidence * 255) AS priority,
    0 AS landed,
    NULL AS actual_profit,
    NULL AS landing_slot,
    toDate(toDateTime64(timestamp / 1000000, 9)) AS date
FROM kafka_decisions;

-- Performance metrics aggregation table
CREATE TABLE IF NOT EXISTS performance_metrics (
    timestamp DateTime CODEC(DoubleDelta, ZSTD(3)),
    
    -- Volume metrics
    decisions_count UInt32 CODEC(T64, ZSTD(3)),
    bundles_submitted UInt32 CODEC(T64, ZSTD(3)),
    bundles_landed UInt32 CODEC(T64, ZSTD(3)),
    
    -- Latency percentiles (microseconds)
    decision_p50 UInt32 CODEC(T64, ZSTD(3)),
    decision_p95 UInt32 CODEC(T64, ZSTD(3)),
    decision_p99 UInt32 CODEC(T64, ZSTD(3)),
    
    -- Success rates
    tpu_success_rate Float32 CODEC(Gorilla, ZSTD(3)),
    jito_success_rate Float32 CODEC(Gorilla, ZSTD(3)),
    overall_success_rate Float32 CODEC(Gorilla, ZSTD(3)),
    
    -- Financial metrics
    total_profit Int64 CODEC(T64, ZSTD(3)),
    total_gas_cost UInt64 CODEC(T64, ZSTD(3)),
    net_profit Int64 CODEC(T64, ZSTD(3)),
    avg_profit_per_bundle Float64 CODEC(Gorilla, ZSTD(3)),
    
    date Date DEFAULT toDate(timestamp) CODEC(DoubleDelta, ZSTD(3))
)
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, timestamp)
TTL date + INTERVAL 180 DAY;

-- Materialized view for 1-minute performance rollups
CREATE MATERIALIZED VIEW IF NOT EXISTS performance_1m_mv TO performance_metrics AS
SELECT
    toStartOfMinute(timestamp) AS timestamp,
    count() AS decisions_count,
    countIf(submission_path != 'none') AS bundles_submitted,
    countIf(landed = 1) AS bundles_landed,
    
    quantile(0.5)(decision_time_us) AS decision_p50,
    quantile(0.95)(decision_time_us) AS decision_p95,
    quantile(0.99)(decision_time_us) AS decision_p99,
    
    -- These would be calculated from submission stats
    0.0 AS tpu_success_rate,
    0.0 AS jito_success_rate,
    avg(toUInt8(landed)) AS overall_success_rate,
    
    sum(net_profit) AS total_profit,
    sum(gas_cost) AS total_gas_cost,
    sum(net_profit) AS net_profit,
    avgIf(net_profit, landed = 1) AS avg_profit_per_bundle,
    
    toDate(timestamp) AS date
FROM mev_sandwich
GROUP BY toStartOfMinute(timestamp);

-- Feature statistics for ML training
CREATE TABLE IF NOT EXISTS feature_stats (
    date Date CODEC(DoubleDelta, ZSTD(3)),
    feature_name String CODEC(ZSTD(3)),
    
    -- Statistical measures
    mean Float64 CODEC(Gorilla, ZSTD(3)),
    stddev Float64 CODEC(Gorilla, ZSTD(3)),
    min Float64 CODEC(Gorilla, ZSTD(3)),
    max Float64 CODEC(Gorilla, ZSTD(3)),
    p25 Float64 CODEC(Gorilla, ZSTD(3)),
    p50 Float64 CODEC(Gorilla, ZSTD(3)),
    p75 Float64 CODEC(Gorilla, ZSTD(3)),
    
    -- Correlation with profit
    profit_correlation Float32 CODEC(Gorilla, ZSTD(3)),
    
    sample_count UInt64 CODEC(T64, ZSTD(3))
)
ENGINE = ReplacingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, feature_name);

-- Training dataset view for ML pipeline
CREATE VIEW IF NOT EXISTS ml_training_dataset AS
SELECT
    timestamp,
    decision_time_us,
    
    -- Features (extracted from JSON)
    JSONExtractFloat(features, 'input') AS input_amount,
    JSONExtractFloat(features, 'output') AS output_amount,
    JSONExtractFloat(features, 'gas_price') AS gas_price,
    JSONExtractFloat(features, 'slippage') AS slippage,
    JSONExtractUInt(features, 'depth') AS mempool_depth,
    
    -- Target variables
    landed AS label,
    net_profit AS profit,
    confidence,
    
    -- Context
    submission_path,
    priority,
    tip_amount
FROM mev_sandwich
WHERE landed IS NOT NULL  -- Only labeled data
    AND date >= today() - 7  -- Last 7 days
SETTINGS max_threads = 8;

-- Indexes for common queries
ALTER TABLE mev_sandwich ADD INDEX idx_confidence confidence TYPE minmax GRANULARITY 4;
ALTER TABLE mev_sandwich ADD INDEX idx_profit net_profit TYPE minmax GRANULARITY 4;
ALTER TABLE mev_sandwich ADD INDEX idx_landed landed TYPE set(2) GRANULARITY 8192;

-- Create dictionary for fast lookups
CREATE DICTIONARY IF NOT EXISTS bundle_lookup (
    tx_hash String,
    landed UInt8,
    actual_profit Int64,
    landing_slot UInt64
)
PRIMARY KEY tx_hash
SOURCE(CLICKHOUSE(
    HOST 'localhost'
    PORT 9000
    USER 'default'
    TABLE 'bundle_outcomes'
    DB 'mev_sandwich'
))
LIFETIME(MIN 60 MAX 300)
LAYOUT(HASHED());

-- Optimization settings
SYSTEM RELOAD DICTIONARIES;

-- Create projections for common queries
ALTER TABLE mev_sandwich ADD PROJECTION profit_analysis (
    SELECT 
        date,
        toStartOfHour(timestamp) AS hour,
        sum(net_profit) AS total_profit,
        count() AS trades,
        avg(confidence) AS avg_confidence
    GROUP BY date, hour
);

ALTER TABLE mev_sandwich ADD PROJECTION landing_stats (
    SELECT
        date,
        submission_path,
        countIf(landed = 1) AS landed_count,
        count() AS total_count,
        avg(tip_amount) AS avg_tip
    GROUP BY date, submission_path
);

-- Materialize projections
ALTER TABLE mev_sandwich MATERIALIZE PROJECTION profit_analysis;
ALTER TABLE mev_sandwich MATERIALIZE PROJECTION landing_stats;

-- Grant permissions for high-throughput access
GRANT SELECT, INSERT ON mev_sandwich.* TO default;

-- Optimize table settings for write performance
OPTIMIZE TABLE mev_sandwich FINAL;
OPTIMIZE TABLE bundle_outcomes FINAL;