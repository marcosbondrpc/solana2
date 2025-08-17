-- MEV Sandwich Detector ClickHouse Schema
-- Optimized for 200k+ rows/s write throughput

-- Create database
CREATE DATABASE IF NOT EXISTS mev_sandwich_db
ENGINE = Atomic;

USE mev_sandwich_db;

-- Main sandwich decisions table with ZSTD compression
CREATE TABLE IF NOT EXISTS mev_sandwich (
    -- Timestamp and performance
    timestamp DateTime64(6) DEFAULT now64(6),
    decision_time_us UInt32,
    
    -- Transaction identifiers
    target_tx String,
    front_tx String,
    back_tx String,
    
    -- Financial metrics
    expected_profit UInt64,
    gas_cost UInt64,
    net_profit Int64 MATERIALIZED (toInt64(expected_profit) - toInt64(gas_cost)),
    tip_amount UInt64,
    
    -- ML metrics
    confidence Float32,
    
    -- Features as JSON for flexibility
    features String CODEC(ZSTD(6)),
    
    -- Indexing
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 8192,
    INDEX idx_profit net_profit TYPE minmax GRANULARITY 8192,
    INDEX idx_confidence confidence TYPE minmax GRANULARITY 1024
)
ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (timestamp, target_tx)
TTL timestamp + INTERVAL 30 DAY
SETTINGS index_granularity = 8192,
         min_bytes_for_wide_part = 104857600,  -- 100MB
         write_ahead_log_max_bytes = 1073741824; -- 1GB WAL

-- Bundle outcomes tracking
CREATE TABLE IF NOT EXISTS bundle_outcomes (
    timestamp DateTime64(6) DEFAULT now64(6),
    tx_hash String,
    bundle_hash String,
    
    -- Submission details
    submitted_to Enum8('tpu' = 1, 'jito' = 2, 'both' = 3),
    tip_amount UInt64,
    priority UInt8,
    
    -- Outcome
    landed Bool,
    block_number Nullable(UInt64),
    actual_profit Nullable(Int64),
    landing_time_ms Nullable(UInt32),
    
    -- Competition
    competing_bundles UInt16 DEFAULT 0,
    won_auction Bool DEFAULT false,
    
    INDEX idx_landed landed TYPE set(2) GRANULARITY 4096,
    INDEX idx_tx_hash tx_hash TYPE bloom_filter GRANULARITY 1
)
ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (timestamp, tx_hash)
TTL timestamp + INTERVAL 7 DAY;

-- Aggregated metrics for monitoring (1-minute granularity)
CREATE TABLE IF NOT EXISTS metrics_1m (
    timestamp DateTime64(6),
    
    -- Volume metrics
    decisions_count UInt64,
    bundles_submitted UInt64,
    bundles_landed UInt64,
    
    -- Performance metrics
    avg_decision_time_us Float64,
    p50_decision_time_us UInt32,
    p99_decision_time_us UInt32,
    
    -- Financial metrics
    total_expected_profit UInt64,
    total_actual_profit Int64,
    total_gas_cost UInt64,
    total_tips_paid UInt64,
    
    -- Success rates
    landing_rate Float32,
    tpu_success_rate Float32,
    jito_success_rate Float32,
    
    -- ML metrics
    avg_confidence Float32
)
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY timestamp
TTL timestamp + INTERVAL 7 DAY;

-- Kafka integration for real-time ingestion
CREATE TABLE IF NOT EXISTS kafka_sandwich_raw (
    timestamp UInt64,
    data String
)
ENGINE = Kafka()
SETTINGS 
    kafka_broker_list = 'localhost:9092',
    kafka_topic_list = 'sandwich-raw',
    kafka_group_name = 'clickhouse-consumer',
    kafka_format = 'JSONAsString',
    kafka_max_block_size = 1048576;

-- Materialized view for Kafka → ClickHouse
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_sandwich_raw TO mev_sandwich AS
SELECT
    toDateTime64(JSONExtractUInt(data, 'timestamp') / 1000000, 6) as timestamp,
    JSONExtractUInt(data, 'decision_time_us') as decision_time_us,
    JSONExtractString(data, 'target_tx') as target_tx,
    JSONExtractString(data, 'front_tx') as front_tx,
    JSONExtractString(data, 'back_tx') as back_tx,
    JSONExtractUInt(data, 'expected_profit') as expected_profit,
    JSONExtractUInt(data, 'gas_cost') as gas_cost,
    JSONExtractUInt(data, 'tip_amount') as tip_amount,
    JSONExtractFloat(data, 'confidence') as confidence,
    JSONExtractRaw(data, 'features') as features
FROM kafka_sandwich_raw;

-- Performance monitoring table
CREATE TABLE IF NOT EXISTS system_performance (
    timestamp DateTime64(6) DEFAULT now64(6),
    
    -- System metrics
    cpu_usage Float32,
    memory_usage_gb Float32,
    network_rx_mbps Float32,
    network_tx_mbps Float32,
    
    -- Application metrics
    active_connections UInt16,
    packet_queue_depth UInt32,
    decision_queue_depth UInt32,
    
    -- Latency percentiles (microseconds)
    network_p50 UInt32,
    network_p99 UInt32,
    ml_inference_p50 UInt32,
    ml_inference_p99 UInt32,
    submission_p50 UInt32,
    submission_p99 UInt32,
    
    -- Throughput
    packets_per_second UInt32,
    decisions_per_second UInt32,
    writes_per_second UInt32
)
ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY timestamp
TTL timestamp + INTERVAL 3 DAY;

-- Create buffer tables for ultra-high throughput
CREATE TABLE IF NOT EXISTS mev_sandwich_buffer AS mev_sandwich
ENGINE = Buffer(
    mev_sandwich_db, mev_sandwich,
    16,     -- num_layers
    10,     -- min_time (seconds)
    100,    -- max_time
    100000, -- min_rows
    1000000,-- max_rows
    10000000,   -- min_bytes (10MB)
    100000000   -- max_bytes (100MB)
);

-- Statistics and reporting views
CREATE VIEW IF NOT EXISTS hourly_stats AS
SELECT
    toStartOfHour(timestamp) as hour,
    count() as total_decisions,
    sum(landed) as total_landed,
    avg(confidence) as avg_confidence,
    sum(expected_profit) / 1e9 as expected_profit_sol,
    sum(actual_profit) / 1e9 as actual_profit_sol,
    sum(tip_amount) / 1e9 as tips_paid_sol,
    avg(decision_time_us) as avg_decision_us,
    quantile(0.5)(decision_time_us) as p50_decision_us,
    quantile(0.99)(decision_time_us) as p99_decision_us,
    sum(landed) / count() as landing_rate
FROM mev_sandwich s
LEFT JOIN bundle_outcomes o ON s.target_tx = o.tx_hash
WHERE timestamp >= now() - INTERVAL 24 HOUR
GROUP BY hour
ORDER BY hour DESC;

-- Profitability analysis view
CREATE VIEW IF NOT EXISTS profitability_analysis AS
SELECT
    toDate(timestamp) as date,
    count() as trades,
    sum(landed) as successful_trades,
    sum(actual_profit) / 1e9 as total_profit_sol,
    sum(tip_amount) / 1e9 as total_tips_sol,
    sum(gas_cost) / 1e9 as total_gas_sol,
    (sum(actual_profit) - sum(tip_amount) - sum(gas_cost)) / 1e9 as net_profit_sol,
    avg(confidence) as avg_confidence,
    sum(landed) / count() as success_rate
FROM mev_sandwich s
JOIN bundle_outcomes o ON s.target_tx = o.tx_hash
WHERE timestamp >= now() - INTERVAL 30 DAY
GROUP BY date
ORDER BY date DESC;

-- Indexes for fast queries
ALTER TABLE mev_sandwich ADD INDEX idx_target_tx (target_tx) TYPE bloom_filter GRANULARITY 1;
ALTER TABLE bundle_outcomes ADD INDEX idx_bundle_hash (bundle_hash) TYPE bloom_filter GRANULARITY 1;

-- Optimize for writes
SYSTEM STOP MERGES mev_sandwich_db.mev_sandwich;
ALTER TABLE mev_sandwich MODIFY SETTING merge_with_ttl_timeout = 86400;
SYSTEM START MERGES mev_sandwich_db.mev_sandwich;