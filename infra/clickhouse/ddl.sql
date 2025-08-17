-- ClickHouse DDL for MEV Infrastructure
-- Defensive monitoring and analytics only

-- Create database
CREATE DATABASE IF NOT EXISTS mev;
USE mev;

-- Blocks table
CREATE TABLE IF NOT EXISTS blocks
(
    slot UInt64,
    blockhash String,
    parent_slot UInt64,
    block_time DateTime64(3),
    block_height UInt64,
    leader String,
    transaction_count UInt32,
    compute_units_consumed UInt64,
    rewards Decimal64(4),
    INDEX idx_slot slot TYPE minmax GRANULARITY 1,
    INDEX idx_time block_time TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(block_time)
ORDER BY (slot, block_time)
TTL block_time + INTERVAL 90 DAY TO VOLUME 'cold'
SETTINGS index_granularity = 8192;

-- Transactions table
CREATE TABLE IF NOT EXISTS transactions
(
    signature String,
    slot UInt64,
    block_time DateTime64(3),
    success Bool,
    fee UInt64,
    compute_units_consumed UInt64,
    log_messages Array(String),
    account_keys Array(String),
    recent_blockhash String,
    instruction_count UInt8,
    INDEX idx_signature signature TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_slot slot TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(block_time)
ORDER BY (slot, signature)
TTL block_time + INTERVAL 60 DAY TO VOLUME 'cold'
SETTINGS index_granularity = 8192;

-- Instructions table
CREATE TABLE IF NOT EXISTS instructions
(
    tx_signature String,
    instruction_index UInt8,
    program_id String,
    accounts Array(String),
    data String,
    slot UInt64,
    block_time DateTime64(3),
    INDEX idx_program program_id TYPE bloom_filter(0.01) GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(block_time)
ORDER BY (slot, tx_signature, instruction_index)
TTL block_time + INTERVAL 30 DAY TO VOLUME 'cold'
SETTINGS index_granularity = 8192;

-- DEX fills table
CREATE TABLE IF NOT EXISTS dex_fills
(
    tx_signature String,
    slot UInt64,
    block_time DateTime64(3),
    dex String,
    market String,
    side Enum('buy' = 1, 'sell' = 2),
    price Decimal64(8),
    size Decimal64(8),
    maker String,
    taker String,
    fee Decimal64(8),
    INDEX idx_market market TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_time block_time TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(block_time)
ORDER BY (block_time, dex, market)
TTL block_time + INTERVAL 30 DAY TO VOLUME 'cold'
SETTINGS index_granularity = 8192;

-- Arbitrage events table (detection only)
CREATE TABLE IF NOT EXISTS arbitrage_events
(
    event_id String,
    slot UInt64,
    block_time DateTime64(3),
    tx_signature String,
    legs UInt8,
    roi_pct Float64,
    est_profit Float64,
    tokens Array(String),
    dex_route Array(String),
    confidence Float32,
    detected_at DateTime64(3),
    summary String,
    INDEX idx_slot slot TYPE minmax GRANULARITY 1,
    INDEX idx_profit est_profit TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(block_time)
ORDER BY (slot, tx_signature)
TTL block_time + INTERVAL 180 DAY
SETTINGS index_granularity = 8192;

-- Sandwich events table (detection only)
CREATE TABLE IF NOT EXISTS sandwich_events
(
    event_id String,
    slot UInt64,
    block_time DateTime64(3),
    victim_tx String,
    front_tx String,
    back_tx String,
    victim_loss Float64,
    attacker_profit Float64,
    token_pair String,
    dex String,
    detected_at DateTime64(3),
    INDEX idx_slot slot TYPE minmax GRANULARITY 1,
    INDEX idx_victim victim_tx TYPE bloom_filter(0.01) GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(block_time)
ORDER BY (slot, victim_tx)
TTL block_time + INTERVAL 180 DAY
SETTINGS index_granularity = 8192;

-- Audit events table
CREATE TABLE IF NOT EXISTS audit_events
(
    event_id String DEFAULT generateUUIDv4(),
    timestamp DateTime64(3) DEFAULT now64(3),
    user_id String,
    user_email String,
    action Enum('read' = 1, 'write' = 2, 'delete' = 3, 'export' = 4, 'train' = 5, 'deploy' = 6, 'killswitch' = 7),
    resource String,
    result Enum('success' = 1, 'failure' = 2, 'partial' = 3),
    ip_address String,
    user_agent String,
    request_id String,
    parameters String, -- JSON
    metadata String,   -- JSON
    hash_chain String, -- Previous event hash for integrity
    INDEX idx_user user_id TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_time timestamp TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, user_id)
SETTINGS index_granularity = 8192;

-- System metrics table
CREATE TABLE IF NOT EXISTS system_metrics
(
    timestamp DateTime64(3),
    node_id String,
    metric_name String,
    metric_value Float64,
    labels String, -- JSON
    INDEX idx_time timestamp TYPE minmax GRANULARITY 1,
    INDEX idx_metric metric_name TYPE bloom_filter(0.01) GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (timestamp, node_id, metric_name)
TTL timestamp + INTERVAL 7 DAY
SETTINGS index_granularity = 8192;

-- ML model metadata
CREATE TABLE IF NOT EXISTS ml_models
(
    model_id String,
    model_type Enum('arbitrage_detector' = 1, 'sandwich_detector' = 2, 'liquidation_predictor' = 3, 'confidence_scorer' = 4),
    version String,
    created_at DateTime64(3),
    created_by String,
    accuracy Float32,
    precision Float32,
    recall Float32,
    f1_score Float32,
    parameters String, -- JSON
    dataset_id String,
    deployment_mode Enum('shadow' = 1, 'canary' = 2, 'production' = 3, 'rollback' = 4),
    is_active Bool,
    INDEX idx_model model_id TYPE bloom_filter(0.01) GRANULARITY 1
)
ENGINE = MergeTree
ORDER BY (model_id, version)
SETTINGS index_granularity = 8192;

-- Export jobs table
CREATE TABLE IF NOT EXISTS export_jobs
(
    job_id String DEFAULT generateUUIDv4(),
    created_at DateTime64(3) DEFAULT now64(3),
    created_by String,
    dataset String,
    format Enum('parquet' = 1, 'arrow' = 2, 'csv' = 3, 'json' = 4),
    time_range_start DateTime,
    time_range_end DateTime,
    filters String, -- JSON
    columns Array(String),
    status Enum('pending' = 1, 'running' = 2, 'completed' = 3, 'failed' = 4, 'cancelled' = 5),
    progress_pct Float32,
    output_path String,
    error_message String,
    completed_at DateTime64(3),
    row_count UInt64,
    file_size_bytes UInt64,
    INDEX idx_job job_id TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_user created_by TYPE bloom_filter(0.01) GRANULARITY 1
)
ENGINE = MergeTree
ORDER BY (created_at, job_id)
TTL created_at + INTERVAL 30 DAY
SETTINGS index_granularity = 8192;

-- Materialized views for real-time aggregations

-- TPS by minute
CREATE MATERIALIZED VIEW IF NOT EXISTS tps_by_minute
ENGINE = AggregatingMergeTree
PARTITION BY toYYYYMM(minute)
ORDER BY minute
AS SELECT
    toStartOfMinute(block_time) AS minute,
    count() / 60 AS tps,
    avg(compute_units_consumed) AS avg_compute_units
FROM transactions
GROUP BY minute;

-- Arbitrage statistics by hour
CREATE MATERIALIZED VIEW IF NOT EXISTS arbitrage_stats_hourly
ENGINE = AggregatingMergeTree
PARTITION BY toYYYYMM(hour)
ORDER BY hour
AS SELECT
    toStartOfHour(block_time) AS hour,
    count() AS count,
    avg(roi_pct) AS avg_roi,
    sum(est_profit) AS total_profit,
    avg(legs) AS avg_legs,
    avg(confidence) AS avg_confidence
FROM arbitrage_events
GROUP BY hour;

-- Create storage policies for hot/cold tiering
-- This requires configuration in config.xml
-- Example:
-- <storage_configuration>
--   <disks>
--     <hot><path>/var/lib/clickhouse/hot/</path></hot>
--     <cold><path>/mnt/s3/cold/</path></cold>
--   </disks>
--   <policies>
--     <tiered>
--       <volumes>
--         <hot><disk>hot</disk></hot>
--         <cold><disk>cold</disk></cold>
--       </volumes>
--     </tiered>
--   </policies>
-- </storage_configuration>