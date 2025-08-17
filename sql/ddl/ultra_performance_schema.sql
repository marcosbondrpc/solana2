-- Ultra-Performance ClickHouse Schema for MEV Detection
-- Target: 500k+ rows/s ingestion, <100ms query latency
-- DEFENSIVE-ONLY: Pure monitoring and analysis

-- Drop existing tables
DROP TABLE IF EXISTS mev_detection.shred_stream_buffer;
DROP TABLE IF EXISTS mev_detection.shred_stream;
DROP TABLE IF EXISTS mev_detection.sandwich_detections_buffer;
DROP TABLE IF EXISTS mev_detection.sandwich_detections;
DROP TABLE IF EXISTS mev_detection.mev_opportunities_buffer;
DROP TABLE IF EXISTS mev_detection.mev_opportunities;
DROP TABLE IF EXISTS mev_detection.entity_behaviors_agg;
DROP TABLE IF EXISTS mev_detection.detection_metrics_5s;
DROP DATABASE IF EXISTS mev_detection;

-- Create optimized database
CREATE DATABASE IF NOT EXISTS mev_detection
ENGINE = Atomic
COMMENT 'Ultra-high performance MEV detection database';

USE mev_detection;

-- ============================================================================
-- ULTRA-OPTIMIZED SHRED STREAM TABLE
-- ============================================================================
CREATE TABLE shred_stream_buffer
(
    -- Primary fields with optimal types
    timestamp_ns    UInt64 CODEC(DoubleDelta, LZ4HC(9)),
    slot           UInt64 CODEC(DoubleDelta, LZ4HC(9)),
    shred_index    UInt32 CODEC(DoubleDelta),
    shred_type     Enum8('data' = 1, 'coding' = 2) CODEC(ZSTD(1)),
    
    -- Transaction data (compressed)
    signature      FixedString(64) CODEC(ZSTD(3)),
    instructions   Array(LowCardinality(String)) CODEC(ZSTD(3)),
    accounts       Array(FixedString(44)) CODEC(ZSTD(3)),
    
    -- MEV detection fields
    is_dex_interaction  Bool CODEC(ZSTD(1)),
    is_sandwich_candidate Bool CODEC(ZSTD(1)),
    detected_pattern   LowCardinality(String) CODEC(ZSTD(1)),
    
    -- Performance metrics
    ingestion_latency_us UInt32 CODEC(Gorilla, LZ4HC(9)),
    processing_latency_us UInt32 CODEC(Gorilla, LZ4HC(9)),
    
    -- Computed columns for fast filtering
    hour           DateTime MATERIALIZED toStartOfHour(fromUnixTimestamp64Nano(timestamp_ns)),
    minute         DateTime MATERIALIZED toStartOfMinute(fromUnixTimestamp64Nano(timestamp_ns))
)
ENGINE = Buffer(
    'mev_detection',           -- database
    'shred_stream',            -- target table
    16,                        -- num_layers
    10,                        -- min_time (seconds)
    100,                       -- max_time
    10000,                     -- min_rows
    1000000,                   -- max_rows
    10000000,                  -- min_bytes
    100000000                  -- max_bytes
)
SETTINGS 
    min_bytes_to_use_direct_io = 10485760;

-- Main shred stream table with projections
CREATE TABLE shred_stream
(
    timestamp_ns    UInt64 CODEC(DoubleDelta, LZ4HC(9)),
    slot           UInt64 CODEC(DoubleDelta, LZ4HC(9)),
    shred_index    UInt32 CODEC(DoubleDelta),
    shred_type     Enum8('data' = 1, 'coding' = 2) CODEC(ZSTD(1)),
    signature      FixedString(64) CODEC(ZSTD(3)),
    instructions   Array(LowCardinality(String)) CODEC(ZSTD(3)),
    accounts       Array(FixedString(44)) CODEC(ZSTD(3)),
    is_dex_interaction  Bool CODEC(ZSTD(1)),
    is_sandwich_candidate Bool CODEC(ZSTD(1)),
    detected_pattern   LowCardinality(String) CODEC(ZSTD(1)),
    ingestion_latency_us UInt32 CODEC(Gorilla, LZ4HC(9)),
    processing_latency_us UInt32 CODEC(Gorilla, LZ4HC(9)),
    hour           DateTime CODEC(DoubleDelta),
    minute         DateTime CODEC(DoubleDelta),
    
    -- Bloom filter index for signatures
    INDEX idx_signature signature TYPE bloom_filter(0.001) GRANULARITY 8192,
    -- Set index for accounts
    INDEX idx_accounts accounts TYPE set(100) GRANULARITY 4096,
    -- MinMax index for timestamps
    INDEX idx_timestamp timestamp_ns TYPE minmax GRANULARITY 8192
)
ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(fromUnixTimestamp64Nano(timestamp_ns))
ORDER BY (slot, shred_index, timestamp_ns)
PRIMARY KEY (slot, shred_index)
SAMPLE BY xxHash32(signature)
TTL toDateTime(timestamp_ns / 1000000000) + INTERVAL 7 DAY DELETE
SETTINGS 
    index_granularity = 8192,
    index_granularity_bytes = 10485760,
    min_bytes_for_wide_part = 10485760,
    min_rows_for_wide_part = 10000,
    merge_max_block_size = 8192,
    max_bytes_to_merge_at_max_space_in_pool = 10737418240,
    enable_mixed_granularity_parts = 1,
    use_minimalistic_part_header_in_zookeeper = 1,
    compress_marks = 1,
    compress_primary_key = 1,
    vertical_merge_algorithm_min_rows_to_activate = 100000,
    vertical_merge_algorithm_min_columns_to_activate = 10,
    allow_remote_fs_zero_copy_replication = 1;

-- Projection for DEX queries
ALTER TABLE shred_stream ADD PROJECTION projection_dex
(
    SELECT
        slot,
        timestamp_ns,
        signature,
        instructions,
        accounts,
        detected_pattern
    WHERE is_dex_interaction = 1
    ORDER BY timestamp_ns
);

-- Projection for sandwich detection
ALTER TABLE shred_stream ADD PROJECTION projection_sandwich
(
    SELECT
        slot,
        timestamp_ns,
        signature,
        accounts,
        detected_pattern,
        processing_latency_us
    WHERE is_sandwich_candidate = 1
    ORDER BY (slot, timestamp_ns)
);

-- ============================================================================
-- SANDWICH DETECTION TABLE WITH ADVANCED INDEXING
-- ============================================================================
CREATE TABLE sandwich_detections_buffer
(
    detection_id    UUID DEFAULT generateUUIDv4(),
    timestamp_ns    UInt64 CODEC(DoubleDelta, LZ4HC(9)),
    slot           UInt64 CODEC(DoubleDelta, LZ4HC(9)),
    
    -- Sandwich components
    frontrun_tx    FixedString(64) CODEC(ZSTD(3)),
    victim_tx      FixedString(64) CODEC(ZSTD(3)),
    backrun_tx     FixedString(64) CODEC(ZSTD(3)),
    
    -- Detection metadata
    confidence_score Float32 CODEC(Gorilla),
    pattern_type    LowCardinality(String) CODEC(ZSTD(1)),
    dex_program     LowCardinality(String) CODEC(ZSTD(1)),
    
    -- Economic impact
    victim_loss_lamports  UInt64 CODEC(Gorilla, LZ4HC(9)),
    attacker_profit_lamports UInt64 CODEC(Gorilla, LZ4HC(9)),
    
    -- Detection performance
    detection_latency_us UInt32 CODEC(Gorilla, LZ4HC(9)),
    
    -- Probabilistic data structures
    involved_accounts_hll AggregateFunction(uniqHLL12, String),
    instruction_patterns_bloom AggregateFunction(groupBitmap, UInt32)
)
ENGINE = Buffer(
    'mev_detection',
    'sandwich_detections',
    16, 5, 60, 1000, 100000, 1000000, 10000000
);

CREATE TABLE sandwich_detections
(
    detection_id    UUID,
    timestamp_ns    UInt64 CODEC(DoubleDelta, LZ4HC(9)),
    slot           UInt64 CODEC(DoubleDelta, LZ4HC(9)),
    frontrun_tx    FixedString(64) CODEC(ZSTD(3)),
    victim_tx      FixedString(64) CODEC(ZSTD(3)),
    backrun_tx     FixedString(64) CODEC(ZSTD(3)),
    confidence_score Float32 CODEC(Gorilla),
    pattern_type    LowCardinality(String) CODEC(ZSTD(1)),
    dex_program     LowCardinality(String) CODEC(ZSTD(1)),
    victim_loss_lamports  UInt64 CODEC(Gorilla, LZ4HC(9)),
    attacker_profit_lamports UInt64 CODEC(Gorilla, LZ4HC(9)),
    detection_latency_us UInt32 CODEC(Gorilla, LZ4HC(9)),
    involved_accounts_hll AggregateFunction(uniqHLL12, String),
    instruction_patterns_bloom AggregateFunction(groupBitmap, UInt32),
    
    -- Indexes
    INDEX idx_victim victim_tx TYPE bloom_filter(0.001) GRANULARITY 4096,
    INDEX idx_slot slot TYPE minmax GRANULARITY 1024,
    INDEX idx_confidence confidence_score TYPE minmax GRANULARITY 1024
)
ENGINE = ReplacingMergeTree(timestamp_ns)
PARTITION BY toYYYYMMDD(fromUnixTimestamp64Nano(timestamp_ns))
ORDER BY (slot, detection_id, timestamp_ns)
PRIMARY KEY (slot, detection_id)
TTL toDateTime(timestamp_ns / 1000000000) + INTERVAL 30 DAY DELETE
SETTINGS
    index_granularity = 4096,
    enable_mixed_granularity_parts = 1;

-- ============================================================================
-- MEV OPPORTUNITIES WITH AGGREGATING MERGE TREE
-- ============================================================================
CREATE TABLE mev_opportunities_buffer
(
    timestamp_ns    UInt64 CODEC(DoubleDelta, LZ4HC(9)),
    slot           UInt64 CODEC(DoubleDelta, LZ4HC(9)),
    opportunity_type LowCardinality(String) CODEC(ZSTD(1)),
    
    -- Opportunity details
    program_id     FixedString(44) CODEC(ZSTD(3)),
    involved_accounts Array(FixedString(44)) CODEC(ZSTD(3)),
    
    -- Economic metrics
    potential_profit_lamports UInt64 CODEC(Gorilla, LZ4HC(9)),
    required_capital_lamports UInt64 CODEC(Gorilla, LZ4HC(9)),
    success_probability Float32 CODEC(Gorilla),
    
    -- Risk metrics
    competition_level Enum8('low' = 1, 'medium' = 2, 'high' = 3) CODEC(ZSTD(1)),
    time_sensitivity_ms UInt32 CODEC(Gorilla),
    
    -- Detection metadata
    detection_method LowCardinality(String) CODEC(ZSTD(1)),
    model_version    LowCardinality(String) CODEC(ZSTD(1)),
    confidence_score Float32 CODEC(Gorilla)
)
ENGINE = Buffer(
    'mev_detection',
    'mev_opportunities',
    16, 5, 60, 1000, 100000, 1000000, 10000000
);

CREATE TABLE mev_opportunities
(
    timestamp_ns    UInt64 CODEC(DoubleDelta, LZ4HC(9)),
    slot           UInt64 CODEC(DoubleDelta, LZ4HC(9)),
    opportunity_type LowCardinality(String) CODEC(ZSTD(1)),
    program_id     FixedString(44) CODEC(ZSTD(3)),
    involved_accounts Array(FixedString(44)) CODEC(ZSTD(3)),
    potential_profit_lamports UInt64 CODEC(Gorilla, LZ4HC(9)),
    required_capital_lamports UInt64 CODEC(Gorilla, LZ4HC(9)),
    success_probability Float32 CODEC(Gorilla),
    competition_level Enum8('low' = 1, 'medium' = 2, 'high' = 3) CODEC(ZSTD(1)),
    time_sensitivity_ms UInt32 CODEC(Gorilla),
    detection_method LowCardinality(String) CODEC(ZSTD(1)),
    model_version    LowCardinality(String) CODEC(ZSTD(1)),
    confidence_score Float32 CODEC(Gorilla),
    
    -- Aggregation states
    total_opportunities SimpleAggregateFunction(sum, UInt64) DEFAULT 1,
    avg_profit SimpleAggregateFunction(avg, UInt64) DEFAULT potential_profit_lamports,
    max_profit SimpleAggregateFunction(max, UInt64) DEFAULT potential_profit_lamports,
    
    INDEX idx_type opportunity_type TYPE set(10) GRANULARITY 1024,
    INDEX idx_program program_id TYPE bloom_filter(0.001) GRANULARITY 4096
)
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMMDD(fromUnixTimestamp64Nano(timestamp_ns))
ORDER BY (opportunity_type, slot, timestamp_ns)
PRIMARY KEY (opportunity_type, slot)
TTL toDateTime(timestamp_ns / 1000000000) + INTERVAL 7 DAY
    GROUP BY opportunity_type, slot, toStartOfMinute(fromUnixTimestamp64Nano(timestamp_ns))
    SET
        total_opportunities = sum(total_opportunities),
        avg_profit = avg(avg_profit),
        max_profit = max(max_profit)
SETTINGS
    index_granularity = 8192;

-- ============================================================================
-- ENTITY BEHAVIOR AGGREGATION
-- ============================================================================
CREATE TABLE entity_behaviors_agg
(
    date           Date CODEC(DoubleDelta),
    entity_address FixedString(44) CODEC(ZSTD(3)),
    entity_type    LowCardinality(String) CODEC(ZSTD(1)),
    
    -- Behavioral metrics
    transaction_count SimpleAggregateFunction(sum, UInt64),
    unique_programs SimpleAggregateFunction(uniqExact, String),
    sandwich_attempts SimpleAggregateFunction(sum, UInt64),
    sandwich_successes SimpleAggregateFunction(sum, UInt64),
    
    -- Economic metrics
    total_volume_lamports SimpleAggregateFunction(sum, UInt64),
    total_profit_lamports SimpleAggregateFunction(sum, Int64),
    avg_gas_price SimpleAggregateFunction(avg, UInt64),
    
    -- Timing patterns
    avg_block_delay SimpleAggregateFunction(avg, Float32),
    transaction_timing_variance SimpleAggregateFunction(varPop, Float32),
    
    -- Risk scores
    risk_score SimpleAggregateFunction(max, Float32),
    reputation_score SimpleAggregateFunction(avg, Float32),
    
    -- HyperLogLog for unique interactions
    unique_counterparties AggregateFunction(uniqHLL12, String),
    unique_slots AggregateFunction(uniqHLL12, UInt64)
)
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (entity_type, entity_address, date)
PRIMARY KEY (entity_type, entity_address)
TTL date + INTERVAL 90 DAY
SETTINGS
    index_granularity = 4096;

-- ============================================================================
-- REAL-TIME METRICS (5-SECOND AGGREGATION)
-- ============================================================================
CREATE TABLE detection_metrics_5s
(
    timestamp      DateTime CODEC(DoubleDelta),
    
    -- Ingestion metrics
    rows_ingested  SimpleAggregateFunction(sum, UInt64),
    bytes_ingested SimpleAggregateFunction(sum, UInt64),
    
    -- Detection metrics
    sandwiches_detected SimpleAggregateFunction(sum, UInt64),
    arbitrage_detected SimpleAggregateFunction(sum, UInt64),
    liquidations_detected SimpleAggregateFunction(sum, UInt64),
    
    -- Latency percentiles
    p50_latency_us SimpleAggregateFunction(quantile(0.5), UInt32),
    p95_latency_us SimpleAggregateFunction(quantile(0.95), UInt32),
    p99_latency_us SimpleAggregateFunction(quantile(0.99), UInt32),
    
    -- Economic impact
    total_mev_extracted_lamports SimpleAggregateFunction(sum, UInt64),
    total_victim_loss_lamports SimpleAggregateFunction(sum, UInt64)
)
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY timestamp
PRIMARY KEY timestamp
TTL timestamp + INTERVAL 1 DAY
SETTINGS
    index_granularity = 720; -- 5 seconds * 720 = 1 hour

-- ============================================================================
-- MATERIALIZED VIEWS FOR REAL-TIME AGGREGATION
-- ============================================================================

-- Real-time sandwich detection aggregation
CREATE MATERIALIZED VIEW sandwich_detection_5s_mv
TO detection_metrics_5s
AS SELECT
    toStartOfFiveSeconds(fromUnixTimestamp64Nano(timestamp_ns)) AS timestamp,
    count() AS sandwiches_detected,
    sum(victim_loss_lamports) AS total_victim_loss_lamports,
    sum(attacker_profit_lamports) AS total_mev_extracted_lamports,
    quantile(0.5)(detection_latency_us) AS p50_latency_us,
    quantile(0.95)(detection_latency_us) AS p95_latency_us,
    quantile(0.99)(detection_latency_us) AS p99_latency_us
FROM sandwich_detections
GROUP BY timestamp;

-- Entity behavior aggregation
CREATE MATERIALIZED VIEW entity_behavior_daily_mv
TO entity_behaviors_agg
AS SELECT
    toDate(fromUnixTimestamp64Nano(timestamp_ns)) AS date,
    accounts[1] AS entity_address,
    'unknown' AS entity_type,
    count() AS transaction_count,
    uniqExact(arrayJoin(instructions)) AS unique_programs,
    countIf(is_sandwich_candidate) AS sandwich_attempts,
    0 AS sandwich_successes,
    0 AS total_volume_lamports,
    0 AS total_profit_lamports,
    0 AS avg_gas_price,
    0 AS avg_block_delay,
    0 AS transaction_timing_variance,
    0 AS risk_score,
    0 AS reputation_score,
    uniqHLL12State(arrayJoin(accounts)) AS unique_counterparties,
    uniqHLL12State(slot) AS unique_slots
FROM shred_stream
WHERE length(accounts) > 0
GROUP BY date, entity_address, entity_type;

-- ============================================================================
-- DISTRIBUTED TABLES FOR HORIZONTAL SCALING
-- ============================================================================

-- Create distributed tables if cluster is configured
-- CREATE TABLE shred_stream_distributed AS shred_stream
-- ENGINE = Distributed('mev_cluster', 'mev_detection', 'shred_stream', xxHash32(signature));

-- CREATE TABLE sandwich_detections_distributed AS sandwich_detections
-- ENGINE = Distributed('mev_cluster', 'mev_detection', 'sandwich_detections', xxHash32(detection_id));

-- ============================================================================
-- PERFORMANCE OPTIMIZATIONS
-- ============================================================================

-- Optimize tables for production
OPTIMIZE TABLE shred_stream FINAL;
OPTIMIZE TABLE sandwich_detections FINAL;
OPTIMIZE TABLE mev_opportunities FINAL;
OPTIMIZE TABLE entity_behaviors_agg FINAL;

-- Create dictionary for fast program lookups
CREATE DICTIONARY program_names
(
    program_id String,
    program_name String,
    program_type String
)
PRIMARY KEY program_id
SOURCE(FILE(path '/opt/clickhouse/dictionaries/programs.csv' format 'CSV'))
LIFETIME(MIN 3600 MAX 7200)
LAYOUT(HASHED());

-- ============================================================================
-- MONITORING QUERIES
-- ============================================================================

-- Check ingestion rate
-- SELECT
--     toStartOfMinute(now()) AS minute,
--     count() / 60 AS rows_per_second,
--     sum(length(signature)) / 60 / 1024 / 1024 AS mb_per_second
-- FROM shred_stream
-- WHERE timestamp_ns >= toUnixTimestamp64Nano(now() - INTERVAL 1 MINUTE);

-- Check detection latency
-- SELECT
--     quantile(0.5)(detection_latency_us) / 1000 AS p50_ms,
--     quantile(0.95)(detection_latency_us) / 1000 AS p95_ms,
--     quantile(0.99)(detection_latency_us) / 1000 AS p99_ms
-- FROM sandwich_detections
-- WHERE timestamp_ns >= toUnixTimestamp64Nano(now() - INTERVAL 5 MINUTE);

-- ============================================================================
-- GRANTS FOR DEFENSIVE ACCESS ONLY
-- ============================================================================
CREATE USER IF NOT EXISTS 'mev_detector' IDENTIFIED BY 'secure_password';
GRANT SELECT ON mev_detection.* TO 'mev_detector';
GRANT INSERT ON mev_detection.*_buffer TO 'mev_detector';
REVOKE ALL ON CLUSTER mev_cluster FROM 'mev_detector'; -- No cluster admin

-- Performance settings for session
-- SET max_threads = 16;
-- SET max_memory_usage = 10000000000; -- 10GB
-- SET max_bytes_before_external_group_by = 5000000000; -- 5GB
-- SET distributed_product_mode = 'global';
-- SET optimize_skip_unused_shards = 1;
-- SET force_index_by_date = 1;
-- SET optimize_use_projections = 1;