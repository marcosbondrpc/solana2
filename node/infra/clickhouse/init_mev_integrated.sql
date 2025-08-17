-- MEV Integrated ClickHouse Schema
-- Combines arbitrage detection, MEV counterfactuals, and decision lineage

CREATE DATABASE IF NOT EXISTS mev_data;
USE mev_data;

-- Core MEV Counterfactuals Table
CREATE TABLE IF NOT EXISTS mev_counterfactuals
(
    ts                    DateTime64(6, 'UTC'),
    request_id            UUID,
    slot                  UInt64,
    leader                FixedString(44),
    chosen_route          LowCardinality(String),
    shadow_route          LowCardinality(String),
    predicted_land_prob   Float64,
    predicted_ev          Float64,
    realized_land         UInt8,
    realized_ev           Float64
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(ts)
ORDER BY (ts, request_id, shadow_route)
TTL ts + INTERVAL 30 DAY;

-- MEV Decision Lineage Table
CREATE TABLE IF NOT EXISTS mev_decision_lineage
(
    ts                    DateTime64(6, 'UTC'),
    request_id            UUID,
    slot                  UInt64,
    leader                FixedString(44),
    route                 LowCardinality(String),
    predicted_land_prob   Float64,
    predicted_ev          Float64,
    realized_land         UInt8,
    realized_ev           Float64,
    decision_factors      String,
    metadata              String
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(ts)
ORDER BY (ts, request_id, route)
TTL ts + INTERVAL 30 DAY;

-- Arbitrage Opportunities Table
CREATE TABLE IF NOT EXISTS arbitrage_opportunities
(
    timestamp             DateTime64(9, 'UTC'),
    opportunity_id        UUID,
    block_height          UInt64,
    slot                  UInt64,
    source_dex           LowCardinality(String),
    target_dex           LowCardinality(String),
    token_a              FixedString(44),
    token_b              FixedString(44),
    source_price         Decimal(38, 18),
    target_price         Decimal(38, 18),
    price_diff_percent   Float64,
    estimated_profit_sol Decimal(18, 9),
    estimated_gas_cost   Decimal(18, 9),
    net_profit_sol       Decimal(18, 9),
    confidence_score     Float32,
    execution_status     Enum8('pending' = 0, 'submitted' = 1, 'landed' = 2, 'failed' = 3, 'expired' = 4),
    tx_signature         Nullable(String),
    actual_profit        Nullable(Decimal(18, 9)),
    execution_time_ms    Nullable(UInt32),
    failure_reason       Nullable(String)
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (timestamp, opportunity_id)
TTL timestamp + INTERVAL 90 DAY
SETTINGS index_granularity = 8192;

-- DEX Pool States Table
CREATE TABLE IF NOT EXISTS dex_pool_states
(
    timestamp            DateTime64(9, 'UTC'),
    pool_address        FixedString(44),
    dex_name            LowCardinality(String),
    token_a             FixedString(44),
    token_b             FixedString(44),
    reserve_a           Decimal(38, 18),
    reserve_b           Decimal(38, 18),
    price_a_to_b        Decimal(38, 18),
    liquidity_usd       Decimal(18, 2),
    volume_24h_usd      Decimal(18, 2),
    fee_tier            UInt16,
    slot                UInt64
)
ENGINE = ReplacingMergeTree
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (pool_address, timestamp)
TTL timestamp + INTERVAL 30 DAY;

-- MEV Bundle Submissions Table
CREATE TABLE IF NOT EXISTS mev_bundle_submissions
(
    timestamp           DateTime64(9, 'UTC'),
    bundle_id          UUID,
    slot               UInt64,
    builder_pubkey     FixedString(44),
    num_transactions   UInt16,
    bundle_type        LowCardinality(String),  -- 'arbitrage', 'sandwich', 'liquidation', 'mixed'
    estimated_profit   Decimal(18, 9),
    tip_amount        Decimal(18, 9),
    gas_used          UInt64,
    submission_method  LowCardinality(String),  -- 'jito', 'flashbots', 'direct'
    landing_status     Enum8('pending' = 0, 'landed' = 1, 'rejected' = 2, 'expired' = 3),
    block_hash        Nullable(String),
    position_in_block  Nullable(UInt16),
    actual_profit     Nullable(Decimal(18, 9)),
    competitor_bundles Array(UUID),
    metadata          String
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (timestamp, bundle_id)
TTL timestamp + INTERVAL 60 DAY;

-- Transaction Mempool Events Table
CREATE TABLE IF NOT EXISTS mempool_events
(
    timestamp          DateTime64(9, 'UTC'),
    tx_signature      String,
    event_type        Enum8('seen' = 0, 'included' = 1, 'dropped' = 2, 'replaced' = 3),
    slot              UInt64,
    priority_fee      UInt64,
    compute_units     UInt32,
    accounts          Array(String),
    program_ids       Array(String),
    is_mev_related    UInt8,
    mev_type         Nullable(String),
    estimated_value   Nullable(Decimal(18, 9))
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (timestamp, tx_signature)
TTL timestamp + INTERVAL 7 DAY;

-- Performance Metrics Table
CREATE TABLE IF NOT EXISTS performance_metrics
(
    timestamp         DateTime64(3, 'UTC'),
    metric_name      LowCardinality(String),
    service_name     LowCardinality(String),
    value            Float64,
    tags             Map(String, String)
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (service_name, metric_name, timestamp)
TTL timestamp + INTERVAL 14 DAY;

-- Kafka Engine Tables for Real-time Ingestion
CREATE TABLE IF NOT EXISTS kafka_arbitrage_events
(
    data String
)
ENGINE = Kafka
SETTINGS kafka_broker_list = 'kafka:9092',
         kafka_topic_list = 'arbitrage-events',
         kafka_group_name = 'clickhouse-arbitrage',
         kafka_format = 'JSONAsString',
         kafka_num_consumers = 4;

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_arbitrage_events TO arbitrage_opportunities AS
SELECT
    toDateTime64(JSONExtractUInt(data, 'timestamp') / 1000000000, 9, 'UTC') AS timestamp,
    JSONExtractString(data, 'opportunity_id')::UUID AS opportunity_id,
    JSONExtractUInt(data, 'block_height') AS block_height,
    JSONExtractUInt(data, 'slot') AS slot,
    JSONExtractString(data, 'source_dex') AS source_dex,
    JSONExtractString(data, 'target_dex') AS target_dex,
    JSONExtractString(data, 'token_a')::FixedString(44) AS token_a,
    JSONExtractString(data, 'token_b')::FixedString(44) AS token_b,
    toDecimal128(JSONExtractString(data, 'source_price'), 18) AS source_price,
    toDecimal128(JSONExtractString(data, 'target_price'), 18) AS target_price,
    JSONExtractFloat(data, 'price_diff_percent') AS price_diff_percent,
    toDecimal64(JSONExtractString(data, 'estimated_profit_sol'), 9) AS estimated_profit_sol,
    toDecimal64(JSONExtractString(data, 'estimated_gas_cost'), 9) AS estimated_gas_cost,
    toDecimal64(JSONExtractString(data, 'net_profit_sol'), 9) AS net_profit_sol,
    toFloat32(JSONExtractFloat(data, 'confidence_score')) AS confidence_score,
    0 AS execution_status,
    NULL AS tx_signature,
    NULL AS actual_profit,
    NULL AS execution_time_ms,
    NULL AS failure_reason
FROM kafka_arbitrage_events;

-- Indexes for optimal query performance
ALTER TABLE arbitrage_opportunities ADD INDEX idx_profit (net_profit_sol) TYPE minmax GRANULARITY 4;
ALTER TABLE arbitrage_opportunities ADD INDEX idx_tokens (token_a, token_b) TYPE bloom_filter GRANULARITY 1;
ALTER TABLE dex_pool_states ADD INDEX idx_pool (pool_address) TYPE bloom_filter GRANULARITY 1;
ALTER TABLE mev_bundle_submissions ADD INDEX idx_slot (slot) TYPE minmax GRANULARITY 4;

-- Create aggregated views for dashboards
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_arbitrage_stats
ENGINE = SummingMergeTree
PARTITION BY toYYYYMM(hour)
ORDER BY (hour, source_dex, target_dex)
AS SELECT
    toStartOfHour(timestamp) AS hour,
    source_dex,
    target_dex,
    count() AS opportunity_count,
    sum(net_profit_sol) AS total_profit,
    avg(net_profit_sol) AS avg_profit,
    max(net_profit_sol) AS max_profit,
    countIf(execution_status = 2) AS landed_count,
    sumIf(actual_profit, execution_status = 2) AS realized_profit
FROM arbitrage_opportunities
GROUP BY hour, source_dex, target_dex;

CREATE MATERIALIZED VIEW IF NOT EXISTS daily_mev_stats
ENGINE = SummingMergeTree
PARTITION BY toYYYYMM(day)
ORDER BY (day, bundle_type)
AS SELECT
    toDate(timestamp) AS day,
    bundle_type,
    count() AS bundle_count,
    sum(estimated_profit) AS total_estimated_profit,
    sumIf(actual_profit, landing_status = 1) AS total_realized_profit,
    countIf(landing_status = 1) AS landed_bundles,
    avg(tip_amount) AS avg_tip,
    avg(gas_used) AS avg_gas
FROM mev_bundle_submissions
GROUP BY day, bundle_type;

-- Grants for service accounts
GRANT SELECT, INSERT ON mev_data.* TO default;