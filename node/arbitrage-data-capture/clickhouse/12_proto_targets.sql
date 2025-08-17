-- Typed Target Tables with Optimized Storage
-- MergeTree tables for persistent storage with proper indexing

USE legendary_mev;

-- MEV Opportunities typed table
DROP TABLE IF EXISTS mev_opportunities;
CREATE TABLE mev_opportunities
(
    opportunity_id String,
    block_number UInt64,
    transaction_hash String,
    victim_tx String,
    opportunity_type Enum8('sandwich' = 1, 'frontrun' = 2, 'backrun' = 3, 'liquidation' = 4),
    
    -- Token details
    token_in String,
    token_out String,
    amount_in Decimal128(18),
    expected_profit Decimal128(18),
    
    -- Gas optimization
    gas_estimate UInt64,
    priority_fee Decimal64(8),
    base_fee Decimal64(8),
    
    -- ML features
    confidence_score Float32,
    features Array(Float32),
    model_version String,
    
    -- Decision lineage
    decision_dna_hash String,
    parent_decision String,
    
    -- Execution details
    executed Boolean DEFAULT false,
    execution_tx String,
    actual_profit Decimal128(18),
    execution_time_ms UInt32,
    
    -- Timestamps
    created_at DateTime64(9),
    executed_at DateTime64(9),
    
    -- Indexing date for partitioning
    date Date DEFAULT toDate(created_at)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (block_number, opportunity_id, created_at)
PRIMARY KEY (block_number, opportunity_id)
TTL date + INTERVAL 14 DAY
SETTINGS 
    index_granularity = 8192,
    merge_with_ttl_timeout = 86400,
    ttl_only_drop_parts = 1;

-- Create indices for fast queries
ALTER TABLE mev_opportunities ADD INDEX idx_token_in (token_in) TYPE bloom_filter(0.01) GRANULARITY 4;
ALTER TABLE mev_opportunities ADD INDEX idx_token_out (token_out) TYPE bloom_filter(0.01) GRANULARITY 4;
ALTER TABLE mev_opportunities ADD INDEX idx_confidence (confidence_score) TYPE minmax GRANULARITY 4;
ALTER TABLE mev_opportunities ADD INDEX idx_profit (expected_profit) TYPE minmax GRANULARITY 4;

-- Arbitrage Opportunities typed table
DROP TABLE IF EXISTS arb_opportunities;
CREATE TABLE arb_opportunities
(
    opportunity_id String,
    
    -- Path details
    source_pool String,
    target_pool String,
    token_path Array(String),
    dex_path Array(String),
    hop_count UInt8,
    
    -- Financial metrics
    input_amount Decimal128(18),
    output_amount Decimal128(18),
    profit_amount Decimal128(18),
    profit_usd Decimal64(8),
    gas_cost Decimal64(8),
    net_profit Decimal64(8),
    
    -- Risk metrics
    price_impact Float32,
    slippage_tolerance Float32,
    confidence_score Float32,
    volatility_score Float32,
    
    -- ML features
    features Array(Float32),
    model_version String,
    
    -- Decision lineage
    decision_dna_hash String,
    parent_decision String,
    
    -- Execution tracking
    executed Boolean DEFAULT false,
    execution_tx String,
    actual_profit Decimal128(18),
    execution_latency_ms UInt32,
    
    -- Route selection
    route_selector String, -- 'bandit', 'static', 'ml'
    bandit_arm_id String,
    
    -- Timestamps
    created_at DateTime64(9),
    executed_at DateTime64(9),
    
    -- Partitioning
    date Date DEFAULT toDate(created_at)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, net_profit, opportunity_id)
PRIMARY KEY (date, opportunity_id)
TTL date + INTERVAL 14 DAY
SETTINGS 
    index_granularity = 8192,
    merge_with_ttl_timeout = 86400;

-- Indices for arbitrage table
ALTER TABLE arb_opportunities ADD INDEX idx_source_pool (source_pool) TYPE bloom_filter(0.01) GRANULARITY 4;
ALTER TABLE arb_opportunities ADD INDEX idx_target_pool (target_pool) TYPE bloom_filter(0.01) GRANULARITY 4;
ALTER TABLE arb_opportunities ADD INDEX idx_net_profit (net_profit) TYPE minmax GRANULARITY 4;
ALTER TABLE arb_opportunities ADD INDEX idx_confidence (confidence_score) TYPE minmax GRANULARITY 4;

-- Bandit Events Table for Thompson Sampling
DROP TABLE IF EXISTS bandit_events;
CREATE TABLE bandit_events
(
    event_id String,
    arm_id String,
    decision_id String,
    
    -- Bandit metrics
    reward Float64,
    cost Float64,
    net_reward Float64 MATERIALIZED reward - cost,
    confidence Float64,
    exploration_bonus Float64,
    
    -- Budget tracking
    total_budget Float64,
    remaining_budget Float64,
    budget_utilization Float64 MATERIALIZED (total_budget - remaining_budget) / total_budget,
    
    -- Context for decision
    context_features Map(String, Float32),
    metadata Map(String, String),
    
    -- Performance tracking
    selection_reason Enum8('exploit' = 1, 'explore' = 2, 'forced' = 3),
    cumulative_reward Float64,
    cumulative_selections UInt64,
    
    -- Timestamps
    timestamp DateTime64(9),
    date Date DEFAULT toDate(timestamp)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (arm_id, timestamp)
PRIMARY KEY (arm_id, date)
TTL date + INTERVAL 30 DAY;

-- Control Commands Audit Table
DROP TABLE IF EXISTS control_commands;
CREATE TABLE control_commands
(
    command_id String,
    command_type String,
    payload String,
    
    -- Cryptographic verification
    signature String,
    signer_pubkey String,
    signature_valid Boolean,
    nonce UInt64,
    
    -- Execution tracking
    status Enum8('pending' = 1, 'executing' = 2, 'completed' = 3, 'failed' = 4),
    executor String,
    execution_time_ms UInt32,
    error_message String,
    
    -- ACK chain
    ack_hash String,
    previous_ack_hash String,
    
    -- Timestamps
    created_at DateTime64(9),
    executed_at DateTime64(9),
    date Date DEFAULT toDate(created_at)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, command_id)
PRIMARY KEY command_id
TTL date + INTERVAL 90 DAY; -- Longer retention for audit

-- Decision DNA Lineage Table
DROP TABLE IF EXISTS decision_lineage;
CREATE TABLE decision_lineage
(
    decision_id String,
    decision_hash String,
    parent_hash String,
    
    -- Decision details
    decision_type Enum8('mev' = 1, 'arb' = 2, 'route' = 3, 'fee' = 4),
    model_version String,
    features_hash String,
    
    -- Decision metrics
    confidence Float32,
    expected_value Decimal64(8),
    actual_value Nullable(Decimal64(8)),
    
    -- Merkle anchoring
    merkle_root String,
    merkle_proof Array(String),
    anchor_tx String, -- Daily on-chain anchor
    
    -- Metadata
    metadata Map(String, String),
    
    -- Timestamps
    created_at DateTime64(9),
    anchored_at Nullable(DateTime64(9)),
    date Date DEFAULT toDate(created_at)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, decision_id)
PRIMARY KEY decision_id
TTL date + INTERVAL 365 DAY; -- 1 year retention for lineage

-- Performance Metrics Table (high-frequency)
DROP TABLE IF EXISTS performance_metrics;
CREATE TABLE performance_metrics
(
    metric_name String,
    metric_value Float64,
    
    -- Categorization
    category Enum8('latency' = 1, 'throughput' = 2, 'profit' = 3, 'error' = 4, 'resource' = 5),
    component String,
    
    -- Labels for filtering
    labels Map(String, String),
    
    -- Aggregation helpers
    p50 Float64,
    p95 Float64,
    p99 Float64,
    
    -- Timestamp with microsecond precision
    timestamp DateTime64(6),
    date Date DEFAULT toDate(timestamp)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(date)
ORDER BY (metric_name, timestamp)
PRIMARY KEY (metric_name, date)
TTL date + INTERVAL 7 DAY
SETTINGS
    index_granularity = 32768; -- Higher granularity for metrics

-- Materialized Views for automatic data flow from Kafka to target tables

-- MEV Opportunities MV
CREATE MATERIALIZED VIEW mev_opportunities_mv TO mev_opportunities AS
SELECT 
    opportunity_id,
    block_number,
    transaction_hash,
    victim_tx,
    opportunity_type,
    token_in,
    token_out,
    amount_in,
    expected_profit,
    gas_estimate,
    priority_fee,
    0 as base_fee,
    confidence_score,
    features,
    'v1.0.0' as model_version,
    decision_dna_hash,
    '' as parent_decision,
    false as executed,
    '' as execution_tx,
    0 as actual_profit,
    0 as execution_time_ms,
    created_at,
    toDateTime64('1970-01-01 00:00:00', 9) as executed_at
FROM kafka_mev_opportunities_proto;

-- Arbitrage Opportunities MV
CREATE MATERIALIZED VIEW arb_opportunities_mv TO arb_opportunities AS
SELECT 
    opportunity_id,
    source_pool,
    target_pool,
    token_path,
    dex_path,
    length(dex_path) as hop_count,
    input_amount,
    output_amount,
    profit_amount,
    profit_usd,
    gas_cost,
    net_profit,
    price_impact,
    slippage_tolerance,
    confidence_score,
    0.0 as volatility_score,
    features,
    'v1.0.0' as model_version,
    decision_dna_hash,
    '' as parent_decision,
    false as executed,
    '' as execution_tx,
    0 as actual_profit,
    0 as execution_latency_ms,
    'bandit' as route_selector,
    '' as bandit_arm_id,
    created_at,
    toDateTime64('1970-01-01 00:00:00', 9) as executed_at
FROM kafka_arb_opportunities_proto;

-- Create aggregation views for monitoring

CREATE OR REPLACE VIEW hourly_mev_stats AS
SELECT 
    toStartOfHour(created_at) as hour,
    opportunity_type,
    count() as opportunities,
    sum(expected_profit) as total_expected_profit,
    avg(confidence_score) as avg_confidence,
    max(expected_profit) as max_opportunity,
    quantile(0.5)(expected_profit) as median_profit,
    quantile(0.95)(gas_estimate) as p95_gas
FROM mev_opportunities
WHERE date >= today() - 7
GROUP BY hour, opportunity_type;

CREATE OR REPLACE VIEW hourly_arb_stats AS
SELECT 
    toStartOfHour(created_at) as hour,
    count() as opportunities,
    sum(net_profit) as total_net_profit,
    avg(confidence_score) as avg_confidence,
    avg(hop_count) as avg_hops,
    max(net_profit) as max_opportunity,
    quantile(0.5)(net_profit) as median_profit,
    quantile(0.95)(execution_latency_ms) as p95_latency
FROM arb_opportunities
WHERE date >= today() - 7
GROUP BY hour;