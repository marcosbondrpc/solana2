-- LEGENDARY MEV Infrastructure Performance Tables
-- Ultra-optimized for institutional-scale monitoring

-- System metrics time series
CREATE TABLE IF NOT EXISTS system_metrics (
    timestamp DateTime64(3) DEFAULT now64(3),
    hostname String,
    cpu_usage_percent Float32,
    memory_used_mb UInt32,
    memory_total_mb UInt32,
    network_rx_bytes_sec UInt64,
    network_tx_bytes_sec UInt64,
    disk_read_bytes_sec UInt64,
    disk_write_bytes_sec UInt64,
    open_file_descriptors UInt32,
    threads_count UInt32,
    context_switches_sec UInt32,
    
    -- MEV specific metrics
    active_bundles UInt32,
    pending_transactions UInt32,
    mempool_size UInt32,
    
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 3600
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (hostname, timestamp)
TTL toDateTime(timestamp) + INTERVAL 30 DAY
SETTINGS index_granularity = 8192;

-- MEV bundle tracking
CREATE TABLE IF NOT EXISTS mev_bundles (
    bundle_id UUID DEFAULT generateUUIDv4(),
    created_at DateTime64(6) DEFAULT now64(6),
    submitted_at DateTime64(6),
    landed_at Nullable(DateTime64(6)),
    
    -- Bundle details
    bundle_hash FixedString(32),
    block_number UInt64,
    slot UInt64,
    proposer String,
    builder String,
    
    -- Transactions
    tx_count UInt16,
    tx_hashes Array(String),
    
    -- Economics
    total_gas_used UInt64,
    base_fee_gwei Float64,
    priority_fee_gwei Float64,
    tip_sol Float64,
    gross_profit_sol Float64,
    net_profit_sol Float64,
    
    -- Performance
    build_time_ms UInt32,
    simulation_time_ms UInt32,
    submission_latency_ms UInt32,
    propagation_time_ms UInt32,
    
    -- Status
    status Enum8('building' = 1, 'simulating' = 2, 'submitted' = 3, 
                 'pending' = 4, 'landed' = 5, 'failed' = 6, 'reverted' = 7),
    failure_reason Nullable(String),
    
    INDEX idx_bundle_hash bundle_hash TYPE bloom_filter GRANULARITY 1,
    INDEX idx_slot slot TYPE minmax GRANULARITY 1,
    INDEX idx_status status TYPE set(0) GRANULARITY 1
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(created_at)
ORDER BY (created_at, bundle_id)
TTL created_at + INTERVAL 90 DAY;

-- Arbitrage execution tracking
CREATE TABLE IF NOT EXISTS arbitrage_executions (
    execution_id UUID DEFAULT generateUUIDv4(),
    detected_at DateTime64(6) DEFAULT now64(6),
    executed_at Nullable(DateTime64(6)),
    
    -- Path details
    token_a String,
    token_b String,
    path_json String,  -- JSON array of DEX hops
    path_length UInt8,
    
    -- Profitability
    expected_profit_usd Float64,
    actual_profit_usd Nullable(Float64),
    gas_estimate_sol Float64,
    gas_used_sol Nullable(Float64),
    slippage_percent Float32,
    
    -- Execution details
    input_amount Float64,
    output_amount Nullable(Float64),
    price_impact Float32,
    
    -- Performance metrics
    detection_latency_ms UInt32,
    simulation_time_ms UInt32,
    execution_time_ms Nullable(UInt32),
    total_latency_ms Nullable(UInt32),
    
    -- Risk metrics
    confidence_score Float32,
    risk_score Float32,
    competition_level UInt8,  -- 0-10 scale
    
    -- Status tracking
    status Enum8('detected' = 1, 'simulating' = 2, 'executing' = 3,
                 'completed' = 4, 'failed' = 5, 'expired' = 6),
    failure_reason Nullable(String),
    tx_hash Nullable(String),
    
    INDEX idx_tokens (token_a, token_b) TYPE bloom_filter GRANULARITY 1,
    INDEX idx_profit expected_profit_usd TYPE minmax GRANULARITY 1,
    INDEX idx_status status TYPE set(0) GRANULARITY 1
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(detected_at)
ORDER BY (detected_at, execution_id)
TTL detected_at + INTERVAL 60 DAY;

-- Network latency measurements
CREATE TABLE IF NOT EXISTS network_latency (
    timestamp DateTime64(3) DEFAULT now64(3),
    endpoint String,
    endpoint_type Enum8('rpc' = 1, 'websocket' = 2, 'rest_api' = 3, 
                        'grpc' = 4, 'jito' = 5, 'validator' = 6),
    
    -- Latency metrics (microseconds for precision)
    ping_us UInt32,
    connect_time_us UInt32,
    handshake_time_us UInt32,
    request_time_us UInt32,
    response_time_us UInt32,
    total_time_us UInt32,
    
    -- Throughput
    bytes_sent UInt64,
    bytes_received UInt64,
    
    -- Connection quality
    packet_loss_percent Float32,
    jitter_us UInt32,
    
    -- Status
    success Bool,
    error_code Nullable(UInt16),
    error_message Nullable(String),
    
    INDEX idx_endpoint endpoint TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 3600
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (endpoint_type, endpoint, timestamp)
TTL timestamp + INTERVAL 7 DAY
SETTINGS index_granularity = 8192;

-- Bandit algorithm performance rollup
CREATE TABLE IF NOT EXISTS bandit_performance_hourly (
    hour DateTime DEFAULT toStartOfHour(now()),
    module String,
    policy String,
    route String,
    
    -- Aggregated metrics
    total_pulls UInt64,
    successful_pulls UInt64,
    total_payoff Float64,
    avg_payoff Float64,
    max_payoff Float64,
    min_payoff Float64,
    std_dev_payoff Float64,
    
    -- Landing rates
    landing_rate Float32,
    predicted_landing_rate Float32,
    landing_rate_error Float32,
    
    -- Economic metrics
    total_tips_sol Float64,
    total_profit_sol Float64,
    roi_percent Float32,
    
    -- Performance metrics
    avg_latency_ms Float32,
    p95_latency_ms Float32,
    p99_latency_ms Float32,
    
    INDEX idx_hour hour TYPE minmax GRANULARITY 1,
    INDEX idx_module_policy (module, policy) TYPE bloom_filter GRANULARITY 1
) ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(hour)
ORDER BY (hour, module, policy, route)
TTL hour + INTERVAL 180 DAY;

-- Materialized view for real-time bandit performance
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_bandit_performance_realtime
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMMDD(window_start)
ORDER BY (window_start, module, policy, route)
AS SELECT
    tumbleStart(ts, toIntervalMinute(5)) AS window_start,
    tumbleEnd(ts, toIntervalMinute(5)) AS window_end,
    module,
    policy,
    route,
    count() AS pulls,
    sumIf(1, landed > 0) AS successful_pulls,
    sum(payoff) AS total_payoff,
    avg(payoff) AS avg_payoff,
    max(payoff) AS max_payoff,
    min(payoff) AS min_payoff,
    stddevPop(payoff) AS std_dev_payoff,
    avg(landed) AS landing_rate,
    avg(p_land_est) AS predicted_landing_rate,
    avg(abs(landed - p_land_est)) AS landing_rate_error,
    sum(tip_sol) AS total_tips_sol,
    sum(ev_sol) AS total_ev_sol
FROM bandit_events
GROUP BY window_start, window_end, module, policy, route;

-- Transaction mempool analysis
CREATE TABLE IF NOT EXISTS mempool_analysis (
    timestamp DateTime64(3) DEFAULT now64(3),
    
    -- Mempool state
    total_transactions UInt32,
    total_value_sol Float64,
    avg_gas_price_gwei Float32,
    median_gas_price_gwei Float32,
    p95_gas_price_gwei Float32,
    
    -- MEV opportunities
    sandwich_opportunities UInt32,
    arbitrage_opportunities UInt32,
    liquidation_opportunities UInt32,
    
    -- Competition metrics
    active_searchers UInt16,
    bundle_competition_rate Float32,
    avg_bundle_size UInt16,
    
    -- Network conditions
    block_space_utilization Float32,
    congestion_level UInt8,  -- 0-10 scale
    estimated_wait_blocks Float32,
    
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 3600
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY timestamp
TTL toDateTime(timestamp) + INTERVAL 30 DAY;

-- Risk metrics tracking
CREATE TABLE IF NOT EXISTS risk_metrics (
    timestamp DateTime64(3) DEFAULT now64(3),
    
    -- Position risk
    total_exposure_usd Float64,
    max_position_size_usd Float64,
    open_positions UInt32,
    
    -- Market risk
    volatility_1h Float32,
    volatility_24h Float32,
    max_drawdown_percent Float32,
    
    -- Operational risk
    failed_transactions UInt32,
    reverted_bundles UInt32,
    slippage_incidents UInt32,
    
    -- Liquidity risk
    available_liquidity_usd Float64,
    liquidity_utilization_percent Float32,
    
    -- Composite scores
    overall_risk_score Float32,  -- 0-100
    risk_adjusted_return Float32,
    sharpe_ratio Float32,
    
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 3600
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY timestamp
TTL timestamp + INTERVAL 90 DAY;

-- Audit log for all operations
CREATE TABLE IF NOT EXISTS audit_log (
    event_id UUID DEFAULT generateUUIDv4(),
    timestamp DateTime64(6) DEFAULT now64(6),
    
    -- Event details
    event_type String,
    event_category Enum8('config' = 1, 'execution' = 2, 'risk' = 3, 
                         'security' = 4, 'performance' = 5),
    severity Enum8('debug' = 1, 'info' = 2, 'warning' = 3, 
                   'error' = 4, 'critical' = 5),
    
    -- Context
    component String,
    user Nullable(String),
    ip_address Nullable(IPv4),
    
    -- Event data
    message String,
    details Nullable(String),  -- JSON
    
    -- Tracking
    correlation_id Nullable(UUID),
    parent_event_id Nullable(UUID),
    
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 3600,
    INDEX idx_event_type event_type TYPE bloom_filter GRANULARITY 1,
    INDEX idx_severity severity TYPE set(0) GRANULARITY 1
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (timestamp, event_id)
TTL timestamp + INTERVAL 365 DAY;

-- Create dictionary for fast leader lookups
CREATE DICTIONARY IF NOT EXISTS leader_schedule (
    slot UInt64,
    leader String,
    stake UInt64,
    commission Float32
) PRIMARY KEY slot
SOURCE(CLICKHOUSE(
    HOST 'localhost'
    PORT 9000
    USER 'default'
    DB 'default'
    TABLE 'leader_schedule_source'
))
LIFETIME(MIN 300 MAX 3600)
LAYOUT(RANGE_HASHED());

-- Performance benchmarking results
CREATE TABLE IF NOT EXISTS performance_benchmarks (
    benchmark_id UUID DEFAULT generateUUIDv4(),
    timestamp DateTime64(3) DEFAULT now64(3),
    
    -- Test details
    test_name String,
    test_type Enum8('latency' = 1, 'throughput' = 2, 'stress' = 3, 
                    'endurance' = 4, 'spike' = 5),
    parameters String,  -- JSON
    
    -- Results
    duration_seconds Float32,
    operations_count UInt64,
    operations_per_second Float64,
    
    -- Latency percentiles (microseconds)
    latency_min_us UInt64,
    latency_p50_us UInt64,
    latency_p90_us UInt64,
    latency_p95_us UInt64,
    latency_p99_us UInt64,
    latency_max_us UInt64,
    
    -- Resource usage
    cpu_usage_avg Float32,
    memory_usage_avg_mb UInt32,
    network_throughput_mbps Float32,
    
    -- Status
    success Bool,
    errors_count UInt32,
    
    INDEX idx_test_name test_name TYPE bloom_filter GRANULARITY 1,
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, benchmark_id)
TTL timestamp + INTERVAL 180 DAY;