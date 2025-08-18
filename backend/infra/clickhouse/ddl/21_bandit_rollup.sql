-- ClickHouse Materialized Views for Multi-Armed Bandit Rollups
-- Optimized for MEV endpoint selection and performance tracking

-- Drop existing views if they exist
DROP TABLE IF EXISTS mev_bandit_metrics_mv;
DROP TABLE IF EXISTS mev_bandit_hourly_mv;
DROP TABLE IF EXISTS mev_endpoint_performance_mv;

-- Raw bandit events table (if not exists)
CREATE TABLE IF NOT EXISTS mev_bandit_events (
    timestamp DateTime64(3) CODEC(DoubleDelta),
    slot UInt64 CODEC(Delta),
    endpoint String CODEC(ZSTD(3)),
    action Enum8('explore' = 1, 'exploit' = 2) CODEC(ZSTD(1)),
    reward Float32 CODEC(Gorilla),
    latency_ms UInt32 CODEC(T64),
    success Bool CODEC(ZSTD(1)),
    tip_amount UInt64 CODEC(T64),
    profit_lamports Int64 CODEC(T64),
    bundle_size UInt8 CODEC(ZSTD(1)),
    priority_fee UInt64 CODEC(T64),
    
    -- Bandit state at decision time
    ucb_score Float32 CODEC(Gorilla),
    exploration_count UInt32 CODEC(T64),
    exploitation_count UInt32 CODEC(T64),
    temperature Float32 CODEC(Gorilla),
    
    -- Network conditions
    network_congestion Float32 CODEC(Gorilla),
    fork_detected Bool CODEC(ZSTD(1)),
    leader_id String CODEC(ZSTD(3)),
    
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 8192,
    INDEX idx_endpoint endpoint TYPE bloom_filter(0.01) GRANULARITY 4096,
    INDEX idx_slot slot TYPE minmax GRANULARITY 4096
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (endpoint, timestamp, slot)
TTL timestamp + INTERVAL 90 DAY
SETTINGS index_granularity = 8192;

-- Real-time bandit metrics (1-minute aggregation)
CREATE MATERIALIZED VIEW mev_bandit_metrics_mv
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMMDD(minute)
ORDER BY (endpoint, minute)
TTL minute + INTERVAL 7 DAY
AS SELECT
    toStartOfMinute(timestamp) AS minute,
    endpoint,
    
    -- Performance metrics
    count() AS total_selections,
    sumIf(1, action = 'explore') AS exploration_count,
    sumIf(1, action = 'exploit') AS exploitation_count,
    
    -- Success metrics
    avg(success) AS success_rate,
    sumIf(1, success = true) AS successful_sends,
    sumIf(1, success = false) AS failed_sends,
    
    -- Latency statistics
    quantilesTDigest(0.5, 0.9, 0.95, 0.99)(latency_ms) AS latency_percentiles,
    avg(latency_ms) AS avg_latency_ms,
    min(latency_ms) AS min_latency_ms,
    max(latency_ms) AS max_latency_ms,
    
    -- Profit metrics
    sum(profit_lamports) AS total_profit,
    avg(profit_lamports) AS avg_profit,
    max(profit_lamports) AS max_profit,
    sumIf(profit_lamports, profit_lamports > 0) AS positive_profit_sum,
    
    -- Tip and fee analysis
    avg(tip_amount) AS avg_tip,
    sum(tip_amount) AS total_tips_spent,
    avg(priority_fee) AS avg_priority_fee,
    
    -- UCB metrics
    avg(ucb_score) AS avg_ucb_score,
    max(ucb_score) AS max_ucb_score,
    avg(temperature) AS avg_temperature,
    
    -- Network conditions
    avg(network_congestion) AS avg_congestion,
    sumIf(1, fork_detected = true) AS fork_count,
    
    -- Efficiency metrics
    sum(profit_lamports) / (sum(tip_amount) + sum(priority_fee) + 1) AS roi,
    sumIf(profit_lamports, success = true) / (sumIf(tip_amount + priority_fee, success = true) + 1) AS effective_roi
    
FROM mev_bandit_events
GROUP BY minute, endpoint;

-- Hourly rollup for long-term analysis
CREATE MATERIALIZED VIEW mev_bandit_hourly_mv
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(hour)
ORDER BY (endpoint, hour)
TTL hour + INTERVAL 180 DAY
AS SELECT
    toStartOfHour(minute) AS hour,
    endpoint,
    
    -- Aggregated counts
    sum(total_selections) AS total_selections,
    sum(exploration_count) AS exploration_count,
    sum(exploitation_count) AS exploitation_count,
    
    -- Success metrics (weighted average)
    sum(successful_sends) / sum(total_selections) AS success_rate,
    sum(successful_sends) AS successful_sends,
    sum(failed_sends) AS failed_sends,
    
    -- Latency (use pre-computed percentiles)
    avgArray(latency_percentiles) AS latency_percentiles,
    avg(avg_latency_ms) AS avg_latency_ms,
    min(min_latency_ms) AS min_latency_ms,
    max(max_latency_ms) AS max_latency_ms,
    
    -- Profit rollup
    sum(total_profit) AS total_profit,
    avg(avg_profit) AS avg_profit,
    max(max_profit) AS max_profit,
    sum(positive_profit_sum) AS positive_profit_sum,
    
    -- Cost analysis
    sum(total_tips_spent) AS total_tips_spent,
    avg(avg_tip) AS avg_tip,
    avg(avg_priority_fee) AS avg_priority_fee,
    
    -- Efficiency
    sum(total_profit) / (sum(total_tips_spent) + 1) AS hourly_roi,
    
    -- Network stats
    avg(avg_congestion) AS avg_congestion,
    sum(fork_count) AS fork_count
    
FROM mev_bandit_metrics_mv
GROUP BY hour, endpoint;

-- Endpoint performance comparison view
CREATE MATERIALIZED VIEW mev_endpoint_performance_mv
ENGINE = ReplacingMergeTree()
ORDER BY (evaluation_time, endpoint)
TTL evaluation_time + INTERVAL 30 DAY
AS SELECT
    now() AS evaluation_time,
    endpoint,
    
    -- 1-hour window metrics
    sumIf(total_selections, minute >= now() - INTERVAL 1 HOUR) AS selections_1h,
    avgIf(success_rate, minute >= now() - INTERVAL 1 HOUR) AS success_rate_1h,
    sumIf(total_profit, minute >= now() - INTERVAL 1 HOUR) AS profit_1h,
    avgIf(avg_latency_ms, minute >= now() - INTERVAL 1 HOUR) AS avg_latency_1h,
    
    -- 24-hour window metrics
    sumIf(total_selections, minute >= now() - INTERVAL 24 HOUR) AS selections_24h,
    avgIf(success_rate, minute >= now() - INTERVAL 24 HOUR) AS success_rate_24h,
    sumIf(total_profit, minute >= now() - INTERVAL 24 HOUR) AS profit_24h,
    avgIf(avg_latency_ms, minute >= now() - INTERVAL 24 HOUR) AS avg_latency_24h,
    
    -- 7-day window metrics
    sumIf(total_selections, minute >= now() - INTERVAL 7 DAY) AS selections_7d,
    avgIf(success_rate, minute >= now() - INTERVAL 7 DAY) AS success_rate_7d,
    sumIf(total_profit, minute >= now() - INTERVAL 7 DAY) AS profit_7d,
    avgIf(avg_latency_ms, minute >= now() - INTERVAL 7 DAY) AS avg_latency_7d,
    
    -- Composite score for ranking
    (
        avgIf(success_rate, minute >= now() - INTERVAL 1 HOUR) * 0.4 +
        (1000 / (avgIf(avg_latency_ms, minute >= now() - INTERVAL 1 HOUR) + 1)) * 0.3 +
        (sumIf(total_profit, minute >= now() - INTERVAL 1 HOUR) / 1000000) * 0.3
    ) AS performance_score
    
FROM mev_bandit_metrics_mv
GROUP BY endpoint;

-- Helper function for querying best endpoints
CREATE OR REPLACE FUNCTION get_best_endpoints(limit UInt8)
RETURNS Array(Tuple(String, Float64))
AS $$
    SELECT arraySlice(
        arraySort(x -> -x.2,
            groupArray((endpoint, performance_score))
        ), 1, limit
    )
    FROM mev_endpoint_performance_mv
    WHERE evaluation_time >= now() - INTERVAL 5 MINUTE
$$;

-- Create projection for ultra-fast endpoint lookups
ALTER TABLE mev_bandit_events ADD PROJECTION endpoint_success_projection (
    SELECT 
        endpoint,
        toStartOfMinute(timestamp) as minute,
        countIf(success = true) as success_count,
        count() as total_count,
        avg(latency_ms) as avg_latency
    GROUP BY endpoint, minute
);

-- Optimize table for production
OPTIMIZE TABLE mev_bandit_events FINAL;
OPTIMIZE TABLE mev_bandit_metrics_mv FINAL;
OPTIMIZE TABLE mev_bandit_hourly_mv FINAL;