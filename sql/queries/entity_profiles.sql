-- Entity Behavioral Profiles Query
-- Comprehensive profiling of MEV entities with statistical metrics

WITH entity_metrics AS (
    SELECT 
        entity_id,
        -- Volume metrics
        COUNT(*) as total_transactions,
        COUNT(DISTINCT toDate(timestamp)) as active_days,
        COUNT(DISTINCT wallet_address) as unique_wallets,
        COUNT(DISTINCT pool_address) as unique_pools,
        
        -- MEV strategy breakdown
        countIf(is_sandwich = 1) as sandwich_count,
        countIf(is_backrun = 1) as backrun_count,
        countIf(is_frontrun = 1) as frontrun_count,
        countIf(is_atomic_arb = 1) as atomic_arb_count,
        
        -- Performance metrics
        countIf(landed = 1) / COUNT(*) as landing_rate,
        quantile(0.50)(decision_latency_ms) as p50_latency,
        quantile(0.95)(decision_latency_ms) as p95_latency,
        quantile(0.99)(decision_latency_ms) as p99_latency,
        
        -- Economic metrics
        SUM(profit_sol) as total_profit_sol,
        AVG(profit_sol) as avg_profit_sol,
        SUM(tip_lamports) / 1e9 as total_tips_sol,
        SUM(priority_fee_lamports) / 1e9 as total_fees_sol,
        
        -- Timing patterns
        AVG(toHour(timestamp)) as avg_hour_of_day,
        stddevPop(toHour(timestamp)) as hour_stddev,
        
        -- First and last seen
        MIN(timestamp) as first_seen,
        MAX(timestamp) as last_seen
    FROM mev_transactions
    WHERE timestamp >= now() - INTERVAL 30 DAY
    GROUP BY entity_id
),

wallet_rotation AS (
    SELECT 
        entity_id,
        COUNT(DISTINCT wallet_address) as rotating_wallets,
        AVG(wallet_lifetime_hours) as avg_wallet_lifetime,
        MAX(wallet_lifetime_hours) as max_wallet_lifetime
    FROM (
        SELECT 
            entity_id,
            wallet_address,
            dateDiff('hour', MIN(timestamp), MAX(timestamp)) as wallet_lifetime_hours
        FROM mev_transactions
        WHERE timestamp >= now() - INTERVAL 30 DAY
        GROUP BY entity_id, wallet_address
    )
    GROUP BY entity_id
),

pool_preference AS (
    SELECT 
        entity_id,
        topK(10)(pool_address) as top_10_pools,
        entropy(pool_address) as pool_diversity_entropy
    FROM mev_transactions
    WHERE timestamp >= now() - INTERVAL 30 DAY
    GROUP BY entity_id
)

SELECT 
    m.*,
    w.rotating_wallets,
    w.avg_wallet_lifetime,
    w.max_wallet_lifetime,
    p.top_10_pools,
    p.pool_diversity_entropy,
    -- Classification
    CASE 
        WHEN m.total_profit_sol > 1000 AND m.landing_rate > 0.65 THEN 'ELITE'
        WHEN m.total_profit_sol > 100 AND m.landing_rate > 0.5 THEN 'PROFESSIONAL'
        WHEN m.total_profit_sol > 10 THEN 'ACTIVE'
        ELSE 'AMATEUR'
    END as performance_tier,
    -- Strategy profile
    CASE
        WHEN m.sandwich_count > m.backrun_count * 2 THEN 'SANDWICH_FOCUSED'
        WHEN m.backrun_count > m.sandwich_count * 2 THEN 'BACKRUN_FOCUSED'
        WHEN m.atomic_arb_count > (m.sandwich_count + m.backrun_count) THEN 'ARBITRAGE_FOCUSED'
        ELSE 'MIXED_STRATEGY'
    END as strategy_profile
FROM entity_metrics m
LEFT JOIN wallet_rotation w ON m.entity_id = w.entity_id
LEFT JOIN pool_preference p ON m.entity_id = p.entity_id
ORDER BY m.total_profit_sol DESC;