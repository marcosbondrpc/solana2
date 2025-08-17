-- Behavioral Metrics Query
-- Advanced behavioral analysis for entity profiling

WITH hourly_activity AS (
    SELECT 
        entity_id,
        toStartOfHour(timestamp) as hour,
        COUNT(*) as transactions,
        SUM(profit_sol) as hourly_profit,
        COUNT(DISTINCT wallet_address) as active_wallets,
        COUNT(DISTINCT pool_address) as active_pools,
        AVG(decision_latency_ms) as avg_latency,
        stddevPop(decision_latency_ms) as latency_stddev
    FROM mev_transactions
    WHERE timestamp >= now() - INTERVAL 7 DAY
    GROUP BY entity_id, hour
),

timing_patterns AS (
    SELECT 
        entity_id,
        -- Activity concentration
        entropy(toHour(timestamp)) as hour_entropy,
        entropy(toDayOfWeek(timestamp)) as day_entropy,
        
        -- Peak activity times
        argMax(toHour(timestamp), cnt) as peak_hour,
        argMax(toDayOfWeek(timestamp), cnt) as peak_day,
        
        -- Timing precision
        stddevPop(toUnixTimestamp(timestamp) % 400) as slot_timing_precision,
        
        -- Burst patterns
        MAX(burst_size) as max_burst_size,
        AVG(burst_size) as avg_burst_size
        
    FROM (
        SELECT 
            entity_id,
            timestamp,
            toHour(timestamp) as hr,
            COUNT(*) OVER (PARTITION BY entity_id, toHour(timestamp)) as cnt,
            countIf(
                timestamp BETWEEN timestamp - INTERVAL 1 SECOND 
                AND timestamp + INTERVAL 1 SECOND
            ) OVER (PARTITION BY entity_id ORDER BY timestamp) as burst_size
        FROM mev_transactions
        WHERE timestamp >= now() - INTERVAL 7 DAY
    )
    GROUP BY entity_id
),

wallet_behavior AS (
    SELECT 
        entity_id,
        COUNT(DISTINCT wallet_address) as total_wallets,
        
        -- Wallet rotation metrics
        AVG(wallet_lifetime_hours) as avg_wallet_lifetime,
        MIN(wallet_lifetime_hours) as min_wallet_lifetime,
        MAX(wallet_lifetime_hours) as max_wallet_lifetime,
        stddevPop(wallet_lifetime_hours) as wallet_lifetime_stddev,
        
        -- Wallet usage patterns
        AVG(txs_per_wallet) as avg_txs_per_wallet,
        MAX(txs_per_wallet) as max_txs_per_wallet,
        
        -- Wallet clustering
        COUNT(DISTINCT wallet_cluster) as wallet_clusters
        
    FROM (
        SELECT 
            entity_id,
            wallet_address,
            COUNT(*) as txs_per_wallet,
            dateDiff('hour', MIN(timestamp), MAX(timestamp)) as wallet_lifetime_hours,
            -- Simple clustering based on creation time
            toStartOfDay(MIN(timestamp)) as wallet_cluster
        FROM mev_transactions
        WHERE timestamp >= now() - INTERVAL 30 DAY
        GROUP BY entity_id, wallet_address
    )
    GROUP BY entity_id
),

pool_targeting AS (
    SELECT 
        entity_id,
        
        -- Pool diversity
        COUNT(DISTINCT pool_address) as unique_pools,
        entropy(pool_address) as pool_entropy,
        
        -- Pool concentration (Herfindahl index)
        SUM(pool_share * pool_share) as pool_hhi,
        
        -- Top pool dependency
        MAX(pool_share) as max_pool_concentration,
        
        -- Pool categories
        countIf(pool_tvl > 10000000) as large_pool_count,
        countIf(pool_tvl BETWEEN 1000000 AND 10000000) as medium_pool_count,
        countIf(pool_tvl < 1000000) as small_pool_count
        
    FROM (
        SELECT 
            entity_id,
            pool_address,
            COUNT(*) as pool_txs,
            COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY entity_id) as pool_share,
            any(pool_tvl) as pool_tvl
        FROM mev_transactions t
        LEFT JOIN pool_metadata p ON t.pool_address = p.address
        WHERE t.timestamp >= now() - INTERVAL 7 DAY
        GROUP BY entity_id, pool_address
    )
    GROUP BY entity_id
),

ordering_behavior AS (
    SELECT 
        entity_id,
        
        -- Transaction ordering patterns
        AVG(tx_position_in_block) as avg_tx_position,
        stddevPop(tx_position_in_block) as tx_position_stddev,
        
        -- Bundle patterns
        COUNT(DISTINCT bundle_hash) as unique_bundles,
        AVG(bundle_size) as avg_bundle_size,
        MAX(bundle_size) as max_bundle_size,
        
        -- Adjacency patterns
        SUM(is_adjacent_same_entity) as same_entity_adjacencies,
        SUM(is_adjacent_same_pool) as same_pool_adjacencies,
        
        -- Competition metrics
        AVG(competing_entities_in_block) as avg_competition,
        MAX(competing_entities_in_block) as max_competition
        
    FROM (
        SELECT 
            entity_id,
            transaction_index as tx_position_in_block,
            bundle_hash,
            COUNT(*) OVER (PARTITION BY bundle_hash) as bundle_size,
            
            -- Check adjacency
            CASE WHEN entity_id = LAG(entity_id) OVER w THEN 1 ELSE 0 END as is_adjacent_same_entity,
            CASE WHEN pool_address = LAG(pool_address) OVER w THEN 1 ELSE 0 END as is_adjacent_same_pool,
            
            -- Competition in same block
            COUNT(DISTINCT entity_id) OVER (PARTITION BY block_height) - 1 as competing_entities_in_block
            
        FROM mev_transactions
        WHERE timestamp >= now() - INTERVAL 7 DAY
        WINDOW w AS (PARTITION BY block_height ORDER BY transaction_index)
    )
    GROUP BY entity_id
),

latency_profile AS (
    SELECT 
        entity_id,
        
        -- Latency distribution
        quantile(0.25)(decision_latency_ms) as p25_latency,
        quantile(0.50)(decision_latency_ms) as p50_latency,
        quantile(0.75)(decision_latency_ms) as p75_latency,
        quantile(0.95)(decision_latency_ms) as p95_latency,
        quantile(0.99)(decision_latency_ms) as p99_latency,
        
        -- Latency consistency
        stddevPop(decision_latency_ms) as latency_stddev,
        p95_latency - p50_latency as latency_spread,
        
        -- Ultra-low latency detection
        countIf(decision_latency_ms < 10) / COUNT(*) as sub_10ms_ratio,
        countIf(decision_latency_ms < 50) / COUNT(*) as sub_50ms_ratio
        
    FROM mev_transactions
    WHERE timestamp >= now() - INTERVAL 7 DAY
        AND decision_latency_ms > 0
        AND decision_latency_ms < 10000
    GROUP BY entity_id
)

SELECT 
    t.entity_id,
    
    -- Timing behavior
    t.hour_entropy,
    t.day_entropy,
    t.peak_hour,
    t.peak_day,
    t.slot_timing_precision,
    t.max_burst_size,
    t.avg_burst_size,
    
    -- Wallet behavior
    w.total_wallets,
    w.avg_wallet_lifetime,
    w.wallet_lifetime_stddev,
    w.avg_txs_per_wallet,
    w.wallet_clusters,
    
    -- Pool targeting
    p.unique_pools,
    p.pool_entropy,
    p.pool_hhi,
    p.max_pool_concentration,
    
    -- Ordering behavior
    o.avg_tx_position,
    o.tx_position_stddev,
    o.avg_bundle_size,
    o.same_entity_adjacencies,
    o.avg_competition,
    
    -- Latency profile
    l.p50_latency,
    l.p95_latency,
    l.p99_latency,
    l.latency_spread,
    l.sub_10ms_ratio,
    
    -- Behavioral classification
    CASE
        WHEN l.p99_latency < 20 AND l.latency_spread < 10 THEN 'ULTRA_OPTIMIZED'
        WHEN l.p99_latency < 100 AND w.wallet_clusters > 5 THEN 'SOPHISTICATED'
        WHEN p.pool_entropy > 3 AND o.avg_bundle_size > 2 THEN 'ADVANCED'
        WHEN t.hour_entropy > 2 THEN 'ACTIVE'
        ELSE 'BASIC'
    END as behavioral_class,
    
    -- Anomaly scores
    CASE
        WHEN w.avg_wallet_lifetime < 24 AND w.total_wallets > 10 THEN 1 ELSE 0
    END as rapid_rotation_flag,
    
    CASE
        WHEN t.slot_timing_precision < 50 THEN 1 ELSE 0
    END as precise_timing_flag,
    
    CASE
        WHEN p.max_pool_concentration > 0.8 THEN 1 ELSE 0
    END as pool_concentration_flag
    
FROM timing_patterns t
LEFT JOIN wallet_behavior w ON t.entity_id = w.entity_id
LEFT JOIN pool_targeting p ON t.entity_id = p.entity_id
LEFT JOIN ordering_behavior o ON t.entity_id = o.entity_id
LEFT JOIN latency_profile l ON t.entity_id = l.entity_id
ORDER BY l.p50_latency ASC;