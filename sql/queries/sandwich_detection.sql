-- Sandwich Attack Detection Query
-- Identifies sandwich patterns with victim analysis

WITH transaction_sequences AS (
    SELECT 
        block_height,
        slot,
        transaction_index,
        transaction_hash,
        signer as attacker_wallet,
        entity_id as attacker_entity,
        pool_address,
        instruction_type,
        token_in,
        token_out,
        amount_in,
        amount_out,
        timestamp,
        -- Window functions for pattern detection
        LAG(transaction_hash, 1) OVER w as prev_tx,
        LAG(signer, 1) OVER w as prev_signer,
        LAG(entity_id, 1) OVER w as prev_entity,
        LAG(instruction_type, 1) OVER w as prev_instruction,
        LAG(pool_address, 1) OVER w as prev_pool,
        LAG(token_in, 1) OVER w as prev_token_in,
        LAG(token_out, 1) OVER w as prev_token_out,
        
        LEAD(transaction_hash, 1) OVER w as next_tx,
        LEAD(signer, 1) OVER w as next_signer,
        LEAD(entity_id, 1) OVER w as next_entity,
        LEAD(instruction_type, 1) OVER w as next_instruction,
        LEAD(pool_address, 1) OVER w as next_pool,
        LEAD(token_in, 1) OVER w as next_token_in,
        LEAD(token_out, 1) OVER w as next_token_out,
        LEAD(amount_out, 1) OVER w as victim_amount_out,
        LEAD(amount_in, 1) OVER w as victim_amount_in,
        
        LEAD(transaction_hash, 2) OVER w as next2_tx,
        LEAD(signer, 2) OVER w as next2_signer,
        LEAD(entity_id, 2) OVER w as next2_entity,
        LEAD(instruction_type, 2) OVER w as next2_instruction,
        LEAD(pool_address, 2) OVER w as next2_pool
    FROM mev_transactions
    WHERE timestamp >= now() - INTERVAL 24 HOUR
    WINDOW w AS (PARTITION BY block_height ORDER BY transaction_index)
),

sandwich_candidates AS (
    SELECT 
        block_height,
        slot,
        timestamp,
        -- Attacker transactions
        transaction_hash as frontrun_tx,
        next2_tx as backrun_tx,
        attacker_entity,
        attacker_wallet,
        -- Victim transaction
        next_tx as victim_tx,
        next_signer as victim_wallet,
        -- Pool and tokens
        pool_address,
        token_in,
        token_out,
        -- Amounts
        amount_in as frontrun_amount_in,
        amount_out as frontrun_amount_out,
        victim_amount_in,
        victim_amount_out,
        -- Calculate victim impact
        CASE
            WHEN token_in = prev_token_out AND token_out = prev_token_in THEN
                -- Same pool, opposite direction
                (victim_amount_out / victim_amount_in) - 
                (amount_out / amount_in)
            ELSE 0
        END as slippage_impact,
        -- Pattern confidence
        CASE
            WHEN attacker_entity = next2_entity 
                AND pool_address = next_pool 
                AND pool_address = next2_pool
                AND next_entity != attacker_entity
                AND token_in = next_token_in
                AND token_out = next_token_out
            THEN 1
            ELSE 0
        END as is_sandwich
    FROM transaction_sequences
    WHERE 
        -- Basic sandwich pattern
        attacker_entity = next2_entity  -- Same attacker
        AND attacker_entity != next_entity  -- Different middle entity (victim)
        AND pool_address = next_pool  -- Same pool for all 3
        AND pool_address = next2_pool
),

sandwich_metrics AS (
    SELECT 
        attacker_entity,
        COUNT(*) as sandwich_count,
        COUNT(DISTINCT victim_wallet) as unique_victims,
        COUNT(DISTINCT pool_address) as unique_pools_targeted,
        COUNT(DISTINCT DATE(timestamp)) as active_days,
        
        -- Timing metrics
        AVG(dateDiff('millisecond', 
            toDateTime(block_height / 2),  -- Approximate block time
            timestamp)) as avg_execution_time_ms,
        
        -- Profit metrics
        SUM(slippage_impact * victim_amount_in / 1e9) as estimated_profit_sol,
        AVG(slippage_impact * victim_amount_in / 1e9) as avg_profit_per_sandwich,
        MAX(slippage_impact * victim_amount_in / 1e9) as max_single_profit,
        
        -- Success metrics
        COUNT(DISTINCT block_height) / COUNT(*) as block_efficiency,
        
        -- Victim impact
        SUM(slippage_impact) as total_slippage_caused,
        AVG(slippage_impact) as avg_slippage_per_victim
        
    FROM sandwich_candidates
    WHERE is_sandwich = 1
    GROUP BY attacker_entity
)

SELECT 
    s.*,
    -- Additional context
    e.total_transactions,
    e.landing_rate,
    s.sandwich_count / e.total_transactions as sandwich_ratio,
    -- Rank entities
    ROW_NUMBER() OVER (ORDER BY s.sandwich_count DESC) as rank_by_volume,
    ROW_NUMBER() OVER (ORDER BY s.estimated_profit_sol DESC) as rank_by_profit,
    ROW_NUMBER() OVER (ORDER BY s.avg_profit_per_sandwich DESC) as rank_by_efficiency
FROM sandwich_metrics s
JOIN (
    SELECT 
        entity_id,
        COUNT(*) as total_transactions,
        countIf(landed = 1) / COUNT(*) as landing_rate
    FROM mev_transactions
    WHERE timestamp >= now() - INTERVAL 24 HOUR
    GROUP BY entity_id
) e ON s.attacker_entity = e.entity_id
ORDER BY s.sandwich_count DESC;