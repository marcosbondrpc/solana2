-- Elite MEV Detection Schema for Solana
-- Designed for ultra-low-latency behavioral analysis
-- Target: 200k+ events/sec ingestion, sub-slot detection

-- Raw transaction telemetry with slot-aligned timestamps
CREATE TABLE IF NOT EXISTS solana_transactions (
    -- Primary identifiers
    signature String,
    slot UInt64,
    block_time DateTime64(6),
    
    -- Transaction metadata
    fee UInt64,
    compute_units_consumed UInt64,
    pre_balance_lamports UInt64,
    post_balance_lamports UInt64,
    
    -- Instruction data
    program_ids Array(String),
    instruction_count UInt8,
    instruction_data String,  -- Base58 encoded
    
    -- Account involvement
    account_keys Array(String),
    account_writable_flags Array(UInt8),
    account_signer_flags Array(UInt8),
    
    -- MEV relevant fields
    is_vote_transaction UInt8,
    has_priority_fee UInt8,
    priority_fee_lamports UInt64,
    
    -- Ingestion metadata
    ingestion_timestamp DateTime64(9),
    source String,
    
    -- Indexes for ultra-fast queries
    INDEX idx_slot slot TYPE minmax GRANULARITY 1,
    INDEX idx_block_time block_time TYPE minmax GRANULARITY 1,
    INDEX idx_program_ids program_ids TYPE bloom_filter(0.01) GRANULARITY 1
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(block_time)
ORDER BY (slot, signature)
TTL block_time + INTERVAL 30 DAY
SETTINGS index_granularity = 8192;

-- Sandwich attack candidate detection
CREATE TABLE IF NOT EXISTS sandwich_candidates (
    -- Detection metadata
    detection_id String,
    detection_timestamp DateTime64(9),
    confidence_score Float32,
    
    -- Attack structure
    front_run_signature String,
    victim_signature String,
    back_run_signature String,
    
    -- Slot timing
    front_run_slot UInt64,
    victim_slot UInt64,
    back_run_slot UInt64,
    slot_distance UInt8,
    
    -- Financial metrics
    front_run_amount UInt64,
    victim_amount UInt64,
    back_run_amount UInt64,
    estimated_profit_lamports Int64,
    slippage_percentage Float32,
    
    -- Pool interaction
    pool_address String,
    pool_type String,  -- 'raydium', 'orca', 'pump'
    token_a_mint String,
    token_b_mint String,
    
    -- Attacker profile
    attacker_address String,
    victim_address String,
    
    -- Detection features
    feature_vector Array(Float32),
    detection_model String,
    model_version String,
    
    -- Decision DNA
    decision_dna String,  -- Ed25519 signed hash
    feature_hash String,
    
    INDEX idx_attacker attacker_address TYPE bloom_filter(0.001) GRANULARITY 1,
    INDEX idx_pool pool_address TYPE bloom_filter(0.001) GRANULARITY 1,
    INDEX idx_confidence confidence_score TYPE minmax GRANULARITY 1
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(detection_timestamp)
ORDER BY (detection_timestamp, confidence_score)
TTL detection_timestamp + INTERVAL 90 DAY;

-- Model scoring stream for real-time inference
CREATE TABLE IF NOT EXISTS model_scores (
    transaction_signature String,
    slot UInt64,
    score_timestamp DateTime64(9),
    
    -- Multi-model scores
    rule_based_score Float32,
    statistical_score Float32,
    gnn_score Float32,
    transformer_score Float32,
    ensemble_score Float32,
    
    -- Feature importance
    top_features Array(Tuple(String, Float32)),
    
    -- Classification results
    is_mev_candidate UInt8,
    mev_type String,  -- 'sandwich', 'arbitrage', 'liquidation', 'jit'
    confidence Float32,
    
    -- Latency tracking
    inference_latency_ms Float32,
    feature_extraction_ms Float32,
    
    INDEX idx_ensemble ensemble_score TYPE minmax GRANULARITY 1
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDDHH(score_timestamp)
ORDER BY (slot, transaction_signature)
TTL score_timestamp + INTERVAL 7 DAY;

-- Entity behavioral profiles
CREATE TABLE IF NOT EXISTS entity_profiles (
    entity_address String,
    profile_date Date,
    
    -- Activity metrics
    total_transactions UInt64,
    mev_transactions UInt64,
    success_rate Float32,
    
    -- Financial metrics
    total_volume_lamports UInt64,
    total_profit_lamports Int64,
    average_profit_per_tx Int64,
    max_single_profit Int64,
    
    -- Behavioral patterns
    preferred_pools Array(String),
    preferred_tokens Array(String),
    active_hours Array(UInt8),
    
    -- Attack style metrics
    attack_style String,  -- 'surgical', 'shotgun', 'mixed'
    avg_slippage_imposed Float32,
    victim_selection_pattern String,
    
    -- Risk metrics
    risk_appetite_score Float32,
    fee_posture String,  -- 'aggressive', 'moderate', 'conservative'
    avg_priority_fee UInt64,
    
    -- Temporal patterns
    uptime_percentage Float32,
    avg_txns_per_hour Float32,
    burst_pattern String,
    
    -- Clustering features
    behavioral_embedding Array(Float32),
    cluster_id UInt32,
    
    PRIMARY KEY (entity_address, profile_date)
) ENGINE = ReplacingMergeTree()
PARTITION BY toYYYYMM(profile_date)
ORDER BY (entity_address, profile_date);

-- Materialized view for real-time sandwich detection
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_sandwich_patterns
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMMDD(window_start)
ORDER BY (pool_address, window_start)
AS SELECT
    pool_address,
    toStartOfFiveMinute(detection_timestamp) as window_start,
    count() as sandwich_count,
    avg(confidence_score) as avg_confidence,
    max(estimated_profit_lamports) as max_profit,
    uniqExact(attacker_address) as unique_attackers,
    avg(slippage_percentage) as avg_slippage
FROM sandwich_candidates
GROUP BY pool_address, window_start;

-- Materialized view for entity activity aggregation
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_entity_activity
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMMDD(hour_bucket)
ORDER BY (entity_address, hour_bucket)
AS SELECT
    any(account_keys[1]) as entity_address,
    toStartOfHour(block_time) as hour_bucket,
    count() as transaction_count,
    sum(fee) as total_fees,
    sum(compute_units_consumed) as total_compute,
    max(priority_fee_lamports) as max_priority_fee,
    uniqExact(arrayJoin(program_ids)) as unique_programs
FROM solana_transactions
GROUP BY account_keys[1], hour_bucket;

-- High-frequency trading pattern detection
CREATE TABLE IF NOT EXISTS hft_patterns (
    entity_address String,
    detection_window DateTime64(3),
    
    -- Pattern metrics
    transactions_per_second Float32,
    unique_pools_targeted UInt32,
    avg_latency_between_txs_ms Float32,
    
    -- Execution quality
    success_rate Float32,
    revert_rate Float32,
    
    -- Strategy classification
    strategy_type String,
    confidence Float32,
    
    INDEX idx_entity entity_address TYPE bloom_filter(0.001) GRANULARITY 1
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(detection_window)
ORDER BY (detection_window, entity_address)
TTL detection_window + INTERVAL 14 DAY;

-- Cross-pool arbitrage detection
CREATE TABLE IF NOT EXISTS arbitrage_paths (
    path_id String,
    detection_timestamp DateTime64(9),
    
    -- Path structure
    pool_sequence Array(String),
    token_sequence Array(String),
    
    -- Execution details
    start_amount UInt64,
    end_amount UInt64,
    profit_lamports Int64,
    
    -- Timing
    execution_slots Array(UInt64),
    total_latency_ms Float32,
    
    -- Entity
    arbitrageur_address String,
    
    INDEX idx_arbitrageur arbitrageur_address TYPE bloom_filter(0.001) GRANULARITY 1
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(detection_timestamp)
ORDER BY (detection_timestamp, profit_lamports);

-- JIT liquidity detection
CREATE TABLE IF NOT EXISTS jit_liquidity_events (
    event_id String,
    detection_timestamp DateTime64(9),
    
    -- JIT pattern
    add_liquidity_tx String,
    swap_tx String,
    remove_liquidity_tx String,
    
    -- Timing
    total_duration_slots UInt32,
    
    -- Profit metrics
    lp_fees_earned UInt64,
    impermanent_loss Int64,
    net_profit Int64,
    
    -- Entity
    jit_provider String,
    pool_address String,
    
    INDEX idx_provider jit_provider TYPE bloom_filter(0.001) GRANULARITY 1
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(detection_timestamp)
ORDER BY (detection_timestamp, net_profit);

-- Decision DNA audit trail
CREATE TABLE IF NOT EXISTS decision_dna_log (
    decision_id String,
    timestamp DateTime64(9),
    
    -- Cryptographic proof
    decision_hash String,
    signature String,  -- Ed25519
    merkle_root String,
    
    -- Decision details
    detection_type String,
    entity_address String,
    confidence Float32,
    
    -- Features used
    feature_snapshot String,  -- JSON
    model_version String,
    
    PRIMARY KEY (decision_id)
) ENGINE = ReplacingMergeTree()
ORDER BY (timestamp, decision_id)
TTL timestamp + INTERVAL 365 DAY;