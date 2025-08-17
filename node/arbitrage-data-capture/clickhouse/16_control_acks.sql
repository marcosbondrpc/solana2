-- LEGENDARY Control ACKs Table with Hash-Chain Auditing
-- Provides cryptographic tamper-evidence for all control operations
-- Optimized for time-series queries and compliance auditing

-- Drop existing table if needed (be careful in production!)
-- DROP TABLE IF EXISTS control_acks;

-- Main control ACKs table with hash chaining
CREATE TABLE IF NOT EXISTS control_acks
(
    -- Timestamps
    dt DateTime DEFAULT now() COMMENT 'Insert time (server clock)',
    ts DateTime COMMENT 'ACK timestamp from agent',
    
    -- Core fields
    request_id String COMMENT 'Unique request identifier',
    module LowCardinality(String) COMMENT 'Module type: MEV, ARBITRAGE',
    agent_id LowCardinality(String) COMMENT 'Agent that processed the command',
    status LowCardinality(String) COMMENT 'Status: received, applied, rejected',
    reason String COMMENT 'Rejection reason or additional info',
    
    -- Hash chain fields for tamper evidence
    hash FixedString(32) COMMENT 'BLAKE3 hash of this ACK',
    prev_hash FixedString(32) COMMENT 'Hash of previous ACK in chain',
    sequence UInt64 COMMENT 'Sequence number in hash chain',
    verified Bool DEFAULT 1 COMMENT 'Whether hash chain was verified on insert',
    
    -- Indexes for common queries
    INDEX idx_request_id request_id TYPE bloom_filter(0.01) GRANULARITY 4,
    INDEX idx_agent_id agent_id TYPE set(100) GRANULARITY 2,
    INDEX idx_hash hash TYPE bloom_filter(0.001) GRANULARITY 1
)
ENGINE = MergeTree()
ORDER BY (ts, request_id, sequence)
PARTITION BY toYYYYMM(ts)
TTL ts + INTERVAL 400 DAY DELETE
SETTINGS 
    index_granularity = 8192,
    storage_policy = 'hot_cold',
    min_bytes_for_wide_part = 10485760,
    compress_marks = true,
    compress_primary_key = true;

-- Add column comments
ALTER TABLE control_acks
    COMMENT COLUMN dt 'Insert timestamp (local time)',
    COMMENT COLUMN ts 'Event timestamp from source',
    COMMENT COLUMN request_id 'Unique control request ID',
    COMMENT COLUMN module 'MEV or ARBITRAGE module',
    COMMENT COLUMN agent_id 'Processing agent identifier',
    COMMENT COLUMN status 'ACK status: received/applied/rejected',
    COMMENT COLUMN reason 'Additional context or rejection reason',
    COMMENT COLUMN hash 'BLAKE3 hash for this ACK',
    COMMENT COLUMN prev_hash 'Previous ACK hash for chain verification',
    COMMENT COLUMN sequence 'Monotonic sequence number',
    COMMENT COLUMN verified 'Hash chain verification status';

-- Materialized view for real-time statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS control_acks_stats_mv
ENGINE = SummingMergeTree()
ORDER BY (module, agent_id, status, toStartOfMinute(ts))
POPULATE AS
SELECT
    toStartOfMinute(ts) as minute,
    module,
    agent_id,
    status,
    count() as count,
    countIf(verified = 1) as verified_count,
    countIf(verified = 0) as failed_count
FROM control_acks
GROUP BY minute, module, agent_id, status;

-- View for hash chain verification
CREATE VIEW IF NOT EXISTS control_acks_chain_view AS
SELECT
    ts,
    request_id,
    module,
    agent_id,
    status,
    hex(hash) as hash_hex,
    hex(prev_hash) as prev_hash_hex,
    sequence,
    verified,
    lagInFrame(hash, 1) OVER (ORDER BY sequence) as expected_prev_hash
FROM control_acks
ORDER BY sequence;

-- Table for storing hash chain checkpoints (for recovery)
CREATE TABLE IF NOT EXISTS control_acks_checkpoints
(
    checkpoint_time DateTime DEFAULT now(),
    sequence UInt64,
    hash FixedString(32),
    agent_id LowCardinality(String),
    acks_count UInt64,
    verification_status Enum8('VALID' = 1, 'INVALID' = 2, 'UNKNOWN' = 3)
)
ENGINE = MergeTree()
ORDER BY (checkpoint_time, sequence)
TTL checkpoint_time + INTERVAL 90 DAY DELETE;

-- Function to verify hash chain integrity
-- Note: This is pseudo-code, actual implementation would be in application layer
-- CREATE FUNCTION verify_hash_chain AS (
--     sequence_start UInt64,
--     sequence_end UInt64
-- ) -> Bool
-- LANGUAGE SQL
-- AS $$
--     SELECT every(
--         hash = blake3(concat(
--             toString(request_id),
--             toString(module),
--             toString(agent_id),
--             toString(ts),
--             toString(status),
--             toString(reason),
--             prev_hash
--         ))
--     )
--     FROM control_acks
--     WHERE sequence BETWEEN sequence_start AND sequence_end
--     ORDER BY sequence
-- $$;

-- Aggregated view for compliance reporting
CREATE MATERIALIZED VIEW IF NOT EXISTS control_acks_daily_mv
ENGINE = SummingMergeTree()
ORDER BY (date, module, agent_id)
POPULATE AS
SELECT
    toDate(ts) as date,
    module,
    agent_id,
    countIf(status = 'received') as received_count,
    countIf(status = 'applied') as applied_count,
    countIf(status = 'rejected') as rejected_count,
    count() as total_count,
    countIf(verified = 0) as verification_failures,
    max(sequence) as max_sequence
FROM control_acks
GROUP BY date, module, agent_id;

-- Query examples for operations

-- 1. Find all ACKs for a specific request
-- SELECT * FROM control_acks WHERE request_id = 'req_123' ORDER BY ts;

-- 2. Verify hash chain for last hour
-- SELECT 
--     count() as total,
--     countIf(hash = expected_prev_hash) as valid_chain_count
-- FROM control_acks_chain_view
-- WHERE ts >= now() - INTERVAL 1 HOUR;

-- 3. Get rejection rate by module
-- SELECT
--     module,
--     countIf(status = 'rejected') / count() as rejection_rate,
--     count() as total_commands
-- FROM control_acks
-- WHERE ts >= now() - INTERVAL 1 DAY
-- GROUP BY module;

-- 4. Find potential tampering (broken chains)
-- SELECT
--     ts,
--     request_id,
--     sequence,
--     hex(hash) as hash,
--     hex(prev_hash) as declared_prev,
--     hex(expected_prev_hash) as actual_prev
-- FROM control_acks_chain_view
-- WHERE prev_hash != expected_prev_hash
--     AND sequence > 0
-- ORDER BY sequence
-- LIMIT 100;

-- 5. Agent performance metrics
-- SELECT
--     agent_id,
--     avg(dateDiff('millisecond', ts, dt)) as avg_latency_ms,
--     countIf(status = 'applied') / count() as success_rate,
--     count() as total_processed
-- FROM control_acks
-- WHERE ts >= now() - INTERVAL 1 HOUR
-- GROUP BY agent_id
-- ORDER BY total_processed DESC;

-- Create indexes for optimization
ALTER TABLE control_acks ADD INDEX IF NOT EXISTS idx_ts_module (ts, module) TYPE minmax GRANULARITY 4;
ALTER TABLE control_acks ADD INDEX IF NOT EXISTS idx_status (status) TYPE set(10) GRANULARITY 2;

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT ON control_acks TO 'ingestor';
-- GRANT SELECT ON control_acks_stats_mv TO 'readonly';
-- GRANT SELECT ON control_acks_chain_view TO 'auditor';

-- Table settings for production
ALTER TABLE control_acks
    MODIFY SETTING 
        merge_with_ttl_timeout = 86400,
        max_suspicious_broken_parts = 100,
        replicated_deduplication_window = 100,
        replicated_deduplication_window_seconds = 604800;