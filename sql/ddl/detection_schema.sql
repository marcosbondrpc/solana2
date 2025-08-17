-- ClickHouse Schema for MEV Detection System
-- DETECTION-ONLY: Pure observability, no execution

-- Raw transaction telemetry with slot-aligned data
CREATE TABLE IF NOT EXISTS ch.raw_tx (
  ts DateTime64(9) CODEC(DoubleDelta, LZ4),
  slot UInt64 CODEC(DoubleDelta),
  sig FixedString(88),
  payer FixedString(44),
  fee UInt64 CODEC(T64),
  cu UInt32 CODEC(T64),
  priority_fee UInt64 CODEC(T64),
  programs Array(FixedString(44)),
  ix_kinds Array(UInt16),
  accounts Array(FixedString(44)),
  pool_keys Array(FixedString(44)),
  amount_in Float64 CODEC(Gorilla),
  amount_out Float64 CODEC(Gorilla),
  token_in FixedString(44),
  token_out FixedString(44),
  venue LowCardinality(String),
  -- Additional detection fields
  bundle_id Nullable(FixedString(88)),
  position_in_bundle UInt8,
  landing_status Enum8('landed'=1, 'failed'=2, 'pending'=3),
  revert_reason Nullable(String),
  -- Decision DNA
  dna_fingerprint FixedString(64),
  detection_model LowCardinality(String),
  INDEX idx_slot slot TYPE minmax GRANULARITY 4,
  INDEX idx_payer payer TYPE bloom_filter(0.01) GRANULARITY 8,
  INDEX idx_programs programs TYPE bloom_filter(0.01) GRANULARITY 8
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(ts)
ORDER BY (slot, ts, sig)
TTL ts + INTERVAL 90 DAY;

-- Sandwich attack candidates with evidence scoring
CREATE TABLE IF NOT EXISTS ch.candidates (
  detection_ts DateTime64(9),
  slot UInt64,
  victim_sig FixedString(88),
  attacker_a_sig FixedString(88),
  attacker_b_sig FixedString(88),
  attacker_addr FixedString(44),
  victim_addr FixedString(44),
  pool FixedString(44),
  d_ms Float64 CODEC(Gorilla),
  d_slots UInt16,
  slippage_victim Float64 CODEC(Gorilla),
  price_reversion Float64 CODEC(Gorilla),
  evidence Enum8('bracket'=1, 'slip_rebound'=2, 'both'=3, 'weak'=4),
  score_rule Float32 CODEC(Gorilla),
  score_gnn Float32 CODEC(Gorilla),
  score_transformer Float32 CODEC(Gorilla),
  ensemble_score Float32 CODEC(Gorilla),
  -- Attack characteristics
  attack_style Enum8('surgical'=1, 'shotgun'=2, 'adaptive'=3),
  victim_selection Enum8('retail'=1, 'whale'=2, 'bot'=3, 'unknown'=4),
  -- Economic impact
  victim_loss_sol Float64 CODEC(Gorilla),
  attacker_profit_sol Float64 CODEC(Gorilla),
  fee_burn_sol Float64 CODEC(Gorilla),
  -- Decision DNA
  dna_fingerprint FixedString(64),
  model_version String,
  INDEX idx_slot slot TYPE minmax GRANULARITY 4,
  INDEX idx_attacker attacker_addr TYPE bloom_filter(0.01) GRANULARITY 8
) ENGINE = ReplacingMergeTree(detection_ts)
PARTITION BY toYYYYMMDD(detection_ts)
ORDER BY (slot, victim_sig, attacker_addr)
TTL detection_ts + INTERVAL 180 DAY;

-- Entity behavioral profiles
CREATE TABLE IF NOT EXISTS ch.entity_profiles (
  entity_addr FixedString(44),
  profile_date Date,
  -- Activity metrics
  total_txs UInt64,
  sandwich_count UInt32,
  victim_count UInt32,
  unique_pools UInt32,
  unique_victims UInt32,
  -- Behavioral spectrum
  attack_style_surgical Float32 CODEC(Gorilla),
  attack_style_shotgun Float32 CODEC(Gorilla),
  victim_retail_ratio Float32 CODEC(Gorilla),
  victim_whale_ratio Float32 CODEC(Gorilla),
  risk_appetite Float32 CODEC(Gorilla),
  fee_aggressiveness Float32 CODEC(Gorilla),
  -- Timing patterns
  avg_response_ms Float64 CODEC(Gorilla),
  p50_response_ms Float64 CODEC(Gorilla),
  p99_response_ms Float64 CODEC(Gorilla),
  active_hours_bitmap UInt32,
  -- Economic impact
  total_extraction_sol Float64 CODEC(Gorilla),
  avg_profit_per_attack Float64 CODEC(Gorilla),
  total_fees_paid_sol Float64 CODEC(Gorilla),
  landing_rate Float32 CODEC(Gorilla),
  -- Fleet detection
  linked_wallets Array(FixedString(44)),
  cluster_id Nullable(UInt32),
  rotation_frequency Float32 CODEC(Gorilla)
) ENGINE = ReplacingMergeTree(profile_date)
PARTITION BY toYYYYMM(profile_date)
ORDER BY (entity_addr, profile_date)
TTL profile_date + INTERVAL 365 DAY;

-- Instruction sequence patterns for ML
CREATE TABLE IF NOT EXISTS ch.ix_sequences (
  ts DateTime64(9),
  slot UInt64,
  sig FixedString(88),
  seq_encoded Array(UInt16),
  seq_length UInt8,
  has_sandwich_pattern Bool,
  has_backrun_pattern Bool,
  has_liquidation_pattern Bool,
  -- Graph features
  graph_density Float32 CODEC(Gorilla),
  graph_centrality Float32 CODEC(Gorilla),
  graph_clustering Float32 CODEC(Gorilla),
  -- Transformer embeddings (compressed)
  embedding_vector Array(Float32) CODEC(Gorilla),
  INDEX idx_patterns (has_sandwich_pattern, has_backrun_pattern) TYPE set(0) GRANULARITY 4
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(ts)
ORDER BY (slot, ts)
TTL ts + INTERVAL 30 DAY;

-- Detection model performance metrics
CREATE TABLE IF NOT EXISTS ch.model_metrics (
  ts DateTime64(3),
  model_name LowCardinality(String),
  model_version String,
  -- Performance metrics
  roc_auc Float32,
  precision Float32,
  recall Float32,
  f1_score Float32,
  false_positive_rate Float32,
  -- Latency metrics
  inference_p50_us UInt32,
  inference_p95_us UInt32,
  inference_p99_us UInt32,
  -- Volume metrics
  predictions_count UInt64,
  true_positives UInt32,
  false_positives UInt32,
  true_negatives UInt32,
  false_negatives UInt32
) ENGINE = MergeTree()
ORDER BY (model_name, ts)
TTL ts + INTERVAL 90 DAY;

-- Materialized views for real-time analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS ch.mv_entity_7d
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMMDD(window_start)
ORDER BY (entity_addr, window_start)
AS SELECT
  attacker_addr as entity_addr,
  toStartOfInterval(detection_ts, INTERVAL 1 DAY) as window_start,
  count() as attack_count,
  sum(attacker_profit_sol) as total_profit,
  avg(ensemble_score) as avg_confidence,
  uniqExact(victim_addr) as unique_victims,
  uniqExact(pool) as unique_pools,
  avg(d_ms) as avg_response_time
FROM ch.candidates
WHERE detection_ts >= now() - INTERVAL 7 DAY
GROUP BY entity_addr, window_start;

-- Decision DNA audit trail
CREATE TABLE IF NOT EXISTS ch.decision_dna (
  ts DateTime64(9),
  dna_fingerprint FixedString(64),
  decision_type Enum8('detection'=1, 'classification'=2, 'scoring'=3),
  model_inputs String CODEC(ZSTD(3)),
  model_outputs String CODEC(ZSTD(3)),
  signature FixedString(128),
  merkle_anchor Nullable(FixedString(64)),
  INDEX idx_dna dna_fingerprint TYPE bloom_filter(0.001) GRANULARITY 1
) ENGINE = MergeTree()
ORDER BY (ts, dna_fingerprint)
TTL ts + INTERVAL 365 DAY;