CREATE TABLE IF NOT EXISTS solana_rt_dev.raw_tx
(
  ts        DateTime64(3, 'UTC') CODEC(Delta, ZSTD),
  slot      UInt64               CODEC(T64, LZ4),
  sig       FixedString(88)      CODEC(ZSTD),
  program   LowCardinality(String) CODEC(ZSTD),
  accounts  Array(String)        CODEC(ZSTD),
  status    LowCardinality(String) CODEC(ZSTD),
  fee       UInt64               CODEC(T64, LZ4),
  payload   String               CODEC(ZSTD)
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (slot, ts, sig);

CREATE TABLE IF NOT EXISTS solana_rt_dev.detections
(
  seq      UInt64,
  ts       DateTime64(3, 'UTC'),
  slot     UInt64,
  kind     LowCardinality(String),
  sig      FixedString(88),
  address  String,
  score    Float32,
  payload  String
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (ts, seq);