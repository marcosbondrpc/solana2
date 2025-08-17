CREATE TABLE IF NOT EXISTS solana_rt.detections_1m
(
  bucket    DateTime,
  kind      LowCardinality(String),
  cnt       AggregateFunction(count),
  score_sum AggregateFunction(sum, Float32)
)
ENGINE = AggregatingMergeTree
ORDER BY (bucket, kind);

CREATE MATERIALIZED VIEW IF NOT EXISTS solana_rt.mv_detections_1m
TO solana_rt.detections_1m
AS
SELECT
  toStartOfMinute(ts) AS bucket,
  kind,
  countState() AS cnt,
  sumState(score) AS score_sum
FROM solana_rt.detections
GROUP BY bucket, kind;